import os
import random
import itertools
import collections
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from utils.data_models import School, Station, Route as ProjectRoute, Solution
from utils.solution_utils import (
    simulate_route,
    print_solution_pretty, plot_routes_on_map,
    BUS_COUNT_WEIGHT, TOTAL_TIME_WEIGHT, ROUTE_TIME_WEIGHT, FAIRNESS_WEIGHT,
    MAX_ROUTE_MIN, BUS_CAPACITY, MAX_TOTAL_BUSES, route_cost
)
from utils.instance_generator import gen_instance_multi
from utils.geo_utils import travel_minutes
from lns_solver import run_lns

# --- 1. 資料結構定義 (Section 2) ---
@dataclass(frozen=True)
class Group:
    id: str
    origin: str
    school: str
    demand: int

@dataclass
class Segment:
    nodes: list
    picked: list[Group]
    time: float
    load: int
    first_group_dist: Optional[float] = None
    last_group_dist: Optional[float] = None

@dataclass
class InternalRoute:
    type: str
    v_type: str
    nodes: list
    picked: list[Group]
    cost: float
    travel_time: float
    groups_covered: set = field(default_factory=set)
    project_route: Optional[ProjectRoute] = None

# --- 2. 距離與參數設定 (範例) ---
# 實務上應從外部資料讀取
T_max = MAX_ROUTE_MIN
vehicle_caps = {"bus": BUS_CAPACITY}
MAX_SEGMENTS_PER_ENUM = 4000
MAX_SEGMENT_PICKUP_BRANCHES = 10
MAX_COMPOSITION_PAIRS = 120000

# --- 3. Bounded-Length Segment Enumeration (Section 3) ---
def enumerate_segments(start, end, candidate_stops, eligible_groups, capacity, dist_matrix):
    results = []
    path = [start]
    picked = []
    group_target_dist = {
        g: dist_matrix[g.origin][g.school]
        for stop_groups in eligible_groups.values()
        for g in stop_groups
    }

    def dfs(curr, time_acc, load_acc, remaining_stops, first_group_dist, current_group_dist):
        if len(results) >= MAX_SEGMENTS_PER_ENUM:
            return

        # 終止條件：嘗試走向終點
        total_time = time_acc + dist_matrix[curr][end]
        if total_time <= T_max and picked:
            results.append(
                Segment(
                    list(path + [end]),
                    list(picked),
                    total_time,
                    load_acc,
                    first_group_dist,
                    current_group_dist
                )
            )
            if len(results) >= MAX_SEGMENTS_PER_ENUM:
                return

        branch_count = 0
        ordered_stops = sorted(
            remaining_stops,
            key=lambda stop: dist_matrix[curr][stop] + dist_matrix[stop][end]
        )
        for stop in ordered_stops:
            d_ij = dist_matrix[curr][stop]
            if time_acc + d_ij + dist_matrix[stop][end] > T_max:
                continue
            
            groups_at_stop = eligible_groups.get(stop, [])
            # 枚舉該站點的所有 Group 子集組合 (Section 3.3)
            for r in range(1, len(groups_at_stop) + 1):
                for subset in itertools.combinations(groups_at_stop, r):
                    subset_dists = [group_target_dist[g] for g in subset]
                    if (
                        current_group_dist is not None
                        and any(d >= current_group_dist for d in subset_dists)
                    ):
                        continue

                    sub_demand = sum(g.demand for g in subset)
                    if load_acc + sub_demand > capacity:
                        continue

                    new_current_group_dist = min(subset_dists)
                    new_first_group_dist = (
                        first_group_dist
                        if first_group_dist is not None
                        else new_current_group_dist
                    )
                    
                    # Backtracking
                    path.append(stop)
                    picked.extend(subset)
                    remaining_stops.remove(stop)
                    
                    dfs(
                        stop,
                        time_acc + d_ij,
                        load_acc + sub_demand,
                        remaining_stops,
                        new_first_group_dist,
                        new_current_group_dist
                    )
                    
                    remaining_stops.add(stop)
                    for _ in subset: picked.pop()
                    path.pop()

                    branch_count += 1
                    if (
                        len(results) >= MAX_SEGMENTS_PER_ENUM
                        or branch_count >= MAX_SEGMENT_PICKUP_BRANCHES
                    ):
                        return

    dfs(start, 0, 0, set(candidate_stops), None, None)
    return results

# --- 4. Cost 與 Fairness 計算 (Section 6) ---
def compute_route_cost(nodes, picked, dist_matrix):
    # 計算各站到達時間
    arrival_times = {nodes[0]: 0.0}
    curr_t = 0.0
    for i in range(1, len(nodes)):
        curr_t += dist_matrix[nodes[i-1]][nodes[i]]
        arrival_times[nodes[i]] = curr_t
    
    # Riding time q_s
    q = {g: arrival_times[g.school] - arrival_times[g.origin] for g in picked}
    
    # Fairness Penalty F_r (Section 6.3)
    F = 0.0
    for s, t in itertools.permutations(picked, 2):
        if s.school == t.school:
            d_s = dist_matrix[s.origin][s.school]
            d_t = dist_matrix[t.origin][t.school]
            if d_s < d_t and q[s] > q[t]:
                F += (s.demand * t.demand) / (d_s * d_t)
    
    travel_time = curr_t
    cost = BUS_COUNT_WEIGHT + (TOTAL_TIME_WEIGHT * travel_time) + (ROUTE_TIME_WEIGHT * sum(g.demand * q[g] for g in picked)) + (FAIRNESS_WEIGHT * F)
    return cost, travel_time

# --- 5. 主程式 Pipeline ---
def solve_sbrp(groups: List[Group], stops: List[str], dist_matrix: Dict, 
               school_map: Dict[str, int], schools: List[School], stations: List[Station]):
    route_pool = []
    route_pool_keys = set()
    st_dict = {s.idx: s for s in stations}
    group_by_station_school = {
        (int(g.origin), school_map[g.school]): g
        for g in groups
    }

    def add_internal_route(route: InternalRoute) -> bool:
        key = frozenset(route.groups_covered)
        if not key or key in route_pool_keys:
            return False
        route_pool.append(route)
        route_pool_keys.add(key)
        return True

    def add_segment_route(route_type: str, v_type: str, segment: Segment):
        cost, t_time = compute_route_cost(segment.nodes, segment.picked, dist_matrix)
        add_internal_route(
            InternalRoute(
                route_type,
                v_type,
                segment.nodes,
                segment.picked,
                cost,
                t_time,
                {g.id for g in segment.picked}
            )
        )

    def add_project_route(route_type: str, route: ProjectRoute):
        covered = set()
        for st_idx, sch_idx, _, _ in route.pickup_detail:
            g = group_by_station_school.get((st_idx, sch_idx))
            if g is not None:
                covered.add(g.id)
        add_internal_route(
            InternalRoute(
                route_type,
                "bus",
                [],
                [],
                route_cost(route),
                route.minutes,
                covered,
                route
            )
        )

    # 先加入一組完整啟發式解，讓 set-partitioning master 至少有可行欄位。
    # 原本純枚舉池可能漏掉大量 group，導致 cover_g == 1 無解。
    lns_sol = run_lns(schools, stations, print_log=False)
    if lns_sol.feasible and len(lns_sol.routes) <= MAX_TOTAL_BUSES:
        for route in lns_sol.routes:
            add_project_route("lns", route)
    
    for v_type, v_cap in vehicle_caps.items():
        # Type (a) & (b): 單校枚舉
        groups_a = {s: [g for g in groups if g.origin == s and g.school == 'A'] for s in stops}
        seg_a = enumerate_segments('0', 'A', stops, groups_a, v_cap, dist_matrix)
        for s in seg_a:
            add_segment_route('a', v_type, s)

        groups_b = {s: [g for g in groups if g.origin == s and g.school == 'B'] for s in stops}
        seg_b = enumerate_segments('0', 'B', stops, groups_b, v_cap, dist_matrix)
        for s in seg_b:
            add_segment_route('b', v_type, s)

        # Type (c) Composition: 0 -> VP_B -> B -> VP_A -> A
        seg_0_B = enumerate_segments('0', 'B', stops, groups_b, v_cap, dist_matrix)
        seg_B_A = enumerate_segments('B', 'A', stops, groups_a, v_cap, dist_matrix)
        
        composition_pairs_checked = 0
        for s1 in seg_0_B:
            for s2 in seg_B_A:
                composition_pairs_checked += 1
                if composition_pairs_checked > MAX_COMPOSITION_PAIRS:
                    break
                # 拼接檢查 (Section 5.3)
                if set(s1.nodes[1:-1]) & set(s2.nodes[1:-1]): continue
                if set(s1.picked) & set(s2.picked): continue
                if (
                    s1.last_group_dist is not None
                    and s2.first_group_dist is not None
                    and s2.first_group_dist >= s1.last_group_dist
                ): continue
                if s1.time + s2.time > T_max: continue
                # s1 的 B 校學生已在 B 下車，s2 才接 A 校學生；兩段載重不會重疊。
                if max(s1.load, s2.load) > v_cap: continue
                
                full_nodes = s1.nodes + s2.nodes[1:]
                full_picked = s1.picked + s2.picked
                cost, t_time = compute_route_cost(full_nodes, full_picked, dist_matrix)
                add_internal_route(InternalRoute('c', v_type, full_nodes, full_picked, cost, t_time, {g.id for g in full_picked}))
            if composition_pairs_checked > MAX_COMPOSITION_PAIRS:
                break

    uncovered = [g.id for g in groups if not any(g.id in r.groups_covered for r in route_pool)]
    if uncovered:
        print(f"[DW2] 警告：route pool 仍有 {len(uncovered)} 個 group 未覆蓋，前幾個: {uncovered[:10]}")

    # --- 6. Gurobi IP Master (Section 7) ---
    model = gp.Model("SBRP_Master")
    x = model.addVars(len(route_pool), vtype=GRB.BINARY, name="x")
    
    # Obj: Min Total Cost
    model.setObjective(gp.quicksum(route_pool[r].cost * x[r] for r in range(len(route_pool))), GRB.MINIMIZE)
    
    # Constr 1: 每組學生必須被覆蓋一次 (Set Partitioning)
    for g in groups:
        model.addConstr(
            gp.quicksum(x[r] for r in range(len(route_pool)) if g.id in route_pool[r].groups_covered) == 1,
            name=f"cover_{g.id}"
        )
    
    # Constr 2: 車輛數限制
    model.addConstr(gp.quicksum(x[r] for r in range(len(route_pool))) <= MAX_TOTAL_BUSES, name="vehicle_limit")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        selected_internal = [route_pool[r] for r in range(len(route_pool)) if x[r].X > 0.5]
        
        final_routes = []
        for ir in selected_internal:
            if ir.project_route is not None:
                final_routes.append(ir.project_route)
                continue

            events = []
            # 將 InternalRoute 轉換為 ProjectRoute 的事件序列
            for node in ir.nodes[1:]: # 跳過虛擬起點 '0'
                if node == 'A' or node == 'B':
                    events.append(('drop', school_map[node]))
                else:
                    st_idx = int(node)
                    # 找出在此站點被接走的群組
                    st_groups = [g for g in ir.picked if g.origin == node]
                    if st_groups:
                        demands = {}
                        for g in st_groups:
                            s_idx = school_map[g.school]
                            demands[s_idx] = demands.get(s_idx, 0) + g.demand
                        events.append(('pickup', (st_idx, demands)))
            
            pr = ProjectRoute(events=events)
            pr.minutes, pr.in_vehicle_minutes, pr.pickup_detail, pr.fairness_penalty = simulate_route(
                pr, schools, st_dict
            )
            final_routes.append(pr)

        total_min = sum(r.minutes for r in final_routes)
        total_ivm = sum(r.in_vehicle_minutes for r in final_routes)
        total_fair = sum(r.fairness_penalty for r in final_routes)
        total_served = sum(sum(c for _, _, c, _ in r.pickup_detail) for r in final_routes)
        
        return Solution(final_routes, total_min, total_ivm, total_fair, total_served, True)
    
    return Solution([], 0, 0, 0, 0, False)

def run_dantzig_wolfe_2(schools: List[School], stations: List[Station]) -> Solution:
    """整合適配器：將專案資料結構轉換為此 Solver 格式並執行"""
    if len(schools) < 1: return Solution([], 0, 0, 0, 0, False)
    
    # 建立學校映射 (最多處理兩所)
    school_map = {'A': schools[0].idx}
    if len(schools) > 1: school_map['B'] = schools[1].idx
    rev_map = {v: k for k, v in school_map.items()}
    
    # 建立群組
    groups = []
    for st in stations:
        for sch_idx, count in st.demands.items():
            if sch_idx in rev_map:
                groups.append(Group(id=f"G_{st.idx}_{sch_idx}", origin=str(st.idx), school=rev_map[sch_idx], demand=count))
    
    # 建立距離矩陣
    stops = [str(s.idx) for s in stations]
    all_keys = ['0', 'A', 'B'] + stops
    node_to_coord = {
        '0': schools[0].coord, 'A': schools[0].coord,
        'B': schools[1].coord if len(schools) > 1 else schools[0].coord
    }
    for s in stations: node_to_coord[str(s.idx)] = s.coord
    
    dist_matrix = collections.defaultdict(dict)
    for u, v in itertools.product(all_keys, all_keys):
        dist_matrix[u][v] = travel_minutes(node_to_coord[u], node_to_coord[v])
        
    return solve_sbrp(groups, stops, dist_matrix, school_map, schools, stations)

if __name__ == "__main__":
    random.seed(42)
    schools, stations = gen_instance_multi()
    best_sol = run_dantzig_wolfe_2(schools, stations)
    if best_sol.feasible:
        print_solution_pretty(best_sol, stations, schools)
        m = plot_routes_on_map(best_sol, stations, schools, title="Dantzig-Wolfe 2 Optimized")
        os.makedirs("./data", exist_ok=True)
        m.save("./data/dw2_routes.html")
        print(f"優化完成，結果已存至 ./data/dw2_routes.html")
