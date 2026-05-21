import os
import random
from tqdm import tqdm
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    simulate_route, build_solution,
    BUS_COUNT_WEIGHT,
    TOTAL_TIME_WEIGHT,
    ROUTE_TIME_WEIGHT,
    FAIRNESS_WEIGHT,
    print_solution_pretty, route_cost,
    MAX_ROUTE_MIN, BUS_CAPACITY
)
from utils.instance_generator import gen_instance_multi, load_instance_from_csv
from utils.geo_utils import travel_minutes
from lns_solver import run_lns

# ---------- 參數設定 ----------
MAX_TOTAL_BUSES = 6
DW_ITERATIONS = 2000
MAX_PICKUP_BRANCHES = float("inf")  #15
MAX_ROUTE_EVENTS = float("inf")  #20

# ===============================
# Dantzig-Wolfe Solver
# ===============================
class DantzigWolfeSolver:

    def __init__(self, schools: List[School], stations: List[Station]):
        self.schools = schools
        self.stations = stations
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        self.group_map = {} # For O(1) group lookup by (st_idx, sch_idx)
        for st in stations:
            for sch_idx, count in st.demands.items():
                gid = len(self.groups)
                self.groups.append({
                    'id': gid,
                    'st_idx': st.idx,
                    'sch_idx': sch_idx,
                    'count': count
                })
                self.group_map[(st.idx, sch_idx)] = gid

        self.routes = []
        self.route_pool_keys = set() # For O(1) route duplication check
        self.group_to_routes = [[] for _ in range(len(self.groups))] # For faster Master Problem modeling
        self.duals = [0.0] * len(self.groups)
        self.mu = 0.0
        self.best_feasible_sol = None

    # ===============================
    # Route Pool Management
    # ===============================
    def _add_route(self, r: Route) -> bool:
        """將路徑加入池中並進行去重檢查，返回是否成功加入"""
        g_ids = self._groups_in_route(r)
        g_set = frozenset(g_ids)
        if not g_set or g_set in self.route_pool_keys:
            return False
        
        route_idx = len(self.routes)
        r._g_ids = g_set # Cache for later modeling efficiency
        self.routes.append(r)
        self.route_pool_keys.add(g_set)
        
        # Update group to routes mapping index
        for g_id in g_ids:
            self.group_to_routes[g_id].append(route_idx)
        return True

    # ===============================
    # Initial columns
    # ===============================
    def _generate_random_greedy_solution(self) -> Solution:
        """使用隨機近鄰貪婪方式生成一個可行初始解 - 改進版本"""
        unserved = [g['id'] for g in self.groups]
        routes = []
        
        while unserved and len(routes) < MAX_TOTAL_BUSES:
            curr_route_gids = []
            # 改進：優先選擇需求大的群體作為起點
            start_gid = max(unserved, key=lambda gid: self.groups[gid]['count'])
            curr_route_gids.append(start_gid)
            unserved.remove(start_gid)
            
            while unserved:
                last_gid = curr_route_gids[-1]
                last_st_idx = self.groups[last_gid]['st_idx']
                last_coord = self.st_dict[last_st_idx].coord
                
                # 隨機選擇最近的 3 個候選站點之一，增加初始解多樣性
                candidates = sorted(unserved, key=lambda gid: travel_minutes(last_coord, self.st_dict[self.groups[gid]['st_idx']].coord))
                k = min(3, len(candidates))
                next_gid = random.choice(candidates[:k])
                
                # 測試加入該群組後是否仍符合約束 (載重與最大行駛時間)
                test_gids = curr_route_gids + [next_gid]
                test_route = self._build_route_from_group_ids(test_gids)
                load = sum(self.groups[x]['count'] for x in test_gids)
                
                if test_route and test_route.minutes <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    curr_route_gids.append(next_gid)
                    unserved.remove(next_gid)
                else:
                    break
            
            routes.append(self._build_route_from_group_ids(curr_route_gids))
            
        return build_solution(routes, self.stations)

    def _build_route(self, path: List[Tuple[str, int]]) -> Route:
        """根據路徑序列轉換為 Route 物件"""
        events = []
        for t, idx in path:
            if t == 'g':
                g = self.groups[idx]
                events.append(('pickup', (g['st_idx'], {g['sch_idx']: g['count']})))
            else:
                events.append(('drop', idx))

        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(
            r, self.schools, self.st_dict
        )
        return r

    def _build_route_from_group_ids(self, gids: List[int]) -> Route:
        """根據群組 ID 序列構建 Route 物件，包含基本的 Drop 邏輯"""
        if not gids: return None
        path = [('g', gid) for gid in gids]
        target_schs = list(set(self.groups[gid]['sch_idx'] for gid in gids))
        last_st_idx = self.groups[gids[-1]]['st_idx']
        curr_pos = self.st_dict[last_st_idx].coord
        while target_schs:
            nxt_sch = min(target_schs, key=lambda s: travel_minutes(curr_pos, self.schools[s].coord))
            path.append(('s', nxt_sch))
            curr_pos = self.schools[nxt_sch].coord
            target_schs.remove(nxt_sch)
        return self._build_route(path)
    
    def initialize_columns(self):
        """使用启发法解初始化路径池，确保 Master Problem 一开始就是可行的"""
        print("[DW] 正在初始化路径池 (Initial Columns)...")
        init_sol = self._generate_random_greedy_solution()
        for r in init_sol.routes:
            self._add_route(r)
            
        # 检查是否所有群体都被覆盖
        covered_groups = set().union(*(r._g_ids for r in self.routes if hasattr(r, '_g_ids')))
        
        uncovered = [g for g in self.groups if g['id'] not in covered_groups]
        if uncovered:
            print(f"[DW] 警告：初始路径需覆盖{len(uncovered)}个群体，正在生成单个路径...")
            while uncovered:
                ug = uncovered[0]
                single_route = self._build_route_from_group_ids([ug['id']])
                if single_route:
                    self._add_route(single_route)
                    uncovered.pop(0)
                else:
                    uncovered.pop(0)
            
        print(f"[DW] 初始化完成：共{len(self.routes)}条路径")
            
    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")
        
        # 提高人工變數罰係數以強制找到更好的對偶值
        penalty = 1e7

        m.setObjective(
            gp.quicksum(x[i] * route_cost(self.routes[i]) for i in range(len(self.routes))),
            GRB.MINIMIZE
        )

        constrs = {}
        for g_idx in range(len(self.groups)):
            idxs = self.group_to_routes[g_idx] # 使用預建索引提升效能
            # 每個群體必須被覆蓋剛好一次
            constrs[g_idx] = m.addConstr(gp.quicksum(x[i] for i in idxs) == 1, name=f"Cover_G{g_idx}")

        bus_constr = m.addConstr(gp.quicksum(x[i] for i in range(len(self.routes))) <= MAX_TOTAL_BUSES, name="MaxBus")

        m.optimize()

        if relax and m.status == GRB.OPTIMAL:
            self.duals = [constrs[g].Pi for g in range(len(self.groups))]
            self.mu = bus_constr.Pi
        return m

    def _groups_in_route(self, r: Route) -> List[int]:
        """回傳此路徑包含的群體 ID 列表"""
        # 使用 group_map 進行 O(1) 匹配
        return [self.group_map[(d[0], d[1])] for d in r.pickup_detail]

    # ===============================
    # Pricing 
    # ===============================
    def pricing(self):
        n = len(self.groups)
        best_route = None
        best_rc = 1e-6  # 只接受真正負 RC 的路線

        avg_dual = sum(abs(d) for d in self.duals) / max(1, len(self.duals))
        max_dual = max(abs(d) for d in self.duals) if self.duals else 0

        coverage_counts = [len(self.group_to_routes[g['id']]) for g in self.groups]
        sorted_groups = sorted(self.groups, key=lambda g: coverage_counts[g['id']])

        school_offset = len(self.st_dict)
        station_nodes = {}
        school_nodes = {}

        for i, st_idx in enumerate(self.st_dict):
            station_nodes[st_idx] = i
        for i, sch in enumerate(self.schools):
            school_nodes[i] = school_offset + i

        total_nodes = school_offset + len(self.schools)

        coords = [None] * total_nodes
        for st_idx, nid in station_nodes.items():
            coords[nid] = self.st_dict[st_idx].coord
        for sch_idx, nid in school_nodes.items():
            coords[nid] = self.schools[sch_idx].coord

        dist_matrix = [[0.0] * total_nodes for _ in range(total_nodes)]
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    dist_matrix[i][j] = travel_minutes(coords[i], coords[j])

        max_remaining_dual = sum(max(0, self.duals[g['id']]) for g in self.groups)
        group_target_dist = [
            travel_minutes(self.st_dict[g['st_idx']].coord, self.schools[g['sch_idx']].coord)
            for g in self.groups
        ]

        def pickup_fairness_increment(gid: int, path: List[Tuple[str, int]]) -> float:
            """新 pickup 的 group 若比 path 前面任一 group 離自己的目標學校更遠，則加 1。"""
            new_dist = group_target_dist[gid]
            penalty = 0.0
            for ev_type, prev_gid in path:
                if ev_type == 'g' and new_dist > group_target_dist[prev_gid] + 1e-6:
                    penalty += 1.0
            return penalty

        def dfs(
            curr_node, time, ivm_acc, load,
            dual_acc, fairness_acc,
            visited_mask, onboard_mask, path
        ):
            nonlocal best_route, best_rc

            if len(path) > MAX_ROUTE_EVENTS:
                return

            current_rc = (
                0#BUS_COUNT_WEIGHT
                + TOTAL_TIME_WEIGHT * time
                + ROUTE_TIME_WEIGHT * ivm_acc
                + FAIRNESS_WEIGHT * fairness_acc
                - dual_acc
                - self.mu
            )

            # 完整路線（所有人都下車）→ 更新最佳解
            if onboard_mask == 0 and path:
                
                if current_rc < best_rc:
                    temp_route = self._build_route(path)
                    temp_route.fairness_penalty = fairness_acc
                    temp_g_set = frozenset(self._groups_in_route(temp_route))
                    if temp_g_set not in self.route_pool_keys:
                        best_rc = current_rc
                        best_route = temp_route

            if time > MAX_ROUTE_MIN:
                return
            if load > BUS_CAPACITY:
                return

            schools_to_drop = set()
            for gid in range(n):
                if onboard_mask & (1 << gid):
                    schools_to_drop.add(self.groups[gid]['sch_idx'])

            for sch_idx in schools_to_drop:
                dropping = [
                    gid for gid in range(n)
                    if (onboard_mask & (1 << gid)) and self.groups[gid]['sch_idx'] == sch_idx
                ]

                sch_node = school_nodes[sch_idx]
                dist = dist_matrix[curr_node][sch_node]
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                dropped_load = sum(self.groups[x]['count'] for x in dropping)
                new_onboard = onboard_mask
                for x in dropping:
                    new_onboard &= ~(1 << x)

                new_ivm = ivm_acc + load * dist
                new_path = path + [('s', sch_idx)]

                dfs(
                    sch_node, new_time, new_ivm,
                    load - dropped_load,
                    dual_acc, fairness_acc,
                    visited_mask, new_onboard, new_path
                )

            # Pickup 階段
            pickup_branches = 0
            for g in sorted_groups:
                gid = g['id']
                if visited_mask & (1 << gid):
                    continue
                if load + g['count'] > BUS_CAPACITY:
                    continue

                st_node = station_nodes[g['st_idx']]
                dist = dist_matrix[curr_node][st_node]
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                new_visited = visited_mask | (1 << gid)
                new_onboard = onboard_mask | (1 << gid)
                new_ivm = ivm_acc + load * dist
                new_fairness = fairness_acc + pickup_fairness_increment(gid, path)
                new_path = path + [('g', gid)]

                dfs(
                    st_node, new_time, new_ivm,
                    load + g['count'],
                    dual_acc + self.duals[gid],
                    new_fairness,
                    new_visited, new_onboard, new_path
                )
                pickup_branches += 1
                if pickup_branches >= MAX_PICKUP_BRANCHES:
                    break

        # 搜索起點
        pricing_attempts = 0
        for g in sorted_groups:
            gid = g['id']
            st_node = station_nodes[g['st_idx']]
            pricing_attempts += 1
            dfs(
                st_node, 0.0, 0.0, g['count'],
                self.duals[gid], 0.0,
                (1 << gid), (1 << gid),
                [('g', gid)]
            )

        if best_route:
            tqdm.write(f"[Pricing] 找到RC={best_rc:.4f}，路徑長度={len(best_route.events)}，嘗試{pricing_attempts}個起點")
        else:
            tqdm.write(f"[Pricing] 未找到負RC路線 (最佳RC={best_rc:.4f}，最大對偶={max_dual:.2f}，平均對偶={avg_dual:.2f})")

        return best_route

    # ===============================
    # Main loop
    # ===============================
    def run(self) -> Solution:

        self.initialize_columns()

        for it in tqdm(range(DW_ITERATIONS), desc="DW/B&P Iterations"):

            m = self.solve_master(relax=True)
            if m.status != GRB.OPTIMAL:
                tqdm.write(f"[BP] Master problem not optimal at iteration {it}. Status: {m.status}")
                break

            # 每 10 次迭代輸出一次當前的 RMP 目標值
            if it % 10 == 0:
                avg_dual = sum(abs(d) for d in self.duals) / max(1, len(self.duals))
                tqdm.write(f"[DW Iter {it:03d}] RMP目標={m.objVal:.2f} | 平均對偶={avg_dual:.4f} | mu={self.mu:.4f} | 路徑池大小={len(self.routes)}")

            new_route = self.pricing()
            if new_route is None:
                tqdm.write(f"[DW Iter {it:03d}] 找不到具有負縮減成本的路徑，演算法收斂。")
                break
            if not self._add_route(new_route): # 使用 _add_route 進行去重檢查並嘗試加入
                tqdm.write(f"[DW Iter {it:03d}] 找到重複路徑，演算法收斂。") # 如果是重複路徑，則視為收斂
                break

        # final integer solve
        m = self.solve_master(relax=False)
        if m.status != GRB.OPTIMAL:
            tqdm.write("[DW] 整數問題未找到可行解!")
            return Solution([], 0, 0, 0, 0, False)

        x = m.getVars()
        selected = [self.routes[i] for i in range(len(self.routes)) if x[i].X > 0.5]

        total_min = sum(r.minutes for r in selected)
        total_ivm = sum(r.in_vehicle_minutes for r in selected)
        total_fair = sum(r.fairness_penalty for r in selected)
        served = sum(sum(c for _, _, c, _ in r.pickup_detail) for r in selected)

        return Solution(selected, total_min, total_ivm, total_fair, served, True)

def run_dantzig_wolfe(schools: List[School], stations: List[Station]) -> Solution:
    solver = DantzigWolfeSolver(schools, stations)
    return solver.run()

if __name__ == "__main__":
    random.seed(42)
    csv_instance = load_instance_from_csv(stops_csv="./data/stops-b_7.csv", time_csv="./data/time-b_7.csv")
    if csv_instance is not None:
        schools, stations = csv_instance
        print("[DATA] loaded instance from CSV")
    else:
        schools, stations = gen_instance_multi()
    best_sol = run_dantzig_wolfe(schools, stations)
    print_solution_pretty(best_sol, stations, schools)
