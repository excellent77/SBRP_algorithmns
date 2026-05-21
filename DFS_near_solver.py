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
    MAX_ROUTE_MIN
)
from utils.instance_generator import gen_instance_multi, load_instance_from_csv
from utils.geo_utils import travel_minutes
from lns_solver import run_lns

# ---------- 參數設定 ----------
MAX_TOTAL_BUSES = 10
BUS_CAPACITY = 40
DW_ITERATIONS = 200
MAX_PICKUP_BRANCHES = 10
MAX_ROUTE_EVENTS = 14

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
        self.alpha = 0.7  # dual stabilization
        self.best_feasible_sol = None
        self._prepare_pricing_cache()

    def _prepare_pricing_cache(self):
        """Precompute static graph and group data used by every pricing DFS."""
        self.n_groups = len(self.groups)
        self.school_offset = len(self.st_dict)
        self.station_nodes = {st_idx: i for i, st_idx in enumerate(self.st_dict)}
        self.school_nodes = {
            sch_idx: self.school_offset + sch_idx
            for sch_idx in range(len(self.schools))
        }
        total_nodes = self.school_offset + len(self.schools)

        coords = [None] * total_nodes
        for st_idx, node in self.station_nodes.items():
            coords[node] = self.st_dict[st_idx].coord
        for sch_idx, node in self.school_nodes.items():
            coords[node] = self.schools[sch_idx].coord

        self.dist_matrix = [[0.0] * total_nodes for _ in range(total_nodes)]
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    self.dist_matrix[i][j] = travel_minutes(coords[i], coords[j])

        self.group_station_node = [0] * self.n_groups
        self.group_school_node = [0] * self.n_groups
        self.group_school_idx = [0] * self.n_groups
        self.group_count = [0] * self.n_groups
        self.group_direct_school_dist = [0.0] * self.n_groups
        self.school_group_masks = [0] * len(self.schools)
        self.school_group_loads = [0] * len(self.schools)

        for g in self.groups:
            gid = g['id']
            sch_idx = g['sch_idx']
            st_node = self.station_nodes[g['st_idx']]
            sch_node = self.school_nodes[sch_idx]
            self.group_station_node[gid] = st_node
            self.group_school_node[gid] = sch_node
            self.group_school_idx[gid] = sch_idx
            self.group_count[gid] = g['count']
            self.group_direct_school_dist[gid] = self.dist_matrix[st_node][sch_node]
            self.school_group_masks[sch_idx] |= 1 << gid
            self.school_group_loads[sch_idx] += g['count']

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
        """使用隨機近鄰貪婪方式生成一個可行初始解"""
        unserved = [g['id'] for g in self.groups]
        routes = []
        
        while unserved and len(routes) < MAX_TOTAL_BUSES:
            curr_route_gids = []
            # 隨機挑選起點增加多樣性
            start_gid = random.choice(unserved)
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
        """使用啟發式解初始化路徑池，確保 Master Problem 一開始就是可行的"""
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        init_sol = self._generate_random_greedy_solution()#run_lns(self.schools, self.stations, print_log=False)
        for r in init_sol.routes:
            self._add_route(r)
            
        # 檢查是否所有群體都被涵蓋
        covered_groups = set().union(*(r._g_ids for r in self.routes if hasattr(r, '_g_ids')))
        
        for g in self.groups:
            if g['id'] not in covered_groups:
                raise ValueError(f"初始路徑池未涵蓋所有群體！缺少群體 ID: {g['id']}")

    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")
        # 增加人工變數確保可行性 (Set Partitioning with Slack)，並引導對偶值
        z = m.addVars(len(self.groups), vtype=vtype, lb=0, name="z")
        
        penalty = 1e6

        m.setObjective(
            gp.quicksum(x[i] * route_cost(self.routes[i]) for i in range(len(self.routes))) +
            gp.quicksum(z[g] * penalty for g in range(len(self.groups))),
            GRB.MINIMIZE
        )

        constrs = {}
        for g_idx in range(len(self.groups)):
            idxs = self.group_to_routes[g_idx] # 使用預建索引提升效能
            # 每個群體必須被覆蓋剛好一次
            constrs[g_idx] = m.addConstr(gp.quicksum(x[i] for i in idxs) + z[g_idx] == 1, name=f"Cover_G{g_idx}")

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

        n = self.n_groups
        best_route = None
        best_rc = 0
        total_pos_dual = sum(max(0, d) for d in self.duals)
        group_gids = [g['id'] for g in self.groups]
        dist_matrix = self.dist_matrix
        group_direct_school_dist = self.group_direct_school_dist

        # =========================
        # DFS + DP
        # =========================
        memo = {}
        def dfs(
            curr_node,
            current_gid,
            time,
            ivm_acc,
            load,
            dual_acc,
            fairness_acc,
            visited_mask,
            onboard_mask,
            path
        ):
            nonlocal best_route, best_rc

            if best_rc < -0.0001:
                return
            if len(path) > MAX_ROUTE_EVENTS:
                return

            current_rc = (
                BUS_COUNT_WEIGHT
                + TOTAL_TIME_WEIGHT * time
                + ROUTE_TIME_WEIGHT * ivm_acc
                + FAIRNESS_WEIGHT * fairness_acc
                - dual_acc
                - self.mu
            )

            if current_rc - (total_pos_dual - dual_acc) > best_rc:
                return

            state = (
                curr_node,
                current_gid,
                visited_mask,
                onboard_mask,
                load
            )

            prev_best = memo.get(state)
            if prev_best is not None and prev_best <= current_rc:
                return

            memo[state] = current_rc
            if onboard_mask == 0 and path:
                if current_rc < best_rc:
                    temp_route = self._build_route(path)
                    temp_g_set = frozenset(self._groups_in_route(temp_route))
                    if temp_g_set not in self.route_pool_keys: # 檢查是否為重複路徑
                        best_rc = current_rc
                        best_route = temp_route
                        return # 找到新的最佳路徑，立即停止 DFS 搜尋
                #return
            

            if time > MAX_ROUTE_MIN:
                return
            if load > BUS_CAPACITY:
                return

            tried_schools = 0
            mask = onboard_mask
            while mask:
                lowbit = mask & -mask
                gid = lowbit.bit_length() - 1
                mask ^= lowbit
                sch_idx = self.group_school_idx[gid]
                sch_bit = 1 << sch_idx
                if tried_schools & sch_bit:
                    continue
                tried_schools |= sch_bit

                dropping_mask = onboard_mask & self.school_group_masks[sch_idx]
                sch_node = self.school_nodes[sch_idx]
                dist = dist_matrix[curr_node][sch_node]
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                dropped_load = 0
                drop_iter = dropping_mask
                while drop_iter:
                    lowbit = drop_iter & -drop_iter
                    dropped_load += self.group_count[lowbit.bit_length() - 1]
                    drop_iter ^= lowbit

                new_onboard = onboard_mask & ~dropping_mask

                new_ivm = ivm_acc + load * dist
                new_path = path + [('s', sch_idx)]

                dfs(
                    sch_node,
                    current_gid,
                    new_time,
                    new_ivm,
                    load - dropped_load,
                    dual_acc,
                    fairness_acc,
                    visited_mask,
                    new_onboard,
                    new_path
                )


            pickup_branches = 0
            for gid in group_gids:
                if visited_mask & (1 << gid):
                    continue
                if group_direct_school_dist[gid] >= group_direct_school_dist[current_gid]:
                    continue
                if load + self.group_count[gid] > BUS_CAPACITY:
                    continue

                st_node = self.group_station_node[gid]
                dist = dist_matrix[curr_node][st_node]
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                new_visited = visited_mask | (1 << gid)
                new_onboard = onboard_mask | (1 << gid)
                new_ivm = ivm_acc + load * dist
                new_path = path + [('g', gid)]

                dfs(
                    st_node,
                    gid,
                    new_time,
                    new_ivm,
                    load + self.group_count[gid],
                    dual_acc + self.duals[gid],
                    fairness_acc,
                    new_visited,
                    new_onboard,
                    new_path
                )
                pickup_branches += 1
                if pickup_branches >= MAX_PICKUP_BRANCHES:
                    break

        # =========================
        # Start DFS
        # =========================

        for gid in group_gids:
            st_node = self.group_station_node[gid]
            dfs(
                st_node,
                gid,
                0.0,
                0.0,
                self.group_count[gid],
                self.duals[gid],
                0.0,
                (1 << gid),
                (1 << gid),
                [('g', gid)]
            )
            if best_route and best_rc < -1e-4:
                break
        if best_route:
            tqdm.write(f"[Pricing] Found route RC={best_rc:.2f} path_length={len(best_route.events)}")

        return best_route

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
                tqdm.write(f"[DW Iter {it:03d}] 當前 RMP 目標值 (Relaxed): {m.objVal:.2f}")

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
    csv_instance = load_instance_from_csv(stops_csv="./data/stops-b_30.csv", time_csv="./data/time-b_30.csv")
    if csv_instance is not None:
        schools, stations = csv_instance
        print("[DATA] loaded instance from CSV")
    else:
        schools, stations = gen_instance_multi()
    best_sol = run_dantzig_wolfe(schools, stations)
    print_solution_pretty(best_sol, stations, schools)
