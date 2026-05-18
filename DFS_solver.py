import os
import random
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    simulate_route, build_solution,
    print_solution_pretty, plot_routes_on_map, route_cost, try_merge_routes,
    MAX_ROUTE_MIN, BUS_CAPACITY
)
from utils.instance_generator import gen_instance_multi
from utils.geo_utils import travel_minutes

# ---------- 參數設定 ----------
MAX_TOTAL_BUSES = 6
ROUTE_POOL_SIZE = 12000

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
        self.group_map = {}
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
        self.route_pool_keys = set()
        self.group_to_routes = [[] for _ in range(len(self.groups))]
        self.best_feasible_sol = None

    def _add_route(self, r: Route) -> bool:
        g_ids = self._groups_in_route(r)
        g_set = frozenset(g_ids)
        if not g_set or g_set in self.route_pool_keys:
            return False

        route_idx = len(self.routes)
        r._g_ids = g_set
        self.routes.append(r)
        self.route_pool_keys.add(g_set)
        for g_id in g_ids:
            self.group_to_routes[g_id].append(route_idx)
        return True

    def _single_group_route(self, g):
        """為單一群體建立基礎可行路徑"""
        events = [
            ('pickup', (g['st_idx'], {g['sch_idx']: g['count']})),
            ('drop', g['sch_idx'])
        ]
        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(
            r, self.schools, self.st_dict
        )
        return r

    def _get_coord(self, n_type, n_idx):
        """根據類型獲取座標"""
        if n_type == 'st':
            return self.st_dict[n_idx].coord
        else:
            return self.schools[n_idx].coord

    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        self.x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")

        m.setObjective(
            gp.quicksum(self.x[i] * route_cost(self.routes[i]) for i in range(len(self.routes))),
            GRB.MINIMIZE
        )

        constrs = {}
        for g_idx in range(len(self.groups)):
            idxs = self.group_to_routes[g_idx]
            # 改為 Set Covering 約束，增加可行性
            constrs[g_idx] = m.addConstr(gp.quicksum(self.x[i] for i in idxs)== 1, name=f"Cover_G{g_idx}")

        bus_constr = m.addConstr(gp.quicksum(self.x[i] for i in range(len(self.routes))) <= MAX_TOTAL_BUSES, name="MaxBus")

        m.optimize()
        return m

    def _groups_in_route(self, r: Route) -> List[int]:
        """回傳此路徑包含的群體 ID 列表"""
        if hasattr(r, '_g_ids'):
            return list(r._g_ids)
        return [self.group_map[(st_idx, sch_idx)] for st_idx, sch_idx, _, _ in r.pickup_detail]

    # ===============================
    # Route Pool Generation (Matheuristic)
    # ===============================
    def _enumerate_routes(self, target_size=1000):
        """透過 DFS 窮舉生成多樣性的路徑池，不依賴對偶值"""
        print(f"[Matheuristic] 枚舉路徑中... 目標數量: {target_size}")
        
        # 1. 基礎保險：每個群組的直達路徑
        for g in self.groups:
            self._add_route(self._single_group_route(g))
            
        n = len(self.groups)
        station_nodes = {st.idx: i for i, st in enumerate(self.stations)}
        school_nodes = {i: len(self.stations) + i for i in range(len(self.schools))}
        total_nodes = len(self.stations) + len(self.schools)
        
        coords = [s.coord for s in self.stations] + [s.coord for s in self.schools]
        dist_matrix = [[travel_minutes(coords[i], coords[j]) if i != j else 0.0 for j in range(total_nodes)] for i in range(total_nodes)]
        
        def dfs(curr_node, time, ivm_acc, load, visited_mask, onboard_mask, path):
            if len(self.routes) >= target_size: return

            if onboard_mask == 0 and path:
                g_set = frozenset(gid for t, gid in path if t == 'g')
                if g_set not in self.route_pool_keys:
                    route = self._build_route(path)
                    if route.minutes <= MAX_ROUTE_MIN:
                        self._add_route(route)
                return

            if time > MAX_ROUTE_MIN or load > BUS_CAPACITY or len(path) > 12:
                return

            # 分支 A: 優先嘗試前往學校下車
            onboard_gids = [gid for gid in range(n) if (onboard_mask & (1 << gid))]
            sch_to_drop = list(set(self.groups[gid]['sch_idx'] for gid in onboard_gids))
            for sch_idx in sch_to_drop:
                sch_node = school_nodes[sch_idx]
                d = dist_matrix[curr_node][sch_node]
                if time + d <= MAX_ROUTE_MIN:
                    new_onboard = onboard_mask
                    new_load = load
                    for gid in onboard_gids:
                        if self.groups[gid]['sch_idx'] == sch_idx:
                            new_onboard &= ~(1 << gid)
                            new_load -= self.groups[gid]['count']
                    dfs(sch_node, time + d, ivm_acc + load * d, new_load, visited_mask, new_onboard, path + [('s', sch_idx)])
                    if len(self.routes) >= target_size: return

            # 分支 B: 嘗試撿人
            candidates = [g for g in self.groups if not (visited_mask & (1 << g['id'])) and load + g['count'] <= BUS_CAPACITY]
            random.shuffle(candidates)
            # 必須限制分支因子，否則 DFS 會陷入局部站點的排列組合，導致路徑池缺乏多樣性
            for g in candidates[:5]: 
                st_node = station_nodes[g['st_idx']]
                d = dist_matrix[curr_node][st_node]
                if time + d <= MAX_ROUTE_MIN:
                    dfs(st_node, time + d, ivm_acc + load * d, load + g['count'], 
                        visited_mask | (1 << g['id']), onboard_mask | (1 << g['id']), 
                        path + [('g', g['id'])])
                    if len(self.routes) >= target_size: return

        start_indices = list(range(n))
        random.shuffle(start_indices)
        for idx in start_indices:
            g = self.groups[idx]
            dfs(station_nodes[g['st_idx']], 0.0, 0.0, g['count'], (1 << idx), (1 << idx), [('g', idx)])
            if len(self.routes) >= target_size: break

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
        """主程式：改為一次性枚舉池後求解"""
        self._enumerate_routes(target_size=ROUTE_POOL_SIZE)

        # 執行一次性整數規劃求解
        m = self.solve_master(relax=False)
        if m.status != GRB.OPTIMAL:
            print("[Matheuristic] 模型未找到最優解。")
            return build_solution([], self.stations)

        # 使用具名變數存取結果
        selected = [self.routes[i] for i in range(len(self.routes)) if self.x[i].X > 0.5]

        return build_solution(selected, self.stations)

def run_dantzig_wolfe(schools: List[School], stations: List[Station]) -> Solution:
    solver = DantzigWolfeSolver(schools, stations)
    solution = solver.run()
    
    return try_merge_routes(solution, stations, schools)

if __name__ == "__main__":
    random.seed(42)
    schools, stations = gen_instance_multi()
    best_sol = run_dantzig_wolfe(schools, stations)
    print_solution_pretty(best_sol, stations, schools)
    m = plot_routes_on_map(best_sol, stations, schools, title="Dantzig-Wolfe Optimized")
    os.makedirs("./data", exist_ok=True)
    m.save("./data/dfs_routes.html")
