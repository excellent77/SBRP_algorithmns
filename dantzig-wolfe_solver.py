import os
import random
from tqdm import tqdm
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

from utils.data_models import (
    BUS_COUNT_WEIGHT, TOTAL_TIME_WEIGHT, ROUTE_TIME_WEIGHT, FAIRNESS_WEIGHT,
    School, Station, Route, Solution
)
from utils.solution_utils import (
    build_route, build_solution,print_solution_pretty,
    MAX_ROUTE_MIN, BUS_CAPACITY, MAX_TOTAL_BUSES
)
from utils.instance_processer import load_instance_from_csv
from greedy_solver import run_greedy



# ---------- 參數設定 ----------
DW_ITERATIONS = 2000
MAX_PICKUP_BRANCHES = float("inf")#10
MAX_ROUTE_EVENTS = float("inf")#114



# ===============================
# Dantzig-Wolfe Solver
# ===============================
class DantzigWolfeSolver(object):

    def __init__(self, schools: List[School], stations: List[Station], time_matrix: List[List[float]]):
        self.schools = schools
        self.stations = stations
        self.time_matrix = time_matrix
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        self.group_map = {} # For O(1) group lookup by (st_idx, sch_idx)
        for st in stations:
            for sch_idx, count in st.demands.items():
                gid = len(self.groups)
                self.groups.append(Station(idx=st.idx, name=st.name, demands={sch_idx:count}, orig_idx=st.orig_idx))
                event = self.groups[-1]
                self.group_map[(event.idx, list(event.demands.items())[0][0])] = gid

        self.routes = []
        self.route_pool_keys = set() # For O(1) route duplication check
        self.group_to_routes = [[] for _ in range(len(self.groups))] # For faster Master Problem modeling
        self.duals = [0.0] * len(self.groups)
        self.mu = 0.0
        self.best_feasible_sol = None
        self._prepare_pricing_cache()

    def _prepare_pricing_cache(self):
        """Precompute static graph and group data used by every pricing DFS."""
        self.group_school_idx = [0] * len(self.groups)
        self.group_count = [0] * len(self.groups)
        self.group_direct_school_dist = [0.0] * len(self.groups)
        self.school_group_masks = [0] * len(self.schools)
        self.school_group_loads = [0] * len(self.schools)

        for g in self.groups:
            gid = self.group_map[(g.idx, list(g.demands.items())[0][0])]
            sch_idx = list(g.demands.items())[0][0]
            now_orig = self.groups[gid].orig_idx
            sch_orig = self.schools[sch_idx].orig_idx

            self.group_school_idx[gid] = sch_idx
            self.group_direct_school_dist[gid] = self.time_matrix[now_orig][sch_orig]
            self.group_count[gid] = list(g.demands.items())[0][1]
            self.school_group_masks[sch_idx] |= 1 << gid
            self.school_group_loads[sch_idx] += self.group_count[gid]


    # ===============================
    # Initial columns
    # ===============================
    def route_to_key(self, r: Route):
        """根據路徑事件生成唯一鍵值，保留順序。
        只用 group 集合會誤把不同順序、不同成本的路徑當成重複路徑。
        """
        key = ""
        for event in r.events:
            if isinstance(event, Station):
                sch_idx = list(event.demands.keys())[0]
                key += str(self.group_map[(event.idx, sch_idx)])
            elif isinstance(event, School):
                key += str(-(event.idx + 1))
        return key

    def _add_route(self, r: Route) -> bool:
        """嘗試將路徑加入路徑池，返回是否成功加入（即非重複路徑）"""
        g_set = self.route_to_key(r)
        if not g_set or g_set in self.route_pool_keys:
            return False

        route_idx = len(self.routes)
        self.routes.append(r)
        self.route_pool_keys.add(g_set)

        for event in r.events:
            if isinstance(event, Station):
                sch_idx = list(event.demands.keys())[0]
                self.group_to_routes[self.group_map[(event.idx, sch_idx)]].append(route_idx)
        return True

    def path_to_route(self, path: List[int]) -> Route:
        """根據路徑序列轉換為 Route 物件"""
        events = []
        for idx in path:
            if idx >= 0:
                events.append(self.groups[idx])
            else:
                events.append(self.schools[-idx - 1])
        return build_route(events, self.schools, self.st_dict, self.time_matrix)
    
    def initialize_columns(self):
        """使用啟發式解初始化路徑池，確保 Master Problem 一開始就是可行的"""
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        init_sol = run_greedy(self.schools, self.stations, self.time_matrix)
        for r in init_sol.routes:
            self._add_route(r)

    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")

        m.setObjective(
            gp.quicksum(x[i] * self.routes[i].route_cost() for i in range(len(self.routes))),
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

    # ===============================
    # Pricing 
    # ===============================
    def pricing(self):

        best_route = None
        best_rc = 0.0

        def pickup_fairness_increment(new_gid: int, path: list[int]) -> float:
            new_sch = self.group_school_idx[new_gid]
            new_dist = self.group_direct_school_dist[new_gid]
            penalty = 0.0

            for gid in path:
                if gid < 0:
                    continue
                sch = self.group_school_idx[gid]
                if sch != new_sch:
                    continue
                dist = self.group_direct_school_dist[gid]
                if dist < new_dist:
                    penalty += 1.0

            return penalty

        # =========================
        # DFS
        # =========================
        def dfs(
            path,
            time,
            ivm_acc,
            load,
            dual_acc,
            fairness_acc,
            visited_mask,
            onboard_mask,
        ):
            nonlocal best_route, best_rc

            if len(path) > MAX_ROUTE_EVENTS:
                return
            if time > MAX_ROUTE_MIN:
                return
            if load > BUS_CAPACITY:
                return

            if path[-1] >= 0:
                curr_node = self.groups[path[-1]].orig_idx
            else:
                curr_node = self.schools[-path[-1] - 1].orig_idx
            

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
                dist = abs(self.time_matrix[curr_node][self.schools[sch_idx].orig_idx])
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                dropped_load = 0
                drop_iter = dropping_mask
                while drop_iter:
                    lowbit = drop_iter & -drop_iter
                    dropped_load += self.group_count[lowbit.bit_length() - 1]
                    drop_iter ^= lowbit

                new_ivm = ivm_acc + load * dist
                new_path = path + [-(sch_idx + 1)]
                new_onboard = onboard_mask & ~dropping_mask

                dfs(
                    new_path,
                    new_time,
                    new_ivm,
                    load - dropped_load,
                    dual_acc,
                    fairness_acc,
                    visited_mask,
                    new_onboard,
                )

            pickup_branches = 0
            for gid in range(len(self.groups)):
                if visited_mask & (1 << gid):
                    continue
                #if path[-1] >=0 and self.group_direct_school_dist[path[-1]] < self.group_direct_school_dist[gid]:
                #    continue
                if load + self.group_count[gid] > BUS_CAPACITY:
                    continue

                st_node = self.groups[gid].orig_idx
                dist = self.time_matrix[curr_node][st_node]
                new_time = time + dist

                if new_time > MAX_ROUTE_MIN:
                    continue

                new_path = path + [gid]
                new_visited = visited_mask | (1 << gid)
                new_onboard = onboard_mask | (1 << gid)
                new_fairness = fairness_acc + pickup_fairness_increment(gid, path)
                new_ivm = ivm_acc + load * dist

                dfs(
                    new_path,
                    new_time,
                    new_ivm,
                    load + self.group_count[gid],
                    dual_acc + self.duals[gid],
                    new_fairness,
                    new_visited,
                    new_onboard
                )
                pickup_branches += 1
                if pickup_branches >= MAX_PICKUP_BRANCHES:
                    break

            if onboard_mask == 0 and path:
                current_rc = (
                    BUS_COUNT_WEIGHT
                    + TOTAL_TIME_WEIGHT * time
                    + ROUTE_TIME_WEIGHT * ivm_acc
                    + FAIRNESS_WEIGHT * fairness_acc
                    - dual_acc
                    - self.mu
                )
                if current_rc < best_rc:
                    temp_route = self.path_to_route(path)
                    temp_key = self.route_to_key(temp_route)
                    if temp_key and temp_key not in self.route_pool_keys:
                        best_rc = current_rc
                        best_route = temp_route

        # =========================
        # Start DFS
        # =========================
        for gid in range(len(self.groups)):
            dfs(
                [gid],
                0.0,
                0.0,
                self.group_count[gid],
                self.duals[gid],
                0.0,
                (1 << gid),
                (1 << gid)
            )
        if best_route:
            tqdm.write(f"[Pricing] Found route RC={best_rc:.2f} path_length={len(best_route.events)}")

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
            return build_solution(routes=[], stations=self.stations)

        x = m.getVars()
        selected = [self.routes[i] for i in range(len(self.routes)) if x[i].X > 0.5]

        return build_solution(selected, self.stations)

def run_dantzig_wolfe(schools: List[School], stations: List[Station], time_matrix:List[List[float]]) -> Solution:
    solver = DantzigWolfeSolver(schools, stations, time_matrix)
    return solver.run()


if __name__ == "__main__":
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-b_7.csv", time_csv="./data/time-b_7.csv")
    best_sol = run_dantzig_wolfe(schools, stations, time_matrix)
    print_solution_pretty(best_sol, stations, schools)