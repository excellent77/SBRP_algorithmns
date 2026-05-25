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


DW_ITERATIONS = 2000
MAX_PICKUP_BRANCHES = float("inf")
MAX_ROUTE_EVENTS = float("inf")

class DantzigWolfeSolver(object):
    def __init__(self, schools: List[School], stations: List[Station], time_matrix: List[List[float]]):
        """初始化校車路徑問題 (SBRP) 的 Dantzig-Wolfe 求解器。

        Args:
            schools: 學校物件列表。
            stations: 站點物件列表。
            time_matrix: 所有原始索引之間的行駛時間二維列表。
        """
        self.schools = schools
        self.stations = stations
        self.time_matrix = time_matrix
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        self.group_map = {} # For O(1) group lookup by (st_idx, sch_idx)
        for st in stations:
            for sch_idx, count in st.demands.items():
                gid = len(self.groups) # type: ignore
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
        """預計算定價問題之 DFS 中使用的靜態圖和群組數據。
        這最佳化了列生成 (Column Generation) 過程中的重複查找。
        """
        self.group_school_idx = [0] * len(self.groups)
        self.group_count = [0] * len(self.groups)
        self.group_direct_school_dist = [0.0] * len(self.groups)
        self.school_group_masks = [0] * len(self.schools)
        self.school_group_loads = [0] * len(self.schools)

        for g in self.groups:
            gid = self.group_map[(g.idx, list(g.demands.items())[0][0])] # type: ignore
            sch_idx = list(g.demands.items())[0][0]
            now_orig = self.groups[gid].orig_idx
            sch_orig = self.schools[sch_idx].orig_idx

            self.group_school_idx[gid] = sch_idx
            self.group_direct_school_dist[gid] = self.time_matrix[now_orig][sch_orig]
            self.group_count[gid] = list(g.demands.items())[0][1]
            self.school_group_masks[sch_idx] |= 1 << gid
            self.school_group_loads[sch_idx] += self.group_count[gid]

    def route_to_key(self, r: Route):
        """根據路徑事件為給定路徑生成唯一的字串鍵值。

        此鍵值保留了事件順序，並能區分可能涵蓋相同群組
        但具有不同順序或不同丟客點的路徑。

        Args:
            r: Route 物件。

        Returns:
            代表路徑唯一鍵值的字串。
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
        """嘗試將路徑加入路徑池。

        它使用 `route_to_key` 檢查是否重複，如果路徑是新的，
        則更新主問題的內部數據結構。

        Args:
            r: 要加入的 Route 物件。

        Returns:
            如果成功加入路徑（即非重複），則為 True，否則為 False。
        """
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
        """將路徑（群組/學校索引序列）轉換為 Route 物件。

        路徑中的正索引指代群組 ID（取貨），而負索引指代學校 ID（丟客），
        並以 -1 進行調整（例如：-1 代表學校 0）。

        Args:
            path: 代表事件序列的整數列表。

        Returns:
            根據路徑構建的 Route 物件。
        """
        events = []
        for idx in path:
            if idx >= 0:
                events.append(self.groups[idx])
            else:
                events.append(self.schools[-idx - 1])
        return build_route(events, self.schools, self.st_dict, self.time_matrix)
    
    def initialize_columns(self):
        """使用貪婪啟發式演算法生成的路徑初始化路徑池。

        這確保了受限主問題 (RMP) 有一組初始的可行列可以使用，防止在列生成過程開始時出現問題。
        """
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        init_sol = run_greedy(self.schools, self.stations, self.time_matrix)
        for r in init_sol.routes:
            self._add_route(r)

    def solve_master(self, relax=True):
        """使用 Gurobi 解決受限主問題 (RMP)。

        RMP 是一個集合分割問題，在巴士數量限制下，選擇路徑子集
        以最小成本涵蓋所有群組。它可以以鬆弛 (LP) 或整數 (IP) 形式求解。

        Args:
            relax: 如果為 True，則求解 LP 鬆弛；否則求解 IP。

        Returns:
            最佳化後的 Gurobi 模型物件。
        """
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
            constrs[g_idx] = m.addConstr(gp.quicksum(x[i] for i in idxs) == 1, name=f"Cover_G{g_idx}")
        bus_constr = m.addConstr(gp.quicksum(x[i] for i in range(len(self.routes))) <= MAX_TOTAL_BUSES, name="MaxBus") # type: ignore

        m.optimize()

        if relax and m.status == GRB.OPTIMAL:
            self.duals = [constrs[g].Pi for g in range(len(self.groups))]
            self.mu = bus_constr.Pi

        return m
    
    def pricing(self):
        """解決定價問題（列生成），以找到具有負縮減成本的新路徑。

        這通常是一個帶有資源限制的最短路徑問題，使用深度優先搜索 (DFS) 
        或類似的標籤演算法求解。縮減成本是使用來自受限主問題的對偶值計算的。

        Returns:
            如果找到，則為具有負縮減成本的新 Route 物件，否則為 None。
        """

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
                # Calculate reduced cost for dropping
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
                if path[-1] >=0 and self.group_direct_school_dist[path[-1]] < self.group_direct_school_dist[gid]:
                    continue
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

        for gid in range(len(self.groups)):
            # Start DFS for each group as a potential first pickup
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

    def run(self) -> Solution:
        """執行主要的 Dantzig-Wolfe 列生成演算法。

        它從初始化列開始，然後迭代地解決受限主問題 (RMP) 和定價問題。
        具有負縮減成本的新路徑（列）會加入到 RMP 中，直到找不到更多此類路徑或達到最大迭代次數。
        最後，將 RMP 作為整數規劃求解以獲得最終解。

        Returns:
            代表 Dantzig-Wolfe 演算法所找到之最佳解的 Solution 物件。
        """

        self.initialize_columns()

        for it in tqdm(range(DW_ITERATIONS), desc="DW/B&P Iterations"):

            m = self.solve_master(relax=True)
            if m.status != GRB.OPTIMAL:
                tqdm.write(f"[BP] Master problem not optimal at iteration {it}. Status: {m.status}")
                break
            if it % 10 == 0:
                tqdm.write(f"[DW Iter {it:03d}] 當前 RMP 目標值 (Relaxed): {m.objVal:.2f}")

            new_route = self.pricing()
            if new_route is None:
                tqdm.write(f"[DW Iter {it:03d}] 找不到具有負縮減成本的路徑，演算法收斂。")
                break
            if not self._add_route(new_route):
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
    """運行校車路徑問題的 Dantzig-Wolfe 分解求解器。

    Args:
        schools: 學校物件列表。
        stations: 站點物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。

    Returns:
        代表 Dantzig-Wolfe 演算法所找到之解的 Solution 物件。
    """
    solver = DantzigWolfeSolver(schools, stations, time_matrix)
    return solver.run()


if __name__ == "__main__":
    """
    運行 Dantzig-Wolfe 求解器的入口點。
    從 CSV 檔案載入實例數據，運行求解器並列印解。
    """
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-b_7.csv", time_csv="./data/time-b_7.csv")
    best_sol = run_dantzig_wolfe(schools, stations, time_matrix)
    print_solution_pretty(best_sol, stations, schools)