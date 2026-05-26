import os
import random
from tqdm import tqdm
from typing import List, Optional
import gurobipy as gp
from gurobipy import GRB
import copy

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    build_route, build_solution, print_solution_pretty,
    MAX_ROUTE_MIN, BUS_CAPACITY, MAX_TOTAL_BUSES
)
from utils.instance_processer import load_instance_from_csv
from greedy_solver import run_greedy

# LNS pricing 內部迭代數刻意壓低，避免每次 pricing 呼叫成本過高
LNS_PRICING_ITERATIONS = 2000
DESTROY_ROUTES_PROB    = 0.5   # 50% 機率做整條路線破壞
DESTROY_DEGREE         = 0.8   # random destroy 時移除比例

DW_ITERATIONS   = 20



# ------------------------------------------------------------------------------
# Reduced Cost 計算（定價問題的核心評估函式）
# 供 pricing 內部判斷哪些路線值得加入 pool
# ------------------------------------------------------------------------------
def compute_reduced_cost(
    route: Route,
    group_map: dict,
    duals: List[float],
    mu: float
) -> float:
    """計算單條路線的縮減成本。

    Args:
        route: 要評估的 Route 物件。
        group_map: (st_idx, sch_idx) -> gid 的對應字典。
        duals: 當前 RMP 的對偶值列表（每個 group 一個）。
        mu: 巴士數量限制的對偶值。

    Returns:
        縮減成本（負值代表此路線值得加入 pool）。
    """
    dual_sum = 0.0
    for ev in route.events:
        if isinstance(ev, Station):
            for sch_idx in ev.demands.keys():
                key = (ev.idx, sch_idx)
                if key in group_map:
                    dual_sum += duals[group_map[key]]
    return route.route_cost() - dual_sum - mu


# ------------------------------------------------------------------------------
# LNS 子模組（為 pricing 特化，與原始 lns_solver.py 解耦）
# ------------------------------------------------------------------------------
def _rebuild_solution(
    routes: List[Route],
    stations_list: List[Station],
    schools: List[School],
    time_matrix: List[List[float]]
) -> Solution:
    st_dict = {s.idx: s for s in stations_list}
    rebuilt = []
    for r in routes:
        route_stations = [ev for ev in r.events if isinstance(ev, Station)]
        if not route_stations:
            continue
        rebuilt.append(build_route(route_stations, schools, st_dict, time_matrix, auto_fill=True))
    return build_solution(rebuilt, stations_list)


def _destroy_random(sol: Solution, degree: float):
    all_st = [ev.idx for r in sol.routes for ev in r.events if isinstance(ev, Station)]
    unique_st = list(set(all_st))
    if not unique_st:
        return copy.deepcopy(sol), []
    n = max(1, int(len(unique_st) * degree))
    to_remove = set(random.sample(unique_st, n))
    new_routes = []
    for r in sol.routes:
        remaining = [ev for ev in r.events if isinstance(ev, Station) and ev.idx not in to_remove]
        if remaining:
            new_routes.append(Route(events=remaining))
    return Solution(new_routes, 0, 0, 0, 0, False), list(to_remove)


def _destroy_routes(sol: Solution, num: int = 1):
    if not sol.routes:
        return copy.deepcopy(sol), []
    num = min(len(sol.routes), num)
    indices = set(random.sample(range(len(sol.routes)), num))
    removed_st, new_routes = [], []
    for i, r in enumerate(sol.routes):
        if i in indices:
            removed_st.extend(ev.idx for ev in r.events if isinstance(ev, Station))
        else:
            new_routes.append(r)
    return Solution(new_routes, 0, 0, 0, 0, False), list(set(removed_st))


def _repair_greedy(
    sol: Solution,
    to_insert: List[int],
    schools: List[School],
    stations_list: List[Station],
    time_matrix: List[List[float]],
    group_map: dict,
    duals: List[float],
    mu: float
) -> Solution:
    """修復破壞後的解，插入順序依各站點總需求量由大到小排序（插入導引）。

    評估插入位置時改用 reduced cost 而非 solution_cost()，
    使修復方向與 pricing 目標一致。
    """
    from utils.solution_utils import route_max_load
    st_dict = {s.idx: s for s in stations_list}
    current_sol = _rebuild_solution(copy.deepcopy(sol).routes, stations_list, schools, time_matrix)

    # 需求量大的站點優先插入
    to_insert_sorted = sorted(
        to_insert,
        key=lambda idx: sum(st_dict[idx].demands.values()),
        reverse=True
    )

    for st_idx in to_insert_sorted:
        best_candidate: Optional[Solution] = None
        min_rc = float('inf')
        st_demands = st_dict[st_idx].demands

        # Option 1: 插入現有路線
        for r_idx, orig_route in enumerate(current_sol.routes):
            for p_idx in range(len(orig_route.events) + 1):
                test_events = list(orig_route.events)
                test_events.insert(p_idx, Station(
                    st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx
                ))
                test_route = build_route(test_events, schools, st_dict, time_matrix, auto_fill=True)
                if test_route.minutes > MAX_ROUTE_MIN or route_max_load(test_route) > BUS_CAPACITY:
                    continue
                # ★ 使用 reduced cost 評估插入後的效益
                rc = compute_reduced_cost(test_route, group_map, duals, mu)
                if rc < min_rc:
                    temp_routes = list(current_sol.routes)
                    temp_routes[r_idx] = test_route
                    min_rc = rc
                    best_candidate = build_solution(temp_routes, stations_list)

        # Option 2: 開新路線
        if len(current_sol.routes) < MAX_TOTAL_BUSES:
            new_events = [Station(st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx)]
            for sch_idx in st_demands.keys():
                new_events.append(School(sch_idx, schools[sch_idx].name, schools[sch_idx].orig_idx))
            new_r = build_route(new_events, schools, st_dict, time_matrix, auto_fill=True)
            if new_r.minutes <= MAX_ROUTE_MIN and route_max_load(new_r) <= BUS_CAPACITY:
                rc = compute_reduced_cost(new_r, group_map, duals, mu)
                if rc < min_rc:
                    temp_routes = list(current_sol.routes) + [new_r]
                    best_candidate = build_solution(temp_routes, stations_list)

        if best_candidate is not None:
            current_sol = best_candidate

    return build_solution(current_sol.routes, stations_list)


def _lns_pricing(
    init_sol: Solution,
    schools: List[School],
    stations: List[Station],
    time_matrix: List[List[float]],
    group_map: dict,
    duals: List[float],
    mu: float,
    iterations: int = LNS_PRICING_ITERATIONS
) -> List[Route]:
    """以 LNS 為定價引擎，批量產出 reduced cost < 0 的路線。

    與原始 LNS 的差異：
    1. 接受準則改為 reduced cost < 0（任何有用的 column 都收集）
    2. DESTROY_ROUTES 機率提高至 50%
    3. repair 的插入順序由需求量導引
    4. 回傳的是「所有找到的 rc < 0 路線列表」而非單一最佳解

    Args:
        init_sol: LNS 的起始解（通常是上一輪的最佳解或貪婪初始解）。
        schools, stations, time_matrix: 問題資料。
        group_map: (st_idx, sch_idx) -> gid。
        duals: 當前 RMP 對偶值。
        mu: 巴士數量限制對偶值。
        iterations: LNS 內部迭代次數。

    Returns:
        所有 reduced cost < 0 的新路線列表（可能為空）。
    """
    collected_routes: List[Route] = []
    seen_keys: set = set()

    def _extract_negative_rc_routes(sol: Solution) -> None:
        for r in sol.routes:
            key = tuple(
                (ev.idx, tuple(sorted(ev.demands.items())) if isinstance(ev, Station) else ev.idx)
                for ev in r.events
            )
            if key in seen_keys:
                continue
            rc = compute_reduced_cost(r, group_map, duals, mu)
            if rc < 0:
                seen_keys.add(key)
                collected_routes.append(r)
            

    current_sol = copy.deepcopy(init_sol)
    _extract_negative_rc_routes(current_sol)  # 初始解本身也掃一遍

    for _ in range(iterations):
        # 50% 機率整條路線破壞，50% 隨機站點破壞
        if random.random() < DESTROY_ROUTES_PROB and len(current_sol.routes) > 1:
            temp_sol, removed = _destroy_routes(current_sol, num=1)
        else:
            temp_sol, removed = _destroy_random(current_sol, DESTROY_DEGREE)

        if not removed:
            continue

        temp_sol = _repair_greedy(
            temp_sol, removed, schools, stations, time_matrix,
            group_map, duals, mu
        )

        #接受準則：只要修復後的解裡有 rc < 0 的路線就收集
        _extract_negative_rc_routes(temp_sol)

        # current_sol 持續更新為最新修復結果，讓下一輪破壞有新的起點
        current_sol = temp_sol

    return collected_routes


# ------------------------------------------------------------------------------
# Dantzig-Wolfe 求解器
# ------------------------------------------------------------------------------
class DantzigWolfeSolver(object):
    def __init__(self, schools: List[School], stations: List[Station], time_matrix: List[List[float]]):
        self.schools = schools
        self.stations = stations
        self.time_matrix = time_matrix
        self.st_dict = {s.idx: s for s in stations}

        self.groups = []
        self.group_map = {}
        for st in stations:
            for sch_idx, count in st.demands.items():
                gid = len(self.groups)
                self.groups.append(Station(idx=st.idx, name=st.name, demands={sch_idx: count}, orig_idx=st.orig_idx))
                event = self.groups[-1]
                self.group_map[(event.idx, list(event.demands.items())[0][0])] = gid

        self.routes = []
        self.route_pool_keys = set()
        self.group_to_routes = [[] for _ in range(len(self.groups))]
        self.duals = [0.0] * len(self.groups)
        self.mu = 0.0
        self.best_feasible_sol = None

        # LNS pricing 用的起始解，在 pricing 呼叫間持續更新
        self._lns_current_sol: Optional[Solution] = None

        self._prepare_pricing_cache()

    def _prepare_pricing_cache(self):
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

    def route_to_key(self, r: Route) -> str:
        key_parts = []
        for event in r.events:
            if isinstance(event, Station):
                for sch_idx in sorted(event.demands.keys()):
                    key_parts.append(f"G{self.group_map[(event.idx, sch_idx)]}")
            elif isinstance(event, School):
                key_parts.append(f"S{event.idx}")
        return "_".join(key_parts)

    def _add_route(self, r: Route) -> bool:
        g_set = self.route_to_key(r)
        if not g_set or g_set in self.route_pool_keys:
            return False
        route_idx = len(self.routes)
        self.routes.append(r)
        self.route_pool_keys.add(g_set)
        for event in r.events:
            if isinstance(event, Station):
                for sch_idx in event.demands.keys():
                    gid = self.group_map[(event.idx, sch_idx)]
                    self.group_to_routes[gid].append(route_idx)
        return True

    def initialize_columns(self):
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        init_sol = run_greedy(self.schools, self.stations, self.time_matrix)
        for r in init_sol.routes:
            self._add_route(r)
        # 同時作為 LNS pricing 的初始解
        self._lns_current_sol = init_sol

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
            idxs = self.group_to_routes[g_idx]
            constrs[g_idx] = m.addConstr(
                gp.quicksum(x[i] for i in idxs) == 1, name=f"Cover_G{g_idx}"
            )
        bus_constr = m.addConstr(
            gp.quicksum(x[i] for i in range(len(self.routes))) <= MAX_TOTAL_BUSES,
            name="MaxBus"
        )
        m.optimize()
        if relax and m.status == GRB.OPTIMAL:
            self.duals = [constrs[g].Pi for g in range(len(self.groups))]
            self.mu = bus_constr.Pi
        return m

    def pricing(self) -> bool:
        """以 LNS 為定價引擎，批量尋找 reduced cost < 0 的新路線。

        每次 pricing 呼叫會以當前的 duals/mu 驅動 LNS，
        收集所有有用的 column 並加入 pool。

        Returns:
            True 代表至少加入了一條新路線；False 代表 pool 無新增（視為收斂）。
        """
        # duals 與 mu 已由 solve_master 更新至 self.duals / self.mu
        new_routes = _lns_pricing(
            init_sol=self._lns_current_sol,
            schools=self.schools,
            stations=self.stations,
            time_matrix=self.time_matrix,
            group_map=self.group_map,
            duals=self.duals,
            mu=self.mu,
        )

        added_any = False
        for r in new_routes:
            if self._add_route(r):
                added_any = True

        if added_any:
            best_rc = min(
                compute_reduced_cost(r, self.group_map, self.duals, self.mu)
                for r in new_routes
            )
            tqdm.write(
                f"[Pricing] LNS 找到 {len(new_routes)} 條候選路線，"
                f"加入 {sum(1 for r in new_routes if self.route_to_key(r) in self.route_pool_keys)} 條新路線，"
                f"最小 RC={best_rc:.3f}"
            )

        return added_any

    def run(self) -> Solution:
        self.initialize_columns()

        for it in tqdm(range(DW_ITERATIONS), desc="DW/CG Iterations"):
            m = self.solve_master(relax=True)
            if m.status != GRB.OPTIMAL:
                tqdm.write(f"[DW Iter {it:03d}] Master problem not optimal. Status: {m.status}")
                break
            if it % 10 == 0:
                tqdm.write(f"[DW Iter {it:03d}] RMP 目標值 (Relaxed): {m.objVal:.2f}")

            # pricing 回傳 False 代表找不到任何 rc < 0 的路線，收斂
            if not self.pricing():
                tqdm.write(f"[DW Iter {it:03d}] 找不到負縮減成本路線，演算法收斂。")
                break
                

        # 最終整數規劃
        m = self.solve_master(relax=False)
        if m.status != GRB.OPTIMAL:
            return build_solution(routes=[], stations=self.stations)
        x = m.getVars()
        selected = [self.routes[i] for i in range(len(self.routes)) if x[i].X > 0.5]
        return build_solution(selected, self.stations)


def run_dantzig_wolfe(
    schools: List[School],
    stations: List[Station],
    time_matrix: List[List[float]]
) -> Solution:
    solver = DantzigWolfeSolver(schools, stations, time_matrix)
    return solver.run()


if __name__ == "__main__":
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(
        stops_csv="./data/stops-uniform_25+10.csv",
        time_csv="./data/time-uniform_25+10.csv"
    )
    best_sol = run_dantzig_wolfe(schools, stations, time_matrix)
    print_solution_pretty(best_sol, stations, schools)