from typing import List, Tuple, Optional
import random, copy, math
from tqdm import tqdm

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    BUS_CAPACITY, MAX_ROUTE_MIN, MAX_TOTAL_BUSES,
    build_route, build_solution, try_merge_routes, route_max_load, print_solution_pretty
)
from greedy_solver import run_greedy
from utils.instance_processer import load_instance_from_csv

LNS_ITERATIONS = 500
PENALTY_WEIGHT = 100000
DESTROY_DEGREE = 0.5
SHOW_MAP = True

SA_INITIAL_TEMP  = 5.0   # 初始溫度：控制初期接受劣解的機率
SA_COOLING_RATE  = 0.995 # 冷卻速率：每次迭代乘以此值，0.995
SA_MIN_TEMP      = 0.01  # 最低溫度：低於此值後不再接受劣解



def rebuild_solution(routes, stations_list, schools, time_matrix):
    st_dict = {s.idx: s for s in stations_list}
    rebuilt_routes = []
    for r in routes:
        route_stations = [ev for ev in r.events if isinstance(ev, Station)]
        if not route_stations:
            continue
        rebuilt = build_route(route_stations, schools, st_dict, time_matrix, auto_fill=True)
        rebuilt_routes.append(rebuilt)
    return build_solution(rebuilt_routes, stations_list)


def relaxed_cost(sol, stations_list):
    total_demand = sum(sum(s.demands.values()) for s in stations_list)
    missing = max(0, total_demand - sol.students_served)
    cap_over = 0
    time_over = 0.0
    base_cost = sum(r.route_cost() for r in sol.routes)
    for r in sol.routes:
        load = route_max_load(r)
        cap_over += max(0, load - BUS_CAPACITY)
        time_over += max(0.0, r.minutes - MAX_ROUTE_MIN)
    return (
        base_cost
        + missing * PENALTY_WEIGHT
        + cap_over * PENALTY_WEIGHT
        + time_over * PENALTY_WEIGHT
    )


def get_initial_solution(schools, stations, time_matrix):
    sol = run_greedy(schools, stations, time_matrix)
    return try_merge_routes(sol, stations, schools, auto_fill=True)


def destroy_random(sol, degree):
    all_served_stations = []
    for r in sol.routes:
        for ev in r.events:
            if isinstance(ev, Station):
                all_served_stations.append(ev.idx)
    unique_served_stations = list(set(all_served_stations))
    if not unique_served_stations:
        return copy.deepcopy(sol), []
    num_to_remove = int(len(unique_served_stations) * degree)
    if num_to_remove == 0 and unique_served_stations:
        num_to_remove = 1
    to_remove = random.sample(unique_served_stations, num_to_remove)
    new_routes = []
    for r in sol.routes:
        remaining_stations = [ev for ev in r.events if isinstance(ev, Station) and ev.idx not in to_remove]
        if not remaining_stations:
            continue
        r_new = Route(events=remaining_stations)
        new_routes.append(r_new)
    return Solution(new_routes, 0, 0, 0, 0, False), to_remove


def destroy_routes(sol, num_to_remove=1):
    if not sol.routes:
        return copy.deepcopy(sol), []
    num_to_remove = min(len(sol.routes), num_to_remove)
    indices = random.sample(range(len(sol.routes)), num_to_remove)
    removed_stations = []
    new_routes = []
    for i, r in enumerate(sol.routes):
        if i in indices:
            for ev in r.events:
                if isinstance(ev, Station):
                    removed_stations.append(ev.idx)
        else:
            new_routes.append(r)
    return Solution(new_routes, 0, 0, 0, 0, False), list(set(removed_stations))


def repair_greedy(sol, to_insert, schools, stations_list, time_matrix):
    st_dict = {s.idx: s for s in stations_list}
    current_sol = rebuild_solution(copy.deepcopy(sol).routes, stations_list, schools, time_matrix)
    random.shuffle(to_insert)
    for st_idx in to_insert:
        best_candidate = None
        min_total_cost = float('inf')
        st_demands = st_dict[st_idx].demands
        for route_idx, original_route in enumerate(current_sol.routes):
            for p_idx in range(len(original_route.events) + 1):
                test_events = list(original_route.events)
                test_events.insert(p_idx, Station(st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx))
                test_route = build_route(test_events, schools, st_dict, time_matrix, auto_fill=True)
                load = route_max_load(test_route)
                if test_route.minutes <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    temp_routes = list(current_sol.routes)
                    temp_routes[route_idx] = test_route
                    temp_sol_candidate = build_solution(temp_routes, stations_list)
                    candidate_cost = relaxed_cost(temp_sol_candidate, stations_list)
                    if candidate_cost < min_total_cost:
                        min_total_cost = candidate_cost
                        best_candidate = temp_sol_candidate
        if len(current_sol.routes) < MAX_TOTAL_BUSES:
            temp_routes = list(current_sol.routes)
            new_route_events = [Station(st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx)]
            for sch_idx in st_dict[st_idx].demands.keys():
                new_route_events.append(School(sch_idx, schools[sch_idx].name, schools[sch_idx].orig_idx))
            new_r = build_route(new_route_events, schools, st_dict, time_matrix, auto_fill=True)
            if new_r.minutes <= MAX_ROUTE_MIN and route_max_load(new_r) <= BUS_CAPACITY:
                temp_routes.append(new_r)
                temp_sol_candidate = build_solution(temp_routes, stations_list)
                candidate_cost = relaxed_cost(temp_sol_candidate, stations_list)
                if candidate_cost < min_total_cost:
                    min_total_cost = candidate_cost
                    best_candidate = temp_sol_candidate
        if best_candidate is not None:
            current_sol = best_candidate
    return build_solution(current_sol.routes, stations_list)


def run_lns(
    schools: List[School],
    stations: List[Station],
    time_matrix: List[List[float]],
    print_log: bool = True
) -> Solution:
    """LNS 求解器，接受準則改為 Simulated Annealing。

    與原版唯一的差異在步驟 4：
      原版：只接受嚴格更優的可行解（Hill Climbing）。
      本版：依 Boltzmann 機率決定是否接受較差的解，溫度隨迭代線性冷卻。
            溫度降至 SA_MIN_TEMP 後行為等同 Hill Climbing。
    best_sol 仍然只更新為歷史最優可行解，SA 的接受只作用在 current_sol。
    """
    print("\n[LNS-SA] 正在生成初始解...")
    best_sol = get_initial_solution(schools, stations, time_matrix)
    current_sol = copy.deepcopy(best_sol)

    # ── SA 狀態初始化 ─────────────────────────────────────────────────────────
    temp = SA_INITIAL_TEMP
    current_cost = best_sol.solution_cost()   # SA 用 solution_cost() 評估接受與否
    # ─────────────────────────────────────────────────────────────────────────

    if print_log:
        print(f"[LNS-SA] 初始解成本: {current_cost:.1f}  初始溫度: {temp:.3f}")

    for it in tqdm(range(1, LNS_ITERATIONS + 1), desc="LNS-SA Iterations"):
        # 1. 破壞
        if random.random() < 0.3 and len(current_sol.routes) > 1:
            temp_sol, removed_stations = destroy_routes(current_sol, 1)
        else:
            temp_sol, removed_stations = destroy_random(current_sol, DESTROY_DEGREE)

        # 2. 重建
        temp_sol = repair_greedy(temp_sol, removed_stations, schools, stations, time_matrix)

        # 3. 嘗試合併路線
        temp_sol = try_merge_routes(temp_sol, stations, schools, auto_fill=True)

        # 4. ── Simulated Annealing 接受準則 ───────────────────────────────────
        if temp_sol.feasible:
            new_cost = temp_sol.solution_cost()
            delta = new_cost - current_cost  # 負值 = 改善；正值 = 退步

            # 無條件接受改善；退步時依 Boltzmann 機率接受
            if delta < 0 or (temp > SA_MIN_TEMP and random.random() < math.exp(-delta / temp)):
                current_sol = copy.deepcopy(temp_sol)
                current_cost = new_cost

                # best_sol 只在真正改善時更新
                if new_cost < best_sol.solution_cost():
                    best_sol = copy.deepcopy(temp_sol)
                    if print_log:
                        print(
                            f"[Iter {it:03d}] 發現更優解 -> "
                            f"車輛: {len(best_sol.routes)}, "
                            f"總乘車時間: {best_sol.total_in_vehicle_minutes:.1f}, "
                            f"溫度: {temp:.4f}"
                        )
        # ─────────────────────────────────────────────────────────────────────

        # 5. 降溫
        temp = max(SA_MIN_TEMP, temp * SA_COOLING_RATE)

        if print_log and it % 50 == 0:
            print(
                f"[Iter {it:03d}] 搜尋中... "
                f"當前成本: {current_cost:.1f}  最佳成本: {best_sol.solution_cost():.1f}  溫度: {temp:.4f}"
            )

    return best_sol


if __name__ == "__main__":
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(
        stops_csv="./data/stops-b_7.csv",
        time_csv="./data/time-b_7.csv"
    )
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")
    best_lns_solution = run_lns(schools, stations, time_matrix)
    print_solution_pretty(best_lns_solution, stations, schools)