from typing import List, Tuple, Optional
import random, copy
from tqdm import tqdm

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import(
    BUS_CAPACITY, MAX_ROUTE_MIN, MAX_TOTAL_BUSES,
    build_route, build_solution, try_merge_routes, route_max_load, print_solution_pretty
)
from greedy_solver import run_greedy
from utils.instance_processer import load_instance_from_csv

LNS_ITERATIONS = 300
PENALTY_WEIGHT = 100000
DESTROY_DEGREE = 0.5
SHOW_MAP = True


def rebuild_solution(routes: List[Route], stations_list: List[Station], schools: List[School], time_matrix: List[List[float]]) -> Solution:
    """透過重新模擬每條路徑來重建解，確保所有屬性（時間、載重、取貨詳情）
    在修改後保持最新且一致。

    Args:
        routes: 要重建的 Route 物件列表。
        stations_list: 站點物件列表。
        schools: 學校物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。

    Returns:
        具有重建路徑的新 Solution 物件。
    """
    st_dict = {s.idx: s for s in stations_list}
    rebuilt_routes = []
    for r in routes:
        route_stations = [ev for ev in r.events if isinstance(ev, Station)]
        if not route_stations:
            continue
        rebuilt = build_route(route_stations, schools, st_dict, time_matrix, auto_fill=True)
        rebuilt_routes.append(rebuilt)
    return build_solution(rebuilt_routes, stations_list)


def relaxed_cost(sol: Solution, stations_list: List[Station]) -> float:
    """計算中間解的鬆弛成本，並對違規行為進行懲罰。

    此成本函數在 LNS 搜索期間用於引導演算法，即使解暫時不可行
    （例如：未服務所有學生、超過容量或違反時間限制）。

    Args:
        sol: 要評估的 Solution 物件。
        stations_list: 站點物件列表（用於計算總需求）。

    Returns:
        代表解的懲罰成本的浮點數。
    """
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


def get_initial_solution(schools: List[School], stations: List[Station], time_matrix: List[List[float]]) -> Solution:
    """使用貪婪方法生成初始可行解。

    此函數通常使用簡單的貪婪求解器，然後嘗試合併路徑以提高初始解的品質。

    Args:
        schools: 學校物件列表。
        stations: 站點物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。

    Returns:
        代表初始可行解的 Solution 物件。
    """
    sol = run_greedy(schools, stations, time_matrix)
    return try_merge_routes(sol, stations, schools, auto_fill=True)

def destroy_random(sol: Solution, degree: float) -> Tuple[Solution, List[int]]:
    """透過隨機移除一定比例的唯一站點來破壞解。

    Args:
        sol: 當前的 Solution 物件。
        degree: 要移除的唯一站點百分比（0 到 1 之間的浮點數）。

    Returns:
        包含以下內容的元組：
            - 移除了指定站點的新 Solution 物件。
            - 被移除站點的原始索引列表。
    """
    all_served_stations = []
    for r in sol.routes:
        for ev in r.events:
            if isinstance(ev, Station):
                all_served_stations.append(ev.idx)
    unique_served_stations = list(set(all_served_stations))
    if not unique_served_stations:
        return copy.deepcopy(sol), [] # No stations to remove
    num_to_remove = int(len(unique_served_stations) * degree)
    if num_to_remove == 0 and unique_served_stations: # Ensure at least one is removed if possible
        num_to_remove = 1
    to_remove = random.sample(unique_served_stations, num_to_remove)
    
    new_routes = []
    for r in sol.routes:
        remaining_stations = [ev for ev in r.events if isinstance(ev, Station) and ev.idx not in to_remove]
        if not remaining_stations:
            continue
        r_new = Route(events=remaining_stations)
        new_routes.append(r_new)

    # 更新新解的屬性 (這些會在 repair_greedy 之後重新計算)
    new_sol = Solution(new_routes, 0, 0, 0, 0, False)
    return new_sol, to_remove

def destroy_routes(sol: Solution, num_to_remove: int = 1) -> Tuple[Solution, List[int]]:
    """透過隨機移除整條路徑來破壞解。

    此算子旨在透過強制重新分配被移除路徑中的站點，來鼓勵減少巴士數量。

    Args:
        sol: 當前的 Solution 物件。
        num_to_remove: 要隨機移除的路徑數量。

    Returns:
        包含以下內容的元組：
            - 移除了指定路徑的新 Solution 物件。
            - 被移除路徑上站點的原始索引列表。
    """
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
    new_sol = Solution(new_routes, 0, 0, 0, 0, False)
    return new_sol, list(set(removed_stations))

def repair_greedy(sol: Solution, to_insert: List[int], schools: List[School], stations_list: List[Station], time_matrix:List[List[float]]) -> Solution:
    """透過貪婪地重新插入被移除的站點來修復被破壞的解。

    對於每個要插入的站點，它會嘗試在現有路徑或建立新路徑中找到最佳位置，
    在尊重容量和時間限制的同時使鬆弛成本最小化。

    Args:
        sol: 部分被破壞的 Solution 物件。
        to_insert: 需要重新插入的站點原始索引列表。
        schools: 學校物件列表。
        stations_list: 站點物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。

    Returns:
        所有來自 `to_insert` 的站點都已重新插入的新 Solution 物件。
    """
    st_dict = {s.idx: s for s in stations_list}
    current_sol = rebuild_solution(copy.deepcopy(sol).routes, stations_list, schools, time_matrix)
    
    random.shuffle(to_insert) # 增加隨機性
    
    # For each station to insert, find the best place to insert it
    for st_idx in to_insert:
        best_candidate: Optional[Solution] = None
        min_total_cost = float('inf')
        st_demands = st_dict[st_idx].demands

        # Option 1: Insert into an existing route
        for route_idx, original_route in enumerate(current_sol.routes):
            for p_idx in range(len(original_route.events) + 1): # Try all insertion points
                # 測試插入：建立該路線的臨時副本進行模擬，避免對整個解做 deepcopy 以提升效能
                test_events = list(original_route.events)
                test_events.insert(p_idx, Station(st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx))
                test_route = build_route(test_events, schools, st_dict, time_matrix, auto_fill=True)
                load = route_max_load(test_route)
                
                if test_route.minutes <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    # 建立臨時 Solution 來計算完整成本（包含 BUS_WEIGHT, FAIRNESS_WEIGHT）
                    temp_routes = list(current_sol.routes)
                    temp_routes[route_idx] = test_route

                    temp_sol_candidate = build_solution(temp_routes, stations_list)
                    candidate_cost = relaxed_cost(temp_sol_candidate, stations_list)
                    if candidate_cost < min_total_cost:
                        min_total_cost = candidate_cost
                        best_candidate = temp_sol_candidate

        # Option 2: Create a new route for the station
        if len(current_sol.routes) < MAX_TOTAL_BUSES:
            temp_routes = list(current_sol.routes)
            
            new_route_events = [Station(st_idx, st_dict[st_idx].name, st_demands, st_dict[st_idx].orig_idx)]
            schools_to_drop = list(st_dict[st_idx].demands.keys())
            for sch_idx in schools_to_drop:
                new_route_events.append(School(sch_idx, schools[sch_idx].name, schools[sch_idx].orig_idx))

            new_r = build_route(new_route_events, schools, st_dict, time_matrix, auto_fill=True)
            new_route_load = route_max_load(new_r)
            if new_r.minutes <= MAX_ROUTE_MIN and new_route_load <= BUS_CAPACITY:
                temp_routes.append(new_r)

                temp_sol_candidate = build_solution(temp_routes, stations_list)

                candidate_cost = relaxed_cost(temp_sol_candidate, stations_list)
                if candidate_cost < min_total_cost:
                    min_total_cost = candidate_cost
                    best_candidate = temp_sol_candidate
        # 應用該站點的最佳插入決策
        if best_candidate is not None:
            current_sol = best_candidate
    # 最終返回前，由 build_solution 根據完整性自動評估最終可行性
    return build_solution(current_sol.routes, stations_list)

def run_lns(schools: List[School], stations: List[Station], time_matrix:List[List[float]], print_log:bool=True) -> Solution:
    """執行大鄰域搜索 (LNS) 演算法來解決 SBRP。

    LNS 迭代地破壞當前解的一部分，然後貪婪地修復它們，接受更好的解。
    它還包含路徑合併步驟。

    Args:
        schools: 學校物件列表。
        stations: 站點物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。
        print_log: 布林值，指示是否在執行期間列印進度日誌。

    Returns:
        代表 LNS 找到的最佳解之 Solution 物件。
    """
    print("\n[LNS] 正在生成初始解...")
    best_sol = get_initial_solution(schools, stations, time_matrix)
    current_sol = copy.deepcopy(best_sol)
    if print_log:
        print(f"[LNS] 初始解成本: {best_sol.solution_cost():.1f}")
    for it in tqdm(range(1, LNS_ITERATIONS + 1), desc="LNS Iterations"):
        # 1. 破壞
        # 混合使用隨機移除與路線移除，提高減車機會
        if random.random() < 0.3 and len(current_sol.routes) > 1:
            temp_sol, removed_stations = destroy_routes(current_sol, 1)
        else:
            temp_sol, removed_stations = destroy_random(current_sol, DESTROY_DEGREE)
        # 2. 重建
        temp_sol = repair_greedy(temp_sol, removed_stations, schools, stations, time_matrix)
        # 3. 嘗試合併路線優化
        temp_sol = try_merge_routes(temp_sol, stations, schools, auto_fill=True)
        # 4. 接受準則 (這裡使用簡單的 Hill Climbing，只接受更好的解)
        if temp_sol.feasible and temp_sol.solution_cost() < best_sol.solution_cost():
            best_sol = copy.deepcopy(temp_sol)
            current_sol = copy.deepcopy(temp_sol)
            if print_log:
                print(f"[Iter {it:03d}] 發現更優解 -> 車輛: {len(best_sol.routes)}, 總乘車時間: {best_sol.total_in_vehicle_minutes:.1f}")
        
        if print_log and it % 50 == 0: # type: ignore
            print(f"[Iter {it:03d}] 搜尋中... 當前最佳成本: {best_sol.solution_cost():.1f}")

    return best_sol



if __name__ == "__main__":
    """
    運行大鄰域搜索 (LNS) 求解器的入口點。
    從 CSV 檔案載入實例數據，運行 LNS，並列印解。
    """
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-uniform_15+5.csv", time_csv="./data/time-uniform_15+5.csv")
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")
    best_lns_solution = run_lns(schools, stations, time_matrix)
    print_solution_pretty(best_lns_solution, stations, schools)