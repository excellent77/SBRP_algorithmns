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

# LNS 專用參數
LNS_ITERATIONS = 300    # LNS 迭代次數
PENALTY_WEIGHT = 100000 # 違規懲罰權重（用於 relaxed_cost 計算）
DESTROY_DEGREE = 0.5   # 每次破壞%的站點
SHOW_MAP = True


def rebuild_solution(routes: List[Route], stations_list: List[Station], schools: List[School], time_matrix: List[List[float]]) -> Solution:
    """重新模擬路線，確保 minutes/load/pickup_detail 都是最新狀態。"""
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
    """給中間解使用的成本；即使暫時未覆蓋全部需求，也會懲罰違規。"""
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


# ========= [LNS 核心] 初始解生成：簡單貪婪法 =========
def get_initial_solution(schools: List[School], stations: List[Station], time_matrix: List[List[float]]) -> Solution:
    """利用最簡單的『近鄰法』產生一個初始可行解"""
    sol = run_greedy(schools, stations, time_matrix)
    return try_merge_routes(sol, stations, schools, auto_fill=True)

# ========= [LNS 核心] 破壞算子：隨機移除站點 =========
def destroy_random(sol: Solution, degree: float) -> Tuple[Solution, List[int]]:
    """隨機從現有路線中移除一定比例的站點，回傳新解與被移除的站點清單"""
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

# ========= [LNS 核心] 破壞算子：整路拆除 (更有利於減車) =========
def destroy_routes(sol: Solution, num_to_remove: int = 1) -> Tuple[Solution, List[int]]:
    """隨機移除整條路線，強制將其站點分配到其他路線"""
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

# ========= [LNS 核心] 重建算子：貪婪插入 =========
def repair_greedy(sol: Solution, to_insert: List[int], schools: List[School], stations_list: List[Station], time_matrix:List[List[float]]) -> Solution:
    """將被移除的站點重新插回最合適的路徑中"""
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

# ========= LNS 主程式 =========
def run_lns(schools: List[School], stations: List[Station], time_matrix:List[List[float]], print_log:bool=True) -> Solution:
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
        
        if print_log and it % 50 == 0:
            print(f"[Iter {it:03d}] 搜尋中... 當前最佳成本: {best_sol.solution_cost():.1f}")

    return best_sol



if __name__ == "__main__":
    # ========= 修改後的執行區塊 =========
    random.seed(42)  # 固定隨機種子
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-uniform_15+5.csv", time_csv="./data/time-uniform_15+5.csv")
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")

    # 呼叫 LNS 演算法
    best_lns_solution = run_lns(schools, stations, time_matrix)

    # 後處理與列印 (與原程式相同)
    print_solution_pretty(best_lns_solution, stations, schools)