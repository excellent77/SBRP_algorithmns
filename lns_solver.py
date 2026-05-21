from typing import List, Tuple, Optional
import random, copy

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import simulate_route, solution_cost, build_solution, try_merge_routes, route_max_load, print_solution_pretty
from aco_solver import aco_construct_solution
from utils.instance_generator import gen_instance_multi, load_instance_from_csv

# LNS 專用參數
BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0
MAX_TOTAL_BUSES = 6        # 總派車上限
LNS_ITERATIONS = 300    # LNS 迭代次數
DESTROY_DEGREE = 0.5   # 每次破壞%的站點
SHOW_MAP = True


def rebuild_solution(routes: List[Route], stations_list: List[Station], schools: List[School]) -> Solution:
    """重新模擬路線，確保 minutes/load/pickup_detail 都是最新狀態。"""
    st_dict = {s.idx: s for s in stations_list}
    rebuilt_routes = []
    for r in routes:
        if not any(ev[0] == 'pickup' for ev in r.events):
            continue
        rebuilt = Route(events=list(r.events))
        rebuilt.minutes, rebuilt.in_vehicle_minutes, rebuilt.pickup_detail, rebuilt.fairness_penalty = simulate_route(
            rebuilt, schools, st_dict, auto_fill=True
        )
        rebuilt_routes.append(rebuilt)
    return build_solution(rebuilt_routes, stations_list)


def relaxed_cost(sol: Solution, stations_list: List[Station]) -> float:
    """給中間解使用的成本；即使暫時未覆蓋全部需求，也會懲罰違規。"""
    total_demand = sum(sum(s.demands.values()) for s in stations_list)
    missing = max(0, total_demand - sol.students_served)
    cap_over = 0
    time_over = 0.0
    for r in sol.routes:
        load = route_max_load(r)
        cap_over += max(0, load - BUS_CAPACITY)
        time_over += max(0.0, r.minutes - MAX_ROUTE_MIN)
    return (
        sol.total_in_vehicle_minutes
        + sol.total_minutes * 10
        + sol.fairness_penalty * 100
        + len(sol.routes) * 1000
        + missing * 100000
        + cap_over * 100000
        + time_over * 100000
    )


# ========= [LNS 核心] 初始解生成：簡單貪婪法 =========
def get_initial_solution(schools: List[School], stations: List[Station]) -> Solution:
    """利用最簡單的『近鄰法』產生一個初始可行解"""
    # 這裡借用原有的 ACO 構造邏輯，但將費洛蒙設為常數，使其退化為純貪婪搜尋
    n = 1 + len(stations)
    dummy_tau = [[1.0]*n for _ in range(n)]
    dummy_eta = [[1.0]*n for _ in range(n)]
    sol = aco_construct_solution(schools, stations, dummy_tau, dummy_eta)
    return try_merge_routes(sol, stations, schools, auto_fill=True)

# ========= [LNS 核心] 破壞算子：隨機移除站點 =========
def destroy_random(sol: Solution, degree: float) -> Tuple[Solution, List[int]]:
    """隨機從現有路線中移除一定比例的站點，回傳新解與被移除的站點清單"""
    all_served_stations = []
    for r in sol.routes:
        for ev in r.events:
            if ev[0] == 'pickup':
                all_served_stations.append(ev[1][0])
    
    unique_served_stations = list(set(all_served_stations))
    if not unique_served_stations:
        return copy.deepcopy(sol), [] # No stations to remove

    num_to_remove = int(len(unique_served_stations) * degree)
    if num_to_remove == 0 and unique_served_stations: # Ensure at least one is removed if possible
        num_to_remove = 1

    to_remove = random.sample(unique_served_stations, num_to_remove)
    
    new_routes = []
    for r in sol.routes:
        new_events = []
        for ev in r.events:
            if ev[0] == 'pickup':
                st_idx = ev[1][0]
                if st_idx not in to_remove:
                    new_events.append(ev)
            else: # drop
                new_events.append(ev)
        
        # 重新清理路線（移除沒有乘客的 drop 或空的 pickup）
        if new_events:
            r_new = Route(events=new_events)
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
                if ev[0] == 'pickup':
                    removed_stations.append(ev[1][0])
        else:
            new_routes.append(r)
            
    new_sol = Solution(new_routes, 0, 0, 0, 0, False)
    return new_sol, list(set(removed_stations))

# ========= [LNS 核心] 重建算子：貪婪插入 =========
def repair_greedy(sol: Solution, to_insert: List[int], schools: List[School], stations_list: List[Station]) -> Solution:
    """將被移除的站點重新插回最合適的路徑中"""
    st_dict = {s.idx: s for s in stations_list}
    current_sol = rebuild_solution(copy.deepcopy(sol).routes, stations_list, schools)
    
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
                test_events.insert(p_idx, ('pickup', (st_idx, st_demands)))
                test_route = Route(events=test_events)
                
                mins, ivm, detail, fairness = simulate_route(test_route, schools, st_dict, auto_fill=True)
                load = route_max_load(test_route)
                
                if mins <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    # 建立臨時 Solution 來計算完整成本（包含 BUS_WEIGHT, FAIRNESS_WEIGHT）
                    temp_routes = list(current_sol.routes)
                    updated_route = Route(events=test_events, minutes=mins, in_vehicle_minutes=ivm, 
                                          pickup_detail=detail, fairness_penalty=fairness)
                    temp_routes[route_idx] = updated_route

                    temp_sol_candidate = build_solution(temp_routes, stations_list)

                    candidate_cost = relaxed_cost(temp_sol_candidate, stations_list)
                    if candidate_cost < min_total_cost:
                        min_total_cost = candidate_cost
                        best_candidate = temp_sol_candidate

        # Option 2: Create a new route for the station
        if len(current_sol.routes) < MAX_TOTAL_BUSES:
            temp_routes = list(current_sol.routes)
            
            new_route_events = [('pickup', (st_idx, st_demands))]
            schools_to_drop = list(st_dict[st_idx].demands.keys())
            for sch_idx in schools_to_drop:
                new_route_events.append(('drop', sch_idx))

            new_r = Route(events=new_route_events)
            new_r.minutes, new_r.in_vehicle_minutes, new_r.pickup_detail, new_r.fairness_penalty = simulate_route(new_r, schools, st_dict, auto_fill=True)
            
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
def run_lns(schools: List[School], stations: List[Station], print_log:bool=True) -> Solution:
    print("\n[LNS] 正在生成初始解...")
    best_sol = get_initial_solution(schools, stations)
    current_sol = copy.deepcopy(best_sol)
    
    if print_log:
        print(f"[LNS] 初始解成本: {solution_cost(best_sol):.1f}")
    
    for it in range(1, LNS_ITERATIONS + 1):
        # 1. 破壞
        # 混合使用隨機移除與路線移除，提高減車機會
        if random.random() < 0.3 and len(current_sol.routes) > 1:
            temp_sol, removed_stations = destroy_routes(current_sol, 1)
        else:
            temp_sol, removed_stations = destroy_random(current_sol, DESTROY_DEGREE)

        # 2. 重建
        temp_sol = repair_greedy(temp_sol, removed_stations, schools, stations)
        
        # 3. 嘗試合併路線優化
        temp_sol = try_merge_routes(temp_sol, stations, schools, auto_fill=True)
        
        # 4. 接受準則 (這裡使用簡單的 Hill Climbing，只接受更好的解)
        if temp_sol.feasible and solution_cost(temp_sol) < solution_cost(best_sol):
            best_sol = copy.deepcopy(temp_sol)
            current_sol = copy.deepcopy(temp_sol)
            if print_log:
                print(f"[Iter {it:03d}] 發現更優解 -> 車輛: {len(best_sol.routes)}, 總乘車時間: {best_sol.total_in_vehicle_minutes:.1f}")
        
        if print_log and it % 50 == 0:
            print(f"[Iter {it:03d}] 搜尋中... 當前最佳成本: {solution_cost(best_sol):.1f}")

    return best_sol



if __name__ == "__main__":
    # ========= 修改後的執行區塊 =========
    random.seed(42)  # 固定隨機種子
    csv_instance = load_instance_from_csv(stops_csv="./data/stops-b_7.csv", time_csv="./data/time-b_7.csv")
    if csv_instance is not None:
        schools, stations = csv_instance
        print("[DATA] loaded instance from CSV")
    else:
        schools, stations = gen_instance_multi()
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")

    # 呼叫 LNS 演算法
    best_lns_solution = run_lns(schools, stations)

    # 後處理與列印 (與原程式相同)
    print_solution_pretty(best_lns_solution, stations, schools)
