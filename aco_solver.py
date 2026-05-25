from typing import Dict, List, Optional
from collections import defaultdict
import itertools, math, random

from utils.data_models import School, Station, Route, Solution, Event, Coord

from utils.solution_utils import simulate_route, solution_cost, build_solution, try_merge_routes

BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0
MAX_TOTAL_BUSES = 10
TOTAL_STUDENTS = 200
SERVE_ALL = True

DEBUG_LOG = False
MAX_EVENTS_PER_ROUTE = 200

# ACO 參數（小規模給中等數值，跑得快）
ANTS = 16
ITERATIONS = 40
ALPHA = 1.0
BETA = 3.0
RHO = 0.35
TAU_INIT = 0.1
ELITE_RATE = 0.2
Q = 1000.0

def aco_construct_solution(schools: List[School], station_list: List[Station],
                           tau: List[List[float]], eta: List[List[float]]) -> Solution:
    """使用蟻群最佳化 (ACO) 啟發式演算法構建單個解（路徑集合）。

    此函數模擬一隻螞蟻透過在站點接送學生或在學校放下學生之間做出機率性選擇
    來構建路徑，同時考慮費洛蒙水平 (tau) 和啟發式訊息 (eta)。

    Args:
        schools: 學校物件列表。
        station_list: 站點物件列表。
        tau: 費洛蒙矩陣。
        eta: 啟發式訊息矩陣（行駛時間的倒數）。

    Returns:
        代表螞蟻所構建路徑的 Solution 物件。
    """
    stations = {s.idx: s for s in station_list}
    remaining: Dict[int, Dict[int,int]] = {s.idx: s.demands.copy() for s in station_list}
    total_students = sum(sum(d.values()) for d in remaining.values())

    routes: List[Route] = []
    students_served = 0
    station_ids = [s.idx for s in station_list]

    def any_remaining() -> bool:
        """檢查是否還有未服務的學生需求。"""
        return any(sum(m.values()) > 0 for m in remaining.values())

    def min_finish_time_from(pos: Coord, onboard_by_school: Dict[int, int]) -> float:
        """估計從給定位置送完車上所有學生所需的最短時間。

        Args:
            pos: 巴士的當前座標。
            onboard_by_school: 映射學校索引到該校車上學生人數的字典。

        Returns:
            送完所有學生的最短預估時間。
        """
        return finish_order_from(pos, onboard_by_school)[0]

    def finish_order_from(pos: Coord, onboard_by_school: Dict[int, int]):
        """計算從給定位置送完所有車上學生的最短時間和最佳順序。

        Returns:
            包含以下內容的元組：
                - 送完所有學生的最短時間。
                - 代表送客最佳學校索引順序的元組。
        """
        targets = [k for k, v in onboard_by_school.items() if v > 0]
        if not targets:
            return 0.0, ()
        best = float('inf')
        best_order = ()
        for order in itertools.permutations(targets):
            total = 0.0
            cur = pos
            for k in order:
                total += travel_minutes(cur, schools[k].coord)
                cur = schools[k].coord
            if total < best:
                best = total
                best_order = order
        return best, best_order

    while students_served < total_students and len(routes) < MAX_TOTAL_BUSES:
        events: List[Event] = []
        cur_coord: Optional[Coord] = None
        load_total = 0
        load_by_school: Dict[int,int] = defaultdict(int)
        minutes_used = 0.0
        stall = 0
        steps = 0

        while True:
            if steps > MAX_EVENTS_PER_ROUTE: # type: ignore
                if DEBUG_LOG: print("[route] reach MAX_EVENTS_PER_ROUTE, stop.") # type: ignore
                break

            candidate_stations = [i for i in station_ids if sum(remaining[i].values()) > 0]
            candidate_schools = [k for k,v in load_by_school.items() if v > 0]
            if not candidate_stations and not candidate_schools:
                break

            made_progress = False
            # ---- 決定：丟客 or 撿站 ----
            choose_drop = False
            if load_total >= BUS_CAPACITY - 5 and candidate_schools:
                choose_drop = True

            if not choose_drop and candidate_stations:
                if cur_coord is None:
                    best_i = min(candidate_stations,
                                 key=lambda i: min(travel_minutes(s.coord, stations[i].coord) for s in schools))
                    to_next = min(travel_minutes(s.coord, stations[best_i].coord) for s in schools)
                    pos_after = stations[best_i].coord
                else:
                    best_i = min(candidate_stations, key=lambda i: travel_minutes(cur_coord, stations[i].coord))
                    to_next = travel_minutes(cur_coord, stations[best_i].coord)
                    pos_after = stations[best_i].coord
                finish_min = min_finish_time_from(pos_after, load_by_school)
                if minutes_used + to_next + finish_min > MAX_ROUTE_MIN:
                    choose_drop = True
            # ---- 先丟客 ----
            if choose_drop:
                if not candidate_schools:
                    break
                if cur_coord is None: # Should not happen if choose_drop is True and candidate_schools exist
                    k = max(candidate_schools, key=lambda x: load_by_school[x])
                    next_coord = schools[k].coord
                    travel = 0.0
                else:
                    def drop_score(k_idx):
                        d = travel_minutes(cur_coord, schools[k_idx].coord)
                        return d / max(1, load_by_school[k_idx])
                    k = min(candidate_schools, key=drop_score)
                    next_coord = schools[k].coord
                    travel = travel_minutes(cur_coord, next_coord)

                if minutes_used + travel > MAX_ROUTE_MIN:
                    break

                events.append(('drop', k))
                minutes_used += travel
                cur_coord = next_coord
                load_total -= load_by_school[k]
                load_by_school[k] = 0
                made_progress = True
                stall = 0
                continue
            # ---- 撿站 ----
            if not candidate_stations:
                if candidate_schools and cur_coord is not None:
                    k = min(candidate_schools, key=lambda kk: travel_minutes(cur_coord, schools[kk].coord))
                    t2 = travel_minutes(cur_coord, schools[k].coord)
                    if minutes_used + t2 <= MAX_ROUTE_MIN:
                        events.append(('drop', k))
                        minutes_used += t2
                        cur_coord = schools[k].coord
                        load_total -= load_by_school[k]
                        load_by_school[k] = 0
                        made_progress = True
                if not made_progress:
                    break
                else:
                    stall = 0
                    continue
            probs, denom = [], 0.0
            for i in candidate_stations:
                if cur_coord is None:
                    d = sum(travel_minutes(sch.coord, stations[i].coord) for sch in schools)/len(schools)
                    t = tau[0][i]; e = 1.0/(d+1e-6)
                else:
                    d = travel_minutes(cur_coord, stations[i].coord)
                    t = tau[1][i]; e = 1.0/(d+1e-6) # Using tau[1] for station-to-station pheromone
                val = (t**ALPHA) * (e**BETA)
                probs.append((i, val)); denom += val
            if denom <= 0:
                pick = random.choice(candidate_stations)
            else:
                r_val = random.random()*denom; acc = 0.0; pick = candidate_stations[-1]
                for i,v in probs:
                    acc += v
                    if acc >= r_val: pick = i; break

            next_coord = stations[pick].coord
            travel = 0.0 if cur_coord is None else travel_minutes(cur_coord, next_coord)
            capacity_left = BUS_CAPACITY - load_total
            if capacity_left <= 0:
                if candidate_schools and cur_coord is not None:
                    k = min(candidate_schools, key=lambda kk: travel_minutes(cur_coord, schools[kk].coord))
                    t2 = travel_minutes(cur_coord, schools[k].coord)
                    if minutes_used + t2 <= MAX_ROUTE_MIN:
                        events.append(('drop', k))
                        minutes_used += t2
                        cur_coord = schools[k].coord
                        load_total -= load_by_school[k]
                        load_by_school[k] = 0
                        made_progress = True
                        stall = 0
                        continue
                break

            take_map: Dict[int,int] = {}
            remain_map = remaining[pick]
            for k, left in sorted(remain_map.items(), key=lambda kv: -kv[1]):
                if left <= 0: continue
                c = min(left, capacity_left)
                if c > 0:
                    take_map[k] = c
                    capacity_left -= c
                if capacity_left <= 0: break
            if not take_map:
                candidate_stations.remove(pick)
                stall += 1
                if stall >= 5:
                    break
                continue

            projected_load_by_school = load_by_school.copy()
            for k, c in take_map.items():
                projected_load_by_school[k] += c
            finish_min = min_finish_time_from(next_coord, projected_load_by_school)
            if minutes_used + travel + finish_min > MAX_ROUTE_MIN:
                if candidate_schools and cur_coord is not None:
                    k = min(candidate_schools, key=lambda kk: travel_minutes(cur_coord, schools[kk].coord))
                    t2 = travel_minutes(cur_coord, schools[k].coord)
                    if minutes_used + t2 <= MAX_ROUTE_MIN:
                        events.append(('drop', k))
                        minutes_used += t2
                        cur_coord = schools[k].coord
                        load_total -= load_by_school[k]
                        load_by_school[k] = 0
                        made_progress = True
                        stall = 0
                        continue
                candidate_stations.remove(pick)
                stall += 1
                if stall >= 5:
                    break
                continue

            events.append(('pickup', (pick, take_map)))
            minutes_used += travel
            cur_coord = next_coord
            for k,c in take_map.items():
                remaining[pick][k] -= c
                load_by_school[k] += c
                load_total += c

            made_progress = True
            stall = 0

            if load_total >= BUS_CAPACITY:
                continue
            if minutes_used + min_finish_time_from(cur_coord, load_by_school) > MAX_ROUTE_MIN:
                continue

        # 路線收尾：若還有人在車上，丟到最近學校
        if events:
            onboard_est: Dict[int,int] = defaultdict(int)
            for et,data in events:
                if et == 'pickup':
                    for k,c in data[1].items():  # type: ignore
                        onboard_est[k] += c
                else:
                    onboard_est[int(data)] = 0  # type: ignore
            remaining_onboard = {k:v for k,v in onboard_est.items() if v>0}
            if remaining_onboard:
                last_pos = (stations[events[-1][1][0]].coord if events[-1][0]=='pickup'
                            else schools[int(events[-1][1])].coord)  # type: ignore
                _, finish_order = finish_order_from(last_pos, remaining_onboard)
                for sch_idx in finish_order: # type: ignore
                    events.append(('drop', int(sch_idx)))
        else:
            break

        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(r, schools, stations)
        routes.append(r)

        served_now = sum(t for _,_,t,_ in r.pickup_detail)
        students_served += served_now # type: ignore
        if DEBUG_LOG: # type: ignore
            print(f"[route] events={len(events)} minutes={r.minutes:.1f} served={served_now} fairness={r.fairness_penalty:.1f}")

        if not any_remaining():
            break

    return build_solution(routes, station_list)

def run_aco(schools: List[School], stations: List[Station]) -> Solution:
    """執行蟻群最佳化 (ACO) 演算法來解決 SBRP。

    此函數初始化費洛蒙和啟發式矩陣，然後迭代地派遣螞蟻構建解，
    並根據所找到解的品質更新費洛蒙路徑。

    Args:
        schools: 學校物件列表。
        stations: 站點物件列表。

    Returns:
        代表 ACO 找到的最佳解之 Solution 物件。

    Raises:
        RuntimeError: 如果 ACO 未能找到任何解。
    """
    center = (sum(s.coord[0] for s in schools)/len(schools),
              sum(s.coord[1] for s in schools)/len(schools))
    idx_to_coord: Dict[int,Coord] = {0:center} # Dummy node 0 for school/depot
    for st in stations: idx_to_coord[st.idx] = st.coord
    n = 1 + len(stations) # Node 0 is dummy school, then stations
    eta = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j:
                eta[i][j]=0.0
            else:
                try:
                    eta[i][j]=1.0/(travel_minutes(idx_to_coord[i], idx_to_coord[j])+1e-6)
                except Exception:
                    eta[i][j]=0.0
    tau = [[TAU_INIT]*n for _ in range(n)]

    best: Optional[Solution] = None
    for it in range(1, ITERATIONS+1):
        sols=[]
        for _ in range(ANTS):
            s = aco_construct_solution(schools, stations, tau, eta)
            # It's common to apply local search or post-processing like route merging
            # after an ant constructs a solution, or at the end of the algorithm.
            s = try_merge_routes(s, stations, schools) # type: ignore
            
            sols.append(s)
            if (best is None) or solution_cost(s) < solution_cost(best):
                best = s
                print(f"[Iter {it:02d}] New Best -> buses={len(best.routes)}, served={best.students_served}/{TOTAL_STUDENTS}, "
                      f"ivm={best.total_in_vehicle_minutes:.1f}, cost={solution_cost(best):.1f}")

        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1.0 - RHO)
        feas = [z for z in sols if z.feasible]
        feas.sort(key=lambda x: solution_cost(x))
        elite_k = max(1, int(ELITE_RATE*len(feas))) if feas else 0
        for s in feas[:elite_k]:
            cost = solution_cost(s)
            if not math.isfinite(cost) or cost<=0: continue
            delta = Q / cost
            for r in s.routes:
                prev = 0 # type: ignore
                for ev in r.events:
                    if ev[0]=='pickup':
                        st_idx = ev[1][0]  # type: ignore
                        tau[prev][st_idx] += delta
                        prev = st_idx
                tau[prev][0] += delta # type: ignore

        print(f"[Iter {it:02d}] best_cost={solution_cost(best):.1f}, buses={len(best.routes)}, "
              f"served={best.students_served}/{TOTAL_STUDENTS}, ivm={best.total_in_vehicle_minutes:.1f}")

    if best is None:
        raise RuntimeError("ACO failed to find any solution.")
    return best # type: ignore


if __name__ == "__main__":
    """
    運行蟻群最佳化 (ACO) 求解器的入口點。
    載入實例數據，運行 ACO，列印解，進行稽核並生成地圖。
    """
    from utils.instance_processer import gen_instance_multi, load_default_instance_from_csv
    from utils.solution_utils import print_solution_pretty, audit_solution, plot_routes_on_map
    print("=== Starting ACO Algorithm Test ===")
    random.seed(42)
    csv_instance = load_default_instance_from_csv()
    if csv_instance is not None:
        schools, stations = csv_instance
        print(f"[DATA] loaded instance from CSV: stops-b_7.csv + time-b_7.csv")
    else:
        schools, stations = gen_instance_multi()
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")

    # 2. 執行 ACO 核心邏輯
    best_aco_solution = run_aco(schools, stations)

    # 3. 輸出結果與稽核
    print_solution_pretty(best_aco_solution, stations, schools)
    audit_solution(best_aco_solution, stations, schools)
    m = plot_routes_on_map(best_aco_solution, stations, schools, title="ACO Optimized")
    m.save("./data/aco_routes.html")
    print("已輸出地圖")
