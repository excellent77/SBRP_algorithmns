from typing import Dict, List, Tuple
from collections import defaultdict
import folium

from utils.data_models import School, Station, Route, Solution, Event, Coord
from utils.geo_utils import travel_minutes, haversine_km

# ========= 參數區 =========
BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0
MAX_TOTAL_BUSES = 6
ROUTE_TIME_WEIGHT = 1.0
TOTAL_TIME_WEIGHT = 10.0
BUS_COUNT_WEIGHT = 1000.0 # 顯著提高權重以優先減少派車數
FAIRNESS_WEIGHT = 100.0      # 先後順序謬誤費權重

MAP_COLORS: List[str] = ["blue","green","purple","orange","darkred","cadetblue",
                         "darkgreen","darkpurple","lightred","lightgreen","lightblue","gray"]

# 防卡護欄
# MAX_EVENTS_PER_ROUTE = 200 # Not directly used here, but good to note if it were.


# ========= 路線模擬：由事件序列計算分鐘、人×分、每站-校乘車時間 =========
def simulate_route(route: Route, schools: List[School], stations: Dict[int, Station], auto_fill:bool=False) -> Tuple[float,float,List[Tuple[int,int,int,float]], float]:
    if not route.events:
        return 0.0, 0.0, [], 0.0

    def ev_coord(ev: Event) -> Coord:
        et, data = ev
        if et == 'pickup':
            st_idx, _ = data  # type: ignore
            return stations[st_idx].coord
        else:
            sch_idx = data  # type: ignore
            return schools[int(sch_idx)].coord

    minutes = 0.0
    fairness_penalty = 0.0
    ivm = 0.0
    pickup_detail: List[Tuple[int,int,int,float]] = []
    onboard_by_school: Dict[int,int] = defaultdict(int)
    batches: List[Dict] = []  # 每批上車紀錄：{'station','school','count','acc'}
    finished_batches: List[Dict] = [] # 已送達紀錄

    cur = ev_coord(route.events[0])
    for e in route.events:
        nxt = ev_coord(e)
        travel = travel_minutes(cur, nxt)
        minutes += travel
        total_onboard = sum(onboard_by_school.values())
        ivm += travel * total_onboard
        for b in batches: b['acc'] += travel

        et, data = e
        if et == 'pickup':
            st_idx, take_map = data  # type: ignore
            for k, c in take_map.items():
                onboard_by_school[k] += c
                batches.append({'station':st_idx, 'school':k, 'count':c, 'acc':0.0})
        else:  # drop
            sch_idx = int(data)  # type: ignore
            keep = []
            for b in batches:
                if b['school'] == sch_idx:
                    pickup_detail.append((b['station'], b['school'], b['count'], b['acc']))
                    b['drop_time'] = minutes
                    finished_batches.append(b)
                    onboard_by_school[sch_idx] -= b['count']
                else:
                    keep.append(b)
            batches = keep
        cur = nxt

    if batches:
        if auto_fill:
            # 補足邏輯：若路徑結束仍有人在車上，送往最近學校
            remaining_schools = [k for k, v in onboard_by_school.items() if v > 0]
            while remaining_schools:
                # 尋找離目前位置最近的目標學校
                nxt_sch_idx = min(remaining_schools, key=lambda k: travel_minutes(cur, schools[k].coord))
                nxt_coord = schools[nxt_sch_idx].coord
                travel = travel_minutes(cur, nxt_coord)
                
                minutes += travel
                total_onboard = sum(onboard_by_school.values())
                ivm += travel * total_onboard
                for b in batches: b['acc'] += travel
                
                # 執行 drop 邏輯
                keep = []
                for b in batches:
                    if b['school'] == nxt_sch_idx:
                        pickup_detail.append((b['station'], b['school'], b['count'], b['acc']))
                        b['drop_time'] = minutes
                        finished_batches.append(b)
                        onboard_by_school[nxt_sch_idx] -= b['count']
                    else:
                        keep.append(b)
                batches = keep
                cur = nxt_coord
                remaining_schools.remove(nxt_sch_idx)
        else:
            raise RuntimeError("路徑不完整")

    # 將批次按學校分類，因為公平性比較只發生在去往同一個學校的學生之間
    by_school = defaultdict(list)
    for b in finished_batches:
        by_school[b['school']].append(b)

    for sch_idx, school_batches in by_school.items():
        sch_coord = schools[sch_idx].coord
        # 比較該校所有學生的來源站點對 (i, j)
        for i in range(len(school_batches)):
            for j in range(len(school_batches)):
                if i == j: continue
                b_i = school_batches[i]
                b_j = school_batches[j]

                # 計算兩站點到該校的理論行駛距離（時間）
                dist_i = travel_minutes(stations[b_i['station']].coord, sch_coord)
                dist_j = travel_minutes(stations[b_j['station']].coord, sch_coord)
                
                if dist_i < dist_j and b_i['acc'] > b_j['acc']:
                    fairness_penalty += 1.0

    return minutes, ivm, pickup_detail, fairness_penalty

def build_solution(routes: List[Route], stations: List[Station]) -> Solution:
    """自動化建立 Solution 物件並識別其可行性"""
    total_min = sum(r.minutes for r in routes)
    total_ivm = sum(r.in_vehicle_minutes for r in routes)
    total_fairness = sum(r.fairness_penalty for r in routes)
    students_served = sum(sum(c for _, _, c, _ in r.pickup_detail) for r in routes)

    # 計算總需求以確認是否全數服務
    total_demand = sum(sum(s.demands.values()) for s in stations)
    
    is_feasible = True
    if not routes or students_served != total_demand or len(routes) > MAX_TOTAL_BUSES:
        is_feasible = False
    else:
        for r in routes:
            load = route_max_load(r)
            if load > BUS_CAPACITY or r.minutes > MAX_ROUTE_MIN + 1e-6:
                is_feasible = False
                break

    return Solution(
        routes=routes,
        total_minutes=total_min,
        total_in_vehicle_minutes=total_ivm,
        fairness_penalty=total_fairness,
        students_served=students_served,
        feasible=is_feasible
    )

# ========= 成本函式 =========
def solution_cost(sol: Solution) -> float:
    if not sol.feasible: return float('inf')
    return ROUTE_TIME_WEIGHT*sol.total_in_vehicle_minutes + BUS_COUNT_WEIGHT*len(sol.routes) + FAIRNESS_WEIGHT*sol.fairness_penalty + TOTAL_TIME_WEIGHT*sol.total_minutes

def route_cost(route:Route) -> float:
    return ROUTE_TIME_WEIGHT*route.in_vehicle_minutes + FAIRNESS_WEIGHT*route.fairness_penalty + TOTAL_TIME_WEIGHT*route.minutes + BUS_COUNT_WEIGHT

def route_max_load(route: Route) -> int:
    """計算路線任一時刻的最大車上人數。"""
    onboard_by_school: Dict[int, int] = defaultdict(int)
    max_load = 0
    for et, data in route.events:
        if et == 'pickup':
            _, take_map = data  # type: ignore
            for sch_idx, count in take_map.items():
                onboard_by_school[int(sch_idx)] += count
        else:
            sch_idx = int(data)  # type: ignore
            onboard_by_school[sch_idx] = 0
        max_load = max(max_load, sum(onboard_by_school.values()))
    return max_load

# ========= 併車（串接事件序列；再次模擬核對 60 分鐘） =========
def try_merge_routes(sol: Solution, stations_list: List[Station], schools: List[School], auto_fill:bool=False) -> Solution:
    stations = {s.idx:s for s in stations_list}
    routes = sol.routes[:]
    changed=True
    while changed:
        changed=False
        n=len(routes)
        for i in range(n):
            for j in range(i+1, n):
                r1, r2 = routes[i], routes[j]
                events = r1.events + r2.events
                r_tmp = Route(events=events)
                minutes, ivm, detail, fairness = simulate_route(r_tmp, schools, stations, auto_fill=auto_fill)
                load = route_max_load(r_tmp)
                if minutes <= MAX_ROUTE_MIN + 1e-6 and load <= BUS_CAPACITY:
                    r_tmp.minutes, r_tmp.in_vehicle_minutes, r_tmp.pickup_detail, r_tmp.fairness_penalty = minutes, ivm, detail, fairness
                    keep = [k for k in range(n) if k not in (i,j)]
                    routes = [routes[k] for k in keep] + [r_tmp]
                    changed=True
                    break
            if changed: break

    return build_solution(routes, stations_list)

# ========= 稽核 =========
def audit_solution(sol: Solution, station_list: List[Station], schools: List[School]):
    need = sum(sum(s.demands.values()) for s in station_list)
    got = sum(c for r in sol.routes for _,_,c,_ in r.pickup_detail)
    per_station_served = defaultdict(int)
    for r in sol.routes:
        for st,sch,c,_ in r.pickup_detail:
            per_station_served[st]+=c
    not_served = [s.idx for s in station_list if per_station_served[s.idx]==0 and sum(s.demands.values())>0]
    print("=== 稽核 ===")
    print(f"總需求 vs 總實載：{need} vs {got}")
    print(f"疑似未服務站數：{len(not_served)} -> {not_served[:12]}{' ...' if len(not_served)>12 else ''}")

# ========= 列印（圖二樣式；同站多校分行顯示） =========
def print_solution_pretty(sol: Solution, stations: List[Station], schools: List[School]):
    s_map = {s.idx: s for s in stations}
    sch_name = {s.idx: s.name for s in schools}

    # 標頭
    print("======  解方案摘要  ======")
    print(f"是否可行：{sol.feasible}")
    print(f"總使用車輛數：{len(sol.routes)}")
    print(f"總服務人數：{sol.students_served}")
    print(f"總路線時間(分)：{sol.total_minutes:.1f}")
    print(f"總乘車時間(人×分)：{sol.total_in_vehicle_minutes:.1f}")
    print(f"不公平費用得分：{sol.fairness_penalty:.1f}")
    print(f"總分數: {TOTAL_TIME_WEIGHT*sol.total_minutes + BUS_COUNT_WEIGHT*len(sol.routes) + FAIRNESS_WEIGHT*sol.fairness_penalty + ROUTE_TIME_WEIGHT*sol.total_in_vehicle_minutes:.1f}")
    print("-" * 60)

    # 動態欄寬
    station_w = max(6, max(len(s.name) for s in stations) if stations else 6)

    for b, r in enumerate(sol.routes, 1):
        total_load = sum(c for _, _, c, _ in r.pickup_detail)
        print(f"Bus{b}: 共載 {total_load} 人，總行駛時間={r.minutes:.1f} 分鐘")
        print("-" * 60)

        # 1) 依「撿站順序」輸出（從 events 取得順序最準）
        pick_order: List[int] = []
        for ev in r.events:
            if ev[0] == 'pickup':
                st_idx = ev[1][0]  # type: ignore
                if st_idx not in pick_order:
                    pick_order.append(st_idx)

        # 2) 將同一站不同校的批次合併： {station_idx: [(school_idx, count_sum, ride_sum or avg?)]}
        #   這裡依你範例，把每校各自的 (人數、乘車時間) 一組組列出；乘車時間使用實際該批次的 ride（同校多批次就逐筆列）。
        grouped: Dict[int, List[Tuple[int,int,float]]] = defaultdict(list)
        # 先把同站同校的多批「合併為一筆」：人數加總、乘車時間採加權平均（以人數為權重）
        tmp: Dict[Tuple[int,int], Tuple[int,float]] = defaultdict(lambda: (0,0.0))
        for st, k, c, ride in r.pickup_detail:
            prev_c, prev_w = tmp[(st,k)]
            new_c = prev_c + c
            new_w = prev_w + ride * c
            tmp[(st,k)] = (new_c, new_w)
        for (st,k), (cc, ww) in tmp.items():
            grouped[st].append((k, cc, ww/cc if cc>0 else 0.0))

        # 3) 依撿站順序印出
        line_no = 1
        for st in pick_order:
            items = grouped.get(st, [])
            if not items:
                continue
            # School_k,載 xx 人 (乘車 yy.y 分)；多校用頓號串接
            right = "、".join([f"{sch_name.get(k, f'School_{k+1}')},載 {c:>2} 人 (乘車 {ride:>5.1f} 分, 距離 {haversine_km(s_map[st].coord, schools[k].coord):>5.2f} km)"
                               for (k, c, ride) in items])
            print(f"{line_no:02d}. {s_map[st].name:<{station_w}} → {right}")
            line_no += 1

        # 4) 路徑摘要： 站點序列 -> 學校序列
        # 學校依實際到達時間（即在 pickup_detail 出現順序）排序，包含自動補足的學校
        seen = set()
        school_seq = []
        for _, sch_idx, _, _ in r.pickup_detail:
            if sch_idx not in seen:
                school_seq.append(sch_idx)
                seen.add(sch_idx)

        left = " -> ".join([s_map[st].name for st in pick_order])
        right = " -> ".join([sch_name.get(k, f"School_{k+1}") for k in school_seq])
        if left and right:
            print("")
            print(f"{left} -> {right}")
        elif left:  # 有撿站、但沒明確丟學校事件（理論上不會）
            print("")
            print(left)
        elif right:
            print("")
            print(right)

        print("-" * 60)
    print("")

# ========= 地圖 =========
def plot_routes_on_map(sol: Solution, stations: List[Station], schools: List[School], title: str = "Solution") -> folium.Map:
    center = (sum(s.coord[0] for s in schools)/len(schools),
              sum(s.coord[1] for s in schools)/len(schools))
    m = folium.Map(location=center, zoom_start=12, control_scale=True)

    # 學校標記
    color_icon = ["red","darkblue","black"]
    for sch in schools:
        folium.Marker(sch.coord, tooltip=sch.name,
                      icon=folium.Icon(color=color_icon[sch.idx%len(color_icon)],
                                       icon="graduation-cap", prefix="fa")).add_to(m)

    # 先畫所有站（灰）；tooltip 顯示各校需求拆分
    for s in stations:
        tt = sum(s.demands.values())
        folium.CircleMarker(s.coord, radius=4, color="#999", fill=True, fill_opacity=0.5,
                            tooltip=f"{s.name} 需求={tt} {s.demands}").add_to(m)

    # 依事件畫路線與載人站的彩色圈
    s_map = {s.idx:s for s in stations}
    for b, r in enumerate(sol.routes, 1):
        color = MAP_COLORS[(b-1)%len(MAP_COLORS)]
        pts=[]
        onboard = defaultdict(int)
        for ev in r.events:
            if ev[0]=='pickup':
                st_idx, take_map = ev[1]  # type: ignore
                coord = s_map[st_idx].coord
                pts.append(coord)
                for k, v in take_map.items():
                    onboard[k] += v
            else:
                sch_idx = int(ev[1])   # type: ignore
                pts.append(schools[sch_idx].coord)
                onboard[sch_idx] = 0

        # 補足最後回到學校的線段 (比照 simulate_route 的 auto_fill 邏輯)
        remaining_schools = [k for k, v in onboard.items() if v > 0]
        # 以最後一個事件點為起點
        cur_pos = pts[-1] if pts else None
        while remaining_schools and cur_pos:
            # 尋找離目前位置最近的目標學校
            nxt_sch_idx = min(remaining_schools, key=lambda k: travel_minutes(cur_pos, schools[k].coord))
            nxt_coord = schools[nxt_sch_idx].coord
            pts.append(nxt_coord)
            cur_pos = nxt_coord
            remaining_schools.remove(nxt_sch_idx)

        if len(pts)>=2:
            folium.PolyLine(pts, color=color, weight=4, tooltip=f"{title} Bus{b}").add_to(m)
        for st,sch,c,_ in r.pickup_detail:
            folium.CircleMarker(s_map[st].coord, radius=7, color=color, fill=True, fill_color=color,
                                tooltip=f"{title} Bus{b} - {s_map[st].name} -> School_{sch+1} +{c}").add_to(m)
    return m
