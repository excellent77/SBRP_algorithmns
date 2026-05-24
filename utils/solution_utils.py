import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.data_models import Point, School, Station, Route, Solution

# ========= 參數區 =========
BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0
MAX_TOTAL_BUSES = 6



def load_time_matrix(time_csv: str) -> Tuple[List[List[float]], Dict[int,int]]:
    df_time = pd.read_csv(time_csv, index_col=0)
    df_time.index = df_time.index.astype(int)
    df_time.columns = df_time.columns.astype(int)
    time_matrix = df_time.values.tolist()
    for i in range(len(time_matrix)):
        time_matrix[i][i] = 0.0
    idx_map = {idx: i for i, idx in enumerate(df_time.index)}
    return time_matrix, idx_map

# ========= Route, Solution Build =========
def route_max_load(route: Route) -> int:
    """計算路線任一時刻的最大車上人數。"""
    onboard_by_school: Dict[int, int] = defaultdict(int)
    max_load = 0
    for e in route.events:
        if isinstance(e, Station):
            for sch_idx, count in e.demands.items():
                onboard_by_school[int(sch_idx)] += count
        elif isinstance(e, School):
            sch_idx = int(e.idx)
            onboard_by_school[sch_idx] = 0
        else:
            raise RuntimeError("未知事件類型")
        max_load = max(max_load, sum(onboard_by_school.values()))
    return max_load

def build_route(
        events: List[Point], 
        schools: List[School], 
        stations: Dict[int, Station],
        time_matrix: List[List[float]],
        auto_fill: bool = False
    ) -> Tuple[float,float,List[Tuple[int,int,int,float]], float]:
    
    if not events:
        return Route()

    minutes = 0.0
    fairness_penalty = 0.0
    ivm = 0.0
    pickup_detail: List[Tuple[int,int,int,float]] = []
    onboard_by_school: Dict[int,int] = defaultdict(int) # key=school_idx, value=目前車上該校學生數
    batches: List[Dict] = []  # 每批上車紀錄：{'station','school','count','acc'} 
    finished_batches: List[Dict] = [] # 已送達紀錄

    cur = events[0].orig_idx
    for e in events:
        nxt = e.orig_idx
        travel = time_matrix[cur][nxt]
        minutes += travel
        total_onboard = sum(onboard_by_school.values())
        ivm += travel * total_onboard
        for b in batches: b['acc'] += travel

        if isinstance(e, Station):
            for k, c in e.demands.items():
                onboard_by_school[k] += c
                batches.append({'station': e.idx, 'school': k, 'count': c, 'acc': 0.0})
        else:  # drop
            sch_idx = int(e.idx)
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
                nxt_sch_idx = min(remaining_schools, key=lambda k: time_matrix[cur][schools[k].orig_idx])
                nxt_coord = schools[nxt_sch_idx].orig_idx
                travel = time_matrix[cur][nxt_coord]
                
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
        sch_orig_idx = schools[sch_idx].orig_idx
        # 比較該校所有學生的來源站點對 (i, j)
        for i in range(len(school_batches)):
            for j in range(len(school_batches)):
                if i == j: continue
                b_i = school_batches[i]
                b_j = school_batches[j]

                # 計算兩站點到該校的理論行駛距離（時間）
                dist_i = time_matrix[stations[b_i['station']].orig_idx][sch_orig_idx]
                dist_j = time_matrix[stations[b_j['station']].orig_idx][sch_orig_idx]

                if dist_i < dist_j and b_i['acc'] > b_j['acc']:
                    fairness_penalty += 1.0
    return Route(
        events=events,
        minutes=minutes,
        in_vehicle_minutes=ivm,
        fairness_penalty=fairness_penalty, 
        pickup_detail=pickup_detail
    )

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


# ========= 併車（串接事件序列；再次模擬核對 60 分鐘） =========
def try_merge_routes(sol: Solution, stations_list: List[Station], schools: List[School], auto_fill:bool=False) -> Solution:
    stations = {s.idx:s for s in stations_list}
    routes = sol.routes[:]
    changed = False
    while changed:
        n=len(routes)
        for i in range(n):
            for j in range(i+1, n):
                r1, r2 = routes[i], routes[j]
                events = r1.events + r2.events
                r_tmp = build_route(events, schools, stations, auto_fill=auto_fill)
                load = route_max_load(r_tmp)
                if r_tmp.minutes <= MAX_ROUTE_MIN + 1e-6 and load <= BUS_CAPACITY:
                    keep = [k for k in range(n) if k not in (i,j)]
                    routes = [routes[k] for k in keep] + [r_tmp]
                    changed=True
                    break
            if changed: break

    return build_solution(routes, stations_list)

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
    print(f"總分數: {sol.solution_cost():.1f}")
    print("-" * 60)

    # 動態欄寬,根據站名長度調整；至少 6 字元寬以保持對齊
    station_w = max(6, max(len(s.name) for s in stations) if stations else 6)

    for b, r in enumerate(sol.routes, 1):
        total_load = sum(c for _, _, c, _ in r.pickup_detail)
        print(f"Bus{b}: 共載 {total_load} 人，總行駛時間={r.minutes:.1f} 分鐘")
        print("-" * 60)

        # 1) 依事件順序輸出
        line_no = 1
        for ev in r.events:
            if not isinstance(ev, Station):  # type: ignore
                continue
            take_map = ev.demands
            right = "、".join([
                f"{sch_name.get(k, f'School_{k+1}')},載 {c:>2} 人"
                for k, c in take_map.items()
            ])
            print(f"{line_no:02d}. {s_map[ev.idx].name:<{station_w}} → {right}")
            line_no += 1

        # 2) 路徑摘要： 站點序列 -> 學校序列
        # 站點序列依事件順序，不去重；學校序列依實際 drop 事件順序
        left = " -> ".join([
            s_map[ev.idx].name
            for ev in r.events
            if isinstance(ev, Station)
        ])
        seen = set()
        school_seq = []
        for ev in r.events:
            if not isinstance(ev, School):
                continue
            sch_idx = int(ev.idx)
            if sch_idx not in seen:
                school_seq.append(sch_idx)
                seen.add(sch_idx)
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

    

if __name__=='__main__':
    # 測試讀取 time matrix
    time_matrix, time_map = load_time_matrix("~/SBRP_algorithmns/data/time-b_7.csv")
    print(time_matrix)