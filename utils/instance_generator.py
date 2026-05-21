import random
import pandas as pd
from typing import Dict, List, Tuple

from utils.data_models import School, Station



NUM_SCHOOLS = 2            # 學校
NUM_STATIONS = 30          # 站點數
TOTAL_STUDENTS = 100       # 需求總人數
NUM_SECTORS = 6            # 用於站點散佈（不強制限制在單一扇形）
SCHOOL_SPREAD_KM = 3.0         # 學校群聚尺度（>1 校時才明顯）
STATION_MAX_RADIUS_KM = 6.0    # 站點外圍半徑



# ========= 產生多校多需求實例（即使目前 NUM_SCHOOLS=1 也走同樣流程） =========
def gen_instance_multi() -> Tuple[List[School], List[Station]]:
    schools: List[School] = []
    for k in range(NUM_SCHOOLS):
        # 不再產生真實座標，使用原始 ID 作為佔位座標
        schools.append(School(k, f"School_{k+1}", (float(k), 0.0)))

    # 每站總需求，之後再按學校比例拆分
    weights = [random.random()+0.2 for _ in range(NUM_STATIONS)]
    wsum = sum(weights)
    totals = [max(1, int(round(TOTAL_STUDENTS * w / wsum))) for w in weights]
    diff = TOTAL_STUDENTS - sum(totals)
    i = 0
    while diff != 0:
        j = i % NUM_STATIONS
        if diff > 0: totals[j]+=1; diff-=1
        elif totals[j] > 1: totals[j]-=1; diff+=1
        i += 1

    centroid = (sum(s.coord[0] for s in schools)/len(schools),
                sum(s.coord[1] for s in schools)/len(schools))

    stations: List[Station] = []
    for i in range(NUM_STATIONS):
        coord = (float(i+1), 0.0)
        # 將該站總需求拆給各校（1 校時就全部給 0）
        if NUM_SCHOOLS == 1:
            demand_map = {0: totals[i]}
        else:
            parts = [random.random()+0.1 for _ in range(NUM_SCHOOLS)]
            psum = sum(parts)
            raw = [int(round(totals[i]*p/psum)) for p in parts]
            dd = totals[i] - sum(raw); j=0
            while dd!=0:
                k = j % NUM_SCHOOLS
                if dd>0: raw[k]+=1; dd-=1
                elif raw[k]>0: raw[k]-=1; dd+=1
                j+=1
            demand_map = {k: raw[k] for k in range(NUM_SCHOOLS) if raw[k]>0}
            if not demand_map: demand_map = {random.randrange(NUM_SCHOOLS): totals[i]}

        st = Station(i+1, f"S{i+1:02d}", coord, demand_map)
        # 以最近學校決定 sector（僅供觀察）
        # 由於不再使用地理座標，改以 station id 決定 sector（僅供觀察）
        st.sector = i % NUM_SECTORS
        stations.append(st)

    return schools, stations



def load_instance_from_csv(stops_csv: str, time_csv: str,
                           school_orig_ids: Tuple[int, ...] = (112, 113),
                           ignore_orig_ids: Tuple[int, ...] = (114,)) -> Tuple[List[School], List[Station]]:

    df_stops = pd.read_csv(stops_csv)
    df_time = pd.read_csv(time_csv, index_col=0)
    df_time.index = df_time.index.astype(int)
    df_time.columns = df_time.columns.astype(int)

    target_ids = sorted(set(df_stops['target'].astype(int).tolist()))
    if all(sid in target_ids for sid in school_orig_ids):
        school_ids = [sid for sid in school_orig_ids if sid in target_ids]
    else:
        school_ids = sorted([sid for sid in target_ids if sid not in ignore_orig_ids])

    station_orig_ids = sorted(set(df_stops['index'].astype(int)) - set(ignore_orig_ids) - set(school_ids))
    if not station_orig_ids:
        raise ValueError("No station IDs found in stops CSV after filtering ignore IDs and schools.")

    school_id_map = {orig: idx for idx, orig in enumerate(school_ids)}
    station_id_map = {orig: idx for idx, orig in enumerate(station_orig_ids)}

    demands_by_station: Dict[int, Dict[int, int]] = {orig: {} for orig in station_orig_ids}
    for _, row in df_stops.iterrows():
        orig_idx = int(row['index'])
        target = int(row['target'])
        number = int(row['number'])
        if orig_idx in station_id_map and number > 0 and target in school_id_map:
            school_idx = school_id_map[target]
            demands_by_station[orig_idx][school_idx] = demands_by_station[orig_idx].get(school_idx, 0) + number

    node_ids = station_orig_ids + school_ids
    # 不再產生座標嵌入（MDS）；改以原始 ID 建立唯一佔位座標，並以 time matrix 為距離來源
    coords = {orig: (float(orig), 0.0) for orig in node_ids}

    # register time matrix and coord mapping so other modules can lookup by coords/original ids
    from utils.time_registry import set_time_df, set_coord_map
    set_time_df(df_time)
    set_coord_map(coords)

    schools: List[School] = []
    for orig, school_idx in school_id_map.items():
        coord = coords[orig]
        s = School(school_idx, f"School_{orig}", coord, orig_idx=orig)
        schools.append(s)

    stations: List[Station] = []
    for orig in station_orig_ids:
        coord = coords[orig]
        st = Station(station_id_map[orig], f"S{orig}", coord, demands_by_station[orig], orig_idx=orig)
        stations.append(st)

    return schools, stations

