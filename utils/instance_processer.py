import random
import pandas as pd
from typing import Dict, List, Tuple

from utils.data_models import School, Station
from utils.solution_utils import load_time_matrix



NUM_SCHOOLS = 2            # 學校
NUM_STATIONS = 30          # 站點數
TOTAL_STUDENTS = 100       # 需求總人數



def load_instance_from_csv(
        stops_csv: str,
        time_csv:str,
        school_orig_ids: Tuple[int, ...] = (112, 113),
        ignore_orig_ids: Tuple[int, ...] = (114,)
    ) -> Tuple[List[School], List[Station]]:

    df_stops = pd.read_csv(stops_csv)
    time_matrix, time_idx_map = load_time_matrix(time_csv)

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

    schools: List[School] = []
    for orig, school_idx in school_id_map.items():
        schools.append(School(school_idx, f"School_{orig}", orig_idx=time_idx_map[orig]))

    stations: List[Station] = []
    for orig in station_orig_ids:
        stations.append(Station(station_id_map[orig], f"S{orig}", demands_by_station[orig], orig_idx=time_idx_map[orig]))

    return schools, stations, time_matrix

if __name__ == "__main__":
    print(load_instance_from_csv(
        stops_csv="~/SBRP_algorithmns/data/stops-b_7.csv",
        school_orig_ids=(112, 113),
        ignore_orig_ids=(114,)
     ))  

