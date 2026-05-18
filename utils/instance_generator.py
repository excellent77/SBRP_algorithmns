import random, math
from typing import List, Tuple
from utils.data_models import School, Station
from utils.geo_utils import haversine_km, jitter_coord



NUM_SCHOOLS = 2            # 學校
NUM_STATIONS = 30          # 站點數
TOTAL_STUDENTS = 100       # 需求總人數
NUM_SECTORS = 6            # 用於站點散佈（不強制限制在單一扇形）
SCHOOL_SPREAD_KM = 3.0         # 學校群聚尺度（>1 校時才明顯）
STATION_MAX_RADIUS_KM = 6.0    # 站點外圍半徑


# ========= 產生多校多需求實例（即使目前 NUM_SCHOOLS=1 也走同樣流程） =========
def gen_instance_multi() -> Tuple[List[School], List[Station]]:
    base_center = (23.0000, 120.2000)
    schools: List[School] = []
    for k in range(NUM_SCHOOLS):
        schools.append(School(k, f"School_{k+1}",
                              jitter_coord(base_center, SCHOOL_SPREAD_KM*(0.5+0.8*random.random()))))

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
        coord = jitter_coord(centroid, STATION_MAX_RADIUS_KM)
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
        nearest = min(range(NUM_SCHOOLS), key=lambda k: haversine_km(coord, schools[k].coord))
        ang = math.degrees(math.atan2(coord[0]-schools[nearest].coord[0],
                                      coord[1]-schools[nearest].coord[1]))
        if ang<0: ang += 360
        st.sector = int(ang // (360.0/NUM_SECTORS))
        stations.append(st)

    return schools, stations
