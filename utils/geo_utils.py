import math, random
from typing import Tuple

# ========= 參數區 =========
SPEED_KMH = 30.0
TRAFFIC_FACTOR = 1.0
Coord = Tuple[float, float]



# ========= 幾何/時間 =========

def travel_minutes(a: Coord, b: Coord) -> float:
    # 嚴格使用 time matrix（必須提供 time CSV）；不再回退到座標估算
    from utils.time_registry import get_time_by_coord, has_time
    if has_time():
        t = get_time_by_coord(a, b)
        if t is not None:
            return float(t)
    raise RuntimeError("Time matrix 未註冊或找不到對應座標，請提供 time-b_7.csv 並透過 instance loader 載入")

def jitter_coord(c: Coord, d_km: float) -> Coord:
    r = d_km * math.sqrt(random.random())
    ang = 2*math.pi*random.random()
    dlat = r/111.0
    dlon = r/(111.0*max(math.cos(math.radians(c[0])), 1e-6))
    return (c[0] + dlat*math.sin(ang), c[1] + dlon*math.cos(ang))
