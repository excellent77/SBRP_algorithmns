import math, random
from typing import Tuple

# ========= 參數區 =========
SPEED_KMH = 30.0
TRAFFIC_FACTOR = 1.0
Coord = Tuple[float, float]



# ========= 幾何/時間 =========
def haversine_km(a: Coord, b: Coord) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def travel_minutes(a: Coord, b: Coord) -> float:
    return haversine_km(a, b) / SPEED_KMH * 60.0 * TRAFFIC_FACTOR

def jitter_coord(c: Coord, d_km: float) -> Coord:
    r = d_km * math.sqrt(random.random())
    ang = 2*math.pi*random.random()
    dlat = r/111.0
    dlon = r/(111.0*max(math.cos(math.radians(c[0])), 1e-6))
    return (c[0] + dlat*math.sin(ang), c[1] + dlon*math.cos(ang))
