from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union



Coord = Tuple[float, float]



# ========= 資料結構 =========
@dataclass
class School:
    idx: int
    name: str
    coord: Coord

@dataclass
class Station:
    idx: int
    name: str
    coord: Coord
    demands: Dict[int, int]  # {school_idx: count}
    sector: int = 0

# 事件：('pickup', (station_idx, {school_idx: take})) 或 ('drop', school_idx)
Event = Tuple[str, Union[int, Tuple[int, Dict[int,int]]]]

@dataclass
class Route:
    events: List[Event] = field(default_factory=list)
    minutes: float = 0.0
    in_vehicle_minutes: float = 0.0
    fairness_penalty: float = 0.0
    # 列印用：每個批次（同站同校）乘車時間
    pickup_detail: List[Tuple[int,int,int,float]] = field(default_factory=list)
    # 內容: (station_idx, school_idx, taken, ride_minutes_to_that_school)

@dataclass
class Solution:
    routes: List[Route]
    total_minutes: float
    total_in_vehicle_minutes: float
    fairness_penalty: float
    students_served: int
    feasible: bool

# ---------- 資料結構 ----------
@dataclass
class Demand:
    station: int
    school: int
    students: int
    dist_km: float

@dataclass
class DropTask:
    station: int
    school: int
    students: int

@dataclass
class Bus:
    capacity: int
    used_capacity: int = 0
    pickup_tasks: List[DropTask] = field(default_factory=list)  # 站點裝載（可能含多校）
