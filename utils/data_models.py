from typing import Dict, List, Tuple, Union



INVEHICLE_TIME_WEIGHT = 1
TOTAL_TIME_WEIGHT = 30.0
BUS_COUNT_WEIGHT = 3000.0 # 顯著提高權重以優先減少派車數
FAIRNESS_WEIGHT = 100.0      # 先後順序謬誤費權重



# ========= 資料結構 =========
class Point(object):
    def __init__(self, idx:int, name:str, orig_idx:int):
        self.idx = idx
        self.name = name
        self.orig_idx = orig_idx  # 原始 ID，對應 stops.csv 中的 index 或 target；用於對照 time matrix 和原始需求資料

class School(Point):
    def __init__(self, idx:int, name:str, orig_idx:int):
        super().__init__(idx, name, orig_idx)

class Station(Point):
    def __init__(self, idx:int, name:str, demands:Dict[int, int], orig_idx:int):
        super().__init__(idx, name, orig_idx)
        self.demands = demands

class Route(object):
    def __init__(
            self, 
            events:List[Point] = [],
            minutes:float = 0.0,
            in_vehicle_minutes:float = 0.0,
            fairness_penalty:float = 0.0,
            pickup_detail:List[Tuple[int,int,int,float]] = []
        ):
        self.events = events
        self.minutes = minutes
        self.in_vehicle_minutes = in_vehicle_minutes
        self.fairness_penalty = fairness_penalty
        self.pickup_detail = pickup_detail
     
    def route_cost(self) -> float:
        return (
            INVEHICLE_TIME_WEIGHT*self.in_vehicle_minutes
            + FAIRNESS_WEIGHT*self.fairness_penalty
            + TOTAL_TIME_WEIGHT*self.minutes
            + BUS_COUNT_WEIGHT
        )


class Solution(object):
    def __init__(
            self,
            routes:List[Route] = [],
            total_minutes:float = 0.0,
            total_in_vehicle_minutes:float = 0.0,
            fairness_penalty:float = 0.0,
            students_served:int = 0,
            feasible:bool = False
        ):

        self.routes = routes
        self.total_minutes = total_minutes
        self.total_in_vehicle_minutes = total_in_vehicle_minutes
        self.fairness_penalty = fairness_penalty
        self.students_served = students_served
        self.feasible = feasible

    def solution_cost(self) -> float:
        if not self.feasible: return float('inf')
        return (
            INVEHICLE_TIME_WEIGHT*self.total_in_vehicle_minutes
            + FAIRNESS_WEIGHT*self.fairness_penalty
            + TOTAL_TIME_WEIGHT*self.total_minutes
            + BUS_COUNT_WEIGHT*len(self.routes)
        )