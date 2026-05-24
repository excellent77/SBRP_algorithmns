import random
from tqdm import tqdm
from typing import List, Tuple

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    build_route, build_solution,
    MAX_ROUTE_MIN, MAX_TOTAL_BUSES, BUS_CAPACITY,
    print_solution_pretty
    
)
from utils.instance_processer import load_instance_from_csv



class GreedySOlver(object):
    def __init__(self, schools:List[Station], stations:List[Station], time_matrix:List[List[float]]):
        self.schools = schools
        self.stations = stations
        self.time_matrix = time_matrix
        self.groups = []
        self.group_map = {}
        for st in stations:
            for sch_idx, count in st.demands.items():
                gid = len(self.groups)
                self.groups.append({
                    'id': gid,
                    'st_idx': st.idx,
                    'sch_idx': sch_idx,
                    'count': count
                })
                self.group_map[(st.idx, sch_idx)] = gid
        self.st_dict = {s.idx: s for s in stations}

    def build_route_from_group(self, gids: List[int]) -> Route:
        """根據群組 ID 序列構建 Route 物件，包含基本的 Drop 邏輯"""
        if not gids: return None
        events = [Station(
            self.groups[gid]['st_idx'],
            self.st_dict[self.groups[gid]['st_idx']].name,
            {self.groups[gid]['sch_idx']: self.groups[gid]['count']},
            self.st_dict[self.groups[gid]['st_idx']].orig_idx
            ) for gid in gids]
        target_schs = list(set(self.groups[gid]['sch_idx'] for gid in gids))
        last_st_idx = self.groups[gids[-1]]['st_idx']
        curr_pos = self.st_dict[last_st_idx].orig_idx
        while target_schs:
            nxt_sch = min(target_schs, key=lambda s: self.time_matrix[curr_pos][self.schools[s].orig_idx])
            events.append(School(
                nxt_sch,
                self.schools[nxt_sch].name,
                self.schools[nxt_sch].orig_idx
            ))
            curr_pos = self.schools[nxt_sch].orig_idx
            target_schs.remove(nxt_sch)
        return build_route(events, self.schools, self.st_dict, self.time_matrix)

    def run(self) -> Solution:
        """使用隨機近鄰貪婪方式生成一個可行初始解"""
        unserved = [g['id'] for g in self.groups]
        routes = []
        
        while unserved and len(routes) < MAX_TOTAL_BUSES:
            curr_route_gids = []
            start_gid = random.choice(unserved)
            curr_route_gids.append(start_gid)
            unserved.remove(start_gid)
            
            while unserved:
                last_gid = curr_route_gids[-1]
                last_st_idx = self.groups[last_gid]['st_idx']
                
                # 隨機選擇最近的 3 個候選站點之一，增加初始解多樣性
                candidates = sorted(unserved, key=lambda gid: self.time_matrix[self.st_dict[last_st_idx].orig_idx][self.st_dict[self.groups[gid]['st_idx']].orig_idx])
                k = min(3, len(candidates))
                next_gid = random.choice(candidates[:k])
                
                # 測試加入該群組後是否仍符合約束 (載重與最大行駛時間)
                test_gids = curr_route_gids + [next_gid]
                test_route = self.build_route_from_group(test_gids)
                load = sum(self.groups[x]['count'] for x in test_gids)
                
                if test_route and test_route.minutes <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    curr_route_gids.append(next_gid)
                    unserved.remove(next_gid)
                else:
                    break
            
            routes.append(self.build_route_from_group(curr_route_gids))
            
        return build_solution(routes, self.stations)

def run_greedy(schools: List[School], stations: List[Station], time_matrix:List[List[float]]) -> Solution:
    greedy_solver = GreedySOlver(schools, stations, time_matrix)
    solution = greedy_solver.run()
    return solution

if __name__ == "__main__":
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-b_7.csv", time_csv="./data/time-b_7.csv")
    solution = run_greedy(schools, stations, time_matrix)
    print_solution_pretty(solution, stations, schools)