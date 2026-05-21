import random
import copy
from tqdm import tqdm
from typing import List
from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import simulate_route, solution_cost, build_solution, print_solution_pretty
from utils.geo_utils import travel_minutes
from utils.instance_generator import gen_instance_multi, load_instance_from_csv

# ---------- GA 參數區 ----------
POP_SIZE = 1000           # 種群大小
GENERATIONS = 300       # 迭代代數
CROSSOVER_RATE = 0.8    # 交配機率
MUTATION_RATE = 0.6     # 突變機率
ELITE_COUNT = 100         # 菁英保留數

BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0
MAX_TOTAL_BUSES = 6

class GASolver:
    def __init__(self, schools: List[School], stations: List[Station]):
        self.schools = schools
        self.stations = stations
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        for st in stations:
            for sch_idx, count in st.demands.items():
                self.groups.append({
                    'station_idx': st.idx,
                    'school_idx': sch_idx,
                    'count': count
                })
        self.num_groups = len(self.groups)
        self.m = MAX_TOTAL_BUSES
        self.num_delimiters = self.m - 1
        self.total_demand = sum(g['count'] for g in self.groups)

    def _create_individual(self) -> List[int]:
        """採用隨機貪婪的方式產生初始解：地理最近鄰排列 + 貪婪裝箱切分"""
        remaining = list(range(self.num_groups))
        if not remaining: return [-1] * self.num_delimiters
        
        ordered_groups = []
        # 隨機挑選起點
        curr_g_idx = random.choice(remaining)
        remaining.remove(curr_g_idx)
        ordered_groups.append(curr_g_idx)
        
        # 1. 隨機貪婪序列生成 (Randomized Nearest Neighbor)
        while remaining:
            curr_st_idx = self.groups[curr_g_idx]['station_idx']
            curr_coord = self.st_dict[curr_st_idx].coord
            
            # 計算剩餘站點的距離並排序
            candidates = [(rg, travel_minutes(curr_coord, self.st_dict[self.groups[rg]['station_idx']].coord)) for rg in remaining]
            candidates.sort(key=lambda x: x[1])
            
            # 從最近的 K 個中隨機挑選一個
            k = min(3, len(candidates))
            curr_g_idx = random.choice(candidates[:k])[0]
            remaining.remove(curr_g_idx)
            ordered_groups.append(curr_g_idx)

        # 2. 貪婪切分 (-1)
        ind = []
        curr_seg = []
        delims_left = self.num_delimiters
        for g_idx in ordered_groups:
            if self._build_route_from_groups(curr_seg + [g_idx]) is not None:
                curr_seg.append(g_idx)
            elif delims_left > 0:
                ind.extend(curr_seg + [-1]); delims_left -= 1
                curr_seg = [g_idx]
            else: curr_seg.append(g_idx)
        ind.extend(curr_seg + [-1] * delims_left)
        return ind

    def _build_route_from_groups(self, group_indices: List[int]) -> Route:
        """根據群組序列建立 Route 並加入 Greedy Drop 邏輯"""
        if not group_indices: return None
        
        events = []
        target_schools = set()
        total_load = 0
        
        # Pickup: 依染色體順序，合併相同站點連續 Pickup 以符合資料結構
        curr_st_idx = -1
        curr_demands = {}
        for g_idx in group_indices:
            g = self.groups[g_idx]
            st_idx, sch_idx, count = g['station_idx'], g['school_idx'], g['count']
            target_schools.add(sch_idx)
            total_load += count
            
            if st_idx == curr_st_idx:
                curr_demands[sch_idx] = curr_demands.get(sch_idx, 0) + count
            else:
                if curr_st_idx != -1:
                    events.append(('pickup', (curr_st_idx, curr_demands)))
                curr_st_idx = st_idx
                curr_demands = {sch_idx: count}
        if curr_st_idx != -1:
            events.append(('pickup', (curr_st_idx, curr_demands)))
            
        # Greedy Drop: 採用最近學校加入法
        last_st_idx = self.groups[group_indices[-1]]['station_idx']
        last_pos = self.st_dict[last_st_idx].coord
        rem_schools = list(target_schools)
        while rem_schools:
            nxt = min(rem_schools, key=lambda k: travel_minutes(last_pos, self.schools[k].coord))
            events.append(('drop', nxt))
            last_pos = self.schools[nxt].coord
            rem_schools.remove(nxt)
            
        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(r, self.schools, self.st_dict)
        
        # 確保合理性: 不大於最大公車容量，不大於行駛時間
        if total_load > BUS_CAPACITY or r.minutes > MAX_ROUTE_MIN:
            return None
        return r

    def decode(self, chromosome: List[int]) -> Solution:
        segments = []
        curr = []
        for val in chromosome:
            if val == -1:
                if curr: segments.append(curr); curr = []
            else:
                curr.append(val)
        if curr: segments.append(curr)
            
        routes = []
        served_groups = set()
        feasible = True
        for seg in segments:
            r = self._build_route_from_groups(seg)
            if r is None:
                feasible = False; break
            routes.append(r)
            served_groups.update(seg)
            
        if not feasible or len(served_groups) != self.num_groups:
            return build_solution([], self.stations)
        return build_solution(routes, self.stations)

    def crossover_segment(self, p1: List[int], p2: List[int]) -> List[int]:
        """組合父母的完整巴士路徑片段，並嘗試貪婪合併其餘群組，且確保符合限制式"""
        def get_segs(chrom):
            segs, curr = [], []
            for v in chrom:
                if v == -1:
                    if curr: segs.append(list(curr))
                    curr = []
                else: curr.append(v)
            if curr: segs.append(list(curr))
            return segs

        segs1, segs2 = get_segs(p1), get_segs(p2)
        s1 = random.choice(segs1) if segs1 else []
        s2 = random.choice(segs2) if segs2 else []
        
        # 組合選中的片段作為新染色體的開頭
        combined_groups, seen = [], set()
        for g in s1 + s2:
            if g not in seen: combined_groups.append(g); seen.add(g)
            
        remaining_groups = [g for g in range(self.num_groups) if g not in seen]
        random.shuffle(remaining_groups) # 保持一定的隨機多樣性
        
        new_chrom_segments = []
        current_segment_groups = list(combined_groups) # Start with the combined segment

        # Process remaining groups with greedy packing
        for g_idx in remaining_groups:
            # Try to add the group to the current segment
            test_segment = current_segment_groups + [g_idx]

            # Check if the test_segment forms a valid route
            temp_route = self._build_route_from_groups(test_segment)

            if temp_route is not None: # The segment is still valid with the new group
                current_segment_groups = test_segment # Extend the current segment
            else: # The segment becomes invalid, need a new route
                if current_segment_groups: # If there were groups in the current segment, finalize it
                    new_chrom_segments.append(current_segment_groups)
                # Start a new segment with the current group, regardless of its individual validity
                current_segment_groups = [g_idx]

        # Add the last current_segment_groups if not empty
        if current_segment_groups:
            new_chrom_segments.append(current_segment_groups)

        # Now, reconstruct the chromosome with delimiters
        new_chrom = []
        delimiters_to_add = self.num_delimiters

        for i, segment in enumerate(new_chrom_segments):
            new_chrom.extend(segment)
            if i < len(new_chrom_segments) - 1 and delimiters_to_add > 0:
                new_chrom.append(-1)
                delimiters_to_add -= 1
        
        # Add any remaining delimiters to the end
        while delimiters_to_add > 0:
            new_chrom.append(-1)
            delimiters_to_add -= 1

        return new_chrom

    def mutate(self, chromosome: List[int]):
        if random.random() >= MUTATION_RATE: return
        
        orig = list(chromosome)
        
        # 1. 併車突變：隨機將一個分割點移動到末尾，嘗試將兩個路徑合併
        if random.random() < 0.4 and self.num_delimiters > 0: # 確保有分割點可以移動
            delim_indices = [i for i, v in enumerate(chromosome) if v == -1]
            if delim_indices:
                idx = random.choice(delim_indices)
                val = chromosome.pop(idx)
                chromosome.append(val)

        # 2. 隨機交換相鄰兩個位置 (群或分割標記)
        if len(chromosome) >= 2:
            idx = random.randint(0, len(chromosome) - 2)
            chromosome[idx], chromosome[idx + 1] = chromosome[idx + 1], chromosome[idx]
        
        # 3. 隨機抽取站點移動到新位置 (嘗試插到其他路徑中以填補剩餘空間)
        group_positions = [idx for idx, v in enumerate(chromosome) if v != -1]
        if group_positions:
            pos = random.choice(group_positions)
            val = chromosome.pop(pos)
            chromosome.insert(random.randint(0, len(chromosome)), val)

    def run(self) -> Solution:
        # 初始化：確保起始種群皆為可行解
        population = []
        while len(population) < POP_SIZE:
            ind = self._create_individual()
            if self.decode(ind).feasible:
                population.append(ind)

        best_sol = None 
        print(f"[GA] 啟動進化演算法, 世代數: {GENERATIONS}, 群組總數: {self.num_groups}")
        for gen in tqdm(range(1, GENERATIONS + 1)):
            scored_pop = []
            for ind in population:
                sol = self.decode(ind)
                if sol.feasible:
                    #sol = try_merge_routes(sol, self.stations, self.schools)
                    scored_pop.append((ind, sol))
            

            scored_pop.sort(key=lambda x: solution_cost(x[1]))
            curr_best_sol = scored_pop[0][1]
            if best_sol is None or solution_cost(curr_best_sol) < solution_cost(best_sol):
                best_sol = copy.deepcopy(curr_best_sol)
                tqdm.write(f"[Gen {gen:03d}] 發現更優解 -> 成本: {solution_cost(best_sol):.1f} 車輛數:{len(best_sol.routes)}")

            new_pop = [item[0] for item in scored_pop[:ELITE_COUNT]]
            while len(new_pop) < POP_SIZE:
                p1, p2 = self._tournament(scored_pop), self._tournament(scored_pop)
                c1 = self.crossover_segment(p1, p2) if random.random() < CROSSOVER_RATE else list(p1)
                c2 = self.crossover_segment(p2, p1) if random.random() < CROSSOVER_RATE else list(p2)
                self.mutate(c1); self.mutate(c2)
                new_pop.extend([c1, c2])
            population = new_pop[:POP_SIZE]
        return best_sol

    def _tournament(self, scored_pop, k=3):
        # 確保樣本數不超過當前可用種群數量
        sample_size = min(k, len(scored_pop))
        selected = random.sample(scored_pop, sample_size)
        selected.sort(key=lambda x: solution_cost(x[1]))
        return selected[0][0]

def run_ga(schools: List[School], stations: List[Station]) -> Solution:
    solver = GASolver(schools, stations)
    return solver.run()

if __name__ == "__main__":
    # 測試執行區塊
    random.seed(42)
    csv_instance = load_instance_from_csv(stops_csv="./data/stops-b_8.csv", time_csv="./data/time-b_8.csv")
    if csv_instance is not None:
        schools, stations = csv_instance
        print("[DATA] loaded instance from CSV")
    else:
        schools, stations = gen_instance_multi()
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")

    # 執行 GA 演算法
    best_ga_solution = run_ga(schools, stations)

    # 輸出結果
    print_solution_pretty(best_ga_solution, stations, schools)
    
