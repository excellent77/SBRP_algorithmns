import random
import copy
from tqdm import tqdm
from typing import List, Tuple
from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import(
    MAX_ROUTE_MIN, BUS_CAPACITY, MAX_TOTAL_BUSES,
    build_route, build_solution, print_solution_pretty
)
from utils.instance_processer import load_instance_from_csv


MAX_ATTEMPTS = 5000
POP_SIZE = 4000
GENERATIONS = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.6
ELITE_COUNT = 100

class GASolver(object):
    def __init__(self, schools: List[School], stations: List[Station], time_matrix: List[List[float]]):
        """初始化遺傳演算法求解器。

        Args:
            schools: 學校物件列表。
            stations: 站點物件列表。
            time_matrix: 所有原始索引之間的行駛時間二維列表。
        """
        self.schools = schools
        self.stations = stations
        self.time_matrix = time_matrix
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
        self.num_delimiters = MAX_TOTAL_BUSES - 1
        # evaluation cache: key=tuple(chromosome) -> (feasible, Solution, cost)
        self._eval_cache = {}

    def _create_individual(self, max_nodes: int = 200000) -> List[int]:
        """使用類 DFS 方法生成代表個體解的染色體。

        染色體是群組索引和分隔符號 (-1) 的排列，分隔符號用於區分路徑。空段意味著未使用的巴士。
        DFS 嘗試貪婪地構建有效路徑，以確保有更高機會獲得初始可行解。

        Args:
            max_nodes: 要訪問的 DFS 節點最大數量，以防止過度計算。

        Returns:
            代表染色體的整數列表。
        """
        if self.num_groups == 0:
            return [-1] * self.num_delimiters

        used = [False] * self.num_groups
        solution = None
        visited_nodes = 0

        def dfs(chromosome: List[int], current_segment: List[int], delimiters_left: int, used_count: int) -> bool:
            nonlocal solution, visited_nodes
            if solution is not None:
                return True
            if visited_nodes > max_nodes:
                return False

            if used_count == self.num_groups:
                solution = chromosome + [-1] * delimiters_left
                return True

            for g_idx in range(self.num_groups):
                if used[g_idx]:
                    continue
                next_segment = current_segment + [g_idx]
                if self._build_route_from_groups(next_segment) is None:
                    continue

                used[g_idx] = True
                if dfs(chromosome + [g_idx], next_segment, delimiters_left, used_count + 1):
                    return True
                used[g_idx] = False
            if delimiters_left > 0 and current_segment:
                if dfs(chromosome + [-1], [], delimiters_left - 1, used_count):
                    return True
            return False

        # 先從最小群組數、最多路徑限制開始 DFS
        group_order = list(range(self.num_groups))
        random.shuffle(group_order)
        for first in group_order:
            used[first] = True
            if dfs([first], [first], self.num_delimiters, 1):
                return solution
            used[first] = False
        # 如果 DFS 無法在節點限制內找到，可降級為隨機排列避免程式停滯
        chromosome = list(range(self.num_groups)) + [-1] * self.num_delimiters
        random.shuffle(chromosome)
        return chromosome

    def _build_route_from_groups(self, group_indices: List[int]) -> Route:
        """根據群組索引序列構建 Route 物件，並結合貪婪丟客邏輯。

        此函數合併同一站點的取貨，然後以貪婪方式（最近學校優先）
        加入學校的丟客事件。

        Args:
            group_indices: 代表取貨序列的群組索引列表。

        Returns:
            如果可行則回傳 Route 物件，否則回傳 None（如果超過容量或時間限制）。
        """
        if not group_indices: return None
        
        events = []
        target_schools = set()
        total_load = 0
        
        # Pickup: 依染色體順序，合併相同站點連續 Pickup 以符合資料結構
        curr_st_idx = -1
        curr_demands = {} # school_idx -> count for current station pickup
        for g_idx in group_indices:
            g = self.groups[g_idx]
            st_idx, sch_idx, count = g['station_idx'], g['school_idx'], g['count']
            target_schools.add(sch_idx)
            total_load += count
            
            if st_idx == curr_st_idx:
                curr_demands[sch_idx] = curr_demands.get(sch_idx, 0) + count
            else:
                if curr_st_idx != -1:
                    events.append(Station(curr_st_idx, self.st_dict[curr_st_idx].name, curr_demands, self.st_dict[curr_st_idx].orig_idx))
                curr_st_idx = st_idx
                curr_demands = {sch_idx: count}
        if curr_st_idx != -1:
            events.append(Station(curr_st_idx, self.st_dict[curr_st_idx].name, curr_demands, self.st_dict[curr_st_idx].orig_idx))

        # Greedy Drop: 採用最近學校加入法
        last_st_idx = self.groups[group_indices[-1]]['station_idx']
        last_pos = self.st_dict[last_st_idx].orig_idx
        rem_schools = list(target_schools)
        while rem_schools:
            nxt = min(rem_schools, key=lambda k: self.time_matrix[last_pos][self.schools[k].orig_idx])
            events.append(self.schools[nxt])
            last_pos = self.schools[nxt].orig_idx
            rem_schools.remove(nxt)
        r = build_route(events, self.schools, self.st_dict, self.time_matrix)
        
        # 確保合理性: 不大於最大公車容量，不大於行駛時間
        if total_load > BUS_CAPACITY or r.minutes > MAX_ROUTE_MIN:
            return None
        return r

    def decode(self, chromosome: List[int]) -> Solution:
        """將染色體解碼為 Solution 物件。

        染色體被分隔符號 (-1) 分成片段，每個片段都嘗試構建為一條路徑。

        Args:
            chromosome: 代表染色體的整數列表。

        Returns:
            Solution 物件。如果染色體導致不可行的解（例如：有未服務群組或無效路徑），
            則回傳一個空/不可行的解。
        """
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

    def _evaluate(self, chromosome: List[int]):
        """評估染色體，回傳其可行性、解碼後的 Solution 及其成本。

        此方法使用快取來存儲先前看過的染色體的評估結果，
        以避免冗餘計算。

        Args:
            chromosome: 代表染色體的整數列表。

        Returns:
            元組 (feasible: bool, solution: Solution, cost: float)。
        """
        key = tuple(chromosome)
        if key in self._eval_cache:
            return self._eval_cache[key]
        sol = self.decode(list(chromosome))
        cost = sol.solution_cost() if sol.feasible else float('inf')
        self._eval_cache[key] = (sol.feasible, sol, cost)
        return self._eval_cache[key]

    def crossover_segment(self, p1: List[int], p2: List[int]) -> List[int]:
        """透過結合兩個父代染色體的路徑片段來執行交配操作。

        它從每個父代中隨機選擇一個片段，將其結合，然後將剩餘未分配的群組
        貪婪地打包到新片段或現有片段中，確保路徑的可行性
        （容量和時間限制）。

        Args:
            p1: 第一個父代染色體。
            p2: 第二個父代染色體。

        Returns:
            交配產生的新染色體。
        """
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
        """對染色體應用突變操作。

        突變操作包括：
        1. 透過將分隔符號移動到末尾來合併路徑。
        2. 交換兩個相鄰元素（群組或分隔符號）。
        3. 將群組移動到染色體內的新隨機位置。

        Args:
            chromosome: 要被突變的染色體（原處修改）。
        """
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
        """執行遺傳演算法的主迴圈。

        初始化種群，然後迭代地應用選擇、交配和突變來使種群跨代進化。
        它會追蹤並回傳找到的最佳可行解。

        Returns:
            GA 找到的最佳可行 Solution 物件。

        Raises:
            RuntimeError: 如果無法生成可行個體，或者在進化過程中
                          種群完全變得不可行。
        """
        # 初始化：確保起始種群皆為可行解
        population = []
        for attempts in tqdm(range(MAX_ATTEMPTS), desc="Generating initial population"):
            if len(population) >= POP_SIZE:
                break
            ind = self._create_individual()
            if self.decode(ind).feasible: # type: ignore
                population.append(ind) # type: ignore
        if not population:
            raise RuntimeError(
                f"[GA] 無法在 {MAX_ATTEMPTS} 次嘗試內生成任何可行個體，請檢查約束條件或調整初始化方法。"            )

        best_sol = None
        print(f"[GA] 啟動進化演算法, 世代數: {GENERATIONS}, 群組總數: {self.num_groups}")
        for gen in tqdm(range(1, GENERATIONS + 1)):
            scored_pop = []  # list of (chromosome, Solution, cost)
            for ind in population:
                feasible, sol, cost = self._evaluate(ind)
                if feasible:
                    scored_pop.append((ind, sol, cost))

            if not scored_pop: # type: ignore
                raise RuntimeError("[GA] 當前種群內沒有任何可行個體，演算法無法繼續。")

            scored_pop.sort(key=lambda x: x[2])
            curr_best_sol = scored_pop[0][1]
            if best_sol is None or curr_best_sol.solution_cost() < best_sol.solution_cost():
                best_sol = copy.deepcopy(curr_best_sol)
                tqdm.write(f"[Gen {gen:03d}] 發現更優解 -> 成本: {best_sol.solution_cost():.1f} 車輛數:{len(best_sol.routes)}")

            new_pop = [item[0] for item in scored_pop[:ELITE_COUNT]] # type: ignore
            while len(new_pop) < POP_SIZE:
                p1, p2 = self._tournament(scored_pop), self._tournament(scored_pop)
                c1 = self.crossover_segment(p1, p2) if random.random() < CROSSOVER_RATE else list(p1)
                c2 = self.crossover_segment(p2, p1) if random.random() < CROSSOVER_RATE else list(p2)
                self.mutate(c1); self.mutate(c2)
                # evaluate children (cache) to avoid redundant decode later
                self._evaluate(c1)
                self._evaluate(c2)
                new_pop.extend([c1, c2])
            population = new_pop[:POP_SIZE]
        return best_sol

    def _tournament(self, scored_pop: List[Tuple[List[int], Solution, float]], k: int = 3) -> List[int]:
        """執行競爭式選擇 (Tournament Selection) 來挑選父代染色體。

        從評分後的種群中隨機選擇 `k` 個個體，並回傳其中成本最低（最佳）者的染色體。

        Args:
            scored_pop: 代表已評估種群的元組 (chromosome, Solution, cost) 列表。
            k: 參加競賽的個體數量。

        Returns:
            競賽獲勝者的染色體。
        """
        sample_size = min(k, len(scored_pop))
        selected = random.sample(scored_pop, sample_size)
        selected.sort(key=lambda x: x[2])
        return selected[0][0]

def run_ga(schools: List[School], stations: List[Station], time_matrix: List[List[float]]) -> Solution:
    """運行校車路徑問題的遺傳演算法求解器。

    Args:
        schools: 學校物件列表。
        stations: 站點物件列表。
        time_matrix: 所有原始索引之間的行駛時間二維列表。

    Returns:
        代表 GA 找到的最佳解之 Solution 物件。
    """
    solver = GASolver(schools, stations, time_matrix)
    return solver.run()

if __name__ == "__main__":
    """
    運行遺傳演算法求解器的入口點。
    從 CSV 檔案載入實例數據，運行 GA 並列印解。
    """
    random.seed(42)
    schools, stations, time_matrix = load_instance_from_csv(stops_csv="./data/stops-uniform_25+10.csv", time_csv="./data/time-uniform_25+10.csv")
    print(f"[DATA] 學校數={len(schools)}, 站點數={len(stations)}")
    best_ga_solution = run_ga(schools, stations, time_matrix)
    print_solution_pretty(best_ga_solution, stations, schools)