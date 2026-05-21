import os
import random
from typing import List
import gurobipy as gp
from gurobipy import GRB

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import(
    simulate_route, build_solution,
    print_solution_pretty, plot_routes_on_map, route_cost
)
from utils.instance_generator import gen_instance_multi, load_default_instance_from_csv
from lns_solver import run_lns

# ---------- Matheuristic 參數 ----------
ROUTE_POOL_SIZE = 80  # 候選路徑池大小
MAX_TOTAL_BUSES = 6 #公車數量上限
BUS_CAPACITY = 40
MAX_ROUTE_MIN = 60.0

class MatheuristicSolver:
    def __init__(self, schools: List[School], stations: List[Station]):
        self.schools = schools
        self.stations = stations
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        for st in stations:
            for sch_idx, count in st.demands.items():
                self.groups.append({
                    'id': len(self.groups),
                    'st_idx': st.idx,
                    'sch_idx': sch_idx,
                    'count': count
                })
        self.route_pool: List[Route] = []

    def generate_route_pool(self):
        """
        啟發式階段：生成大量高品質的可行路徑。
        這裡結合了 LNS 優化後的路徑與隨機排列切割。
        """
        print(f"[Matheuristic] 開始生成候選路徑池... 目標: {ROUTE_POOL_SIZE} 條")
        
        # 1. 利用 LNS 邏輯生成高品質路徑
        while len(self.route_pool) < ROUTE_POOL_SIZE:
            # 透過 LNS 迭代尋找一個高品質解
            temp_sol = run_lns(self.schools, self.stations, print_log=False)
            # 提取解中的路徑並存入池中
            for r in temp_sol.routes:
                # 確保路徑符合時間和容量限制
                total_load_on_route = sum(c for _, _, c, _ in r.pickup_detail)
                if r.minutes <= MAX_ROUTE_MIN and total_load_on_route <= BUS_CAPACITY:
                    self.route_pool.append(r)
            
            # 2. 隨機打亂站點並嘗試簡單切割，增加多樣性
            st_indices = [s.idx for s in self.stations]
            random.shuffle(st_indices)
            current_st = []
            current_load = 0
            for idx in st_indices:
                st = self.st_dict[idx]
                load = sum(st.demands.values())
                if current_load + load <= BUS_CAPACITY:
                    current_st.append(idx)
                    current_load += load
                else:
                    if current_st:
                        # 建立路徑（去最近學校）
                        events = [('pickup', (s_idx, self.st_dict[s_idx].demands)) for s_idx in current_st]
                        # 簡單加上 drop 邏輯
                        target_sch = set()
                        for s_idx in current_st:
                            for sch_idx in self.st_dict[s_idx].demands.keys(): target_sch.add(sch_idx)
                        for sch_idx in target_sch: events.append(('drop', sch_idx))
                        
                        r = Route(events=events)
                        mins, ivm, detail, fairness = simulate_route(r, self.schools, self.st_dict)
                        total_load_on_route = sum(c for _, _, c, _ in detail)
                        if mins <= MAX_ROUTE_MIN and total_load_on_route <= BUS_CAPACITY:
                            r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = mins, ivm, detail, fairness
                            self.route_pool.append(r)
                    current_st = [idx]
                    current_load = load

        # 去除重複路徑
        unique_pool = {}
        for r in self.route_pool:
            key = str(r.events)
            unique_pool[key] = r
        self.route_pool = list(unique_pool.values())
        print(f"[Matheuristic] 路徑池建置完成，共有 {len(self.route_pool)} 條不重複路徑")

    def _groups_in_route(self, r: Route) -> List[int]:
        """回傳此路徑包含的群體 ID 列表"""
        served = []
        # 透過 pickup_detail (st, sch, count, ride) 匹配 group
        for st_idx, sch_idx, count, _ in r.pickup_detail:
            for g in self.groups:
                # 這裡假設 pickup_detail 中的 count 是該 (st_idx, sch_idx) 群體的總需求，
                # 如果是部分需求，則需要更精確的匹配邏輯
                if g['st_idx'] == st_idx and g['sch_idx'] == sch_idx:
                    served.append(g['id'])
        return served
    def solve_exact_model(self) -> Solution:
        """
        精確解階段：使用 Gurobi 解決集合劃分問題。
        """
        print("[Matheuristic] 啟動 Gurobi 整數規劃模型...")
        m = gp.Model("SBRP_SetPartitioning")
        m.setParam('OutputFlag', 1) # 開啟日誌供稽核

        # 變數：x[r] = 1 代表選中第 r 條路徑
        x = m.addVars(len(self.route_pool), vtype=GRB.BINARY, name="route_select")

        # 目標函數：最小化總成本 (IVM + 派車權重 + 不公平費)
        obj = gp.quicksum(x[i] * route_cost(self.route_pool[i])
                          for i in range(len(self.route_pool)))
        m.setObjective(obj, GRB.MINIMIZE)

        # 約束 1：每個群體必須被覆蓋（恰好一次）
        # 建立 群體ID -> 路徑索引 的映射
        group_to_routes = {g['id']: [] for g in self.groups}
        for i, r in enumerate(self.route_pool):
            served_groups_in_r = self._groups_in_route(r)
            for g_id in served_groups_in_r:
                group_to_routes[g_id].append(i)

        for g_id, route_indices in group_to_routes.items():
            if not route_indices:
                # 如果某個群體沒有任何路徑可以覆蓋，則問題無解
                print(f"警告: 群體 {g_id} (站點 {self.groups[g_id]['st_idx']} -> 學校 {self.groups[g_id]['sch_idx']}) 沒有任何可行路徑覆蓋！")
                continue
            # 每個群體必須被覆蓋恰好一次 (Set Partitioning)
            m.addConstr(gp.quicksum(x[i] for i in route_indices) == 1, name=f"cover_group_{g_id}")

        # 約束 2：總車輛數限制
        m.addConstr(gp.quicksum(x[i] for i in range(len(self.route_pool))) <= MAX_TOTAL_BUSES, name="max_buses")

        m.optimize()

        if m.status == GRB.OPTIMAL:
            selected_routes = [self.route_pool[i] for i in range(len(self.route_pool)) if x[i].x > 0.5]
            return build_solution(selected_routes, self.stations)
        else:
            print("[Error] Gurobi 無法在當前路徑池中找到可行解")
            return build_solution([], self.stations)

def run_matheuristic(schools, stations) -> Solution:
    solver = MatheuristicSolver(schools, stations)
    solver.generate_route_pool()
    return solver.solve_exact_model()

if __name__ == "__main__":
    random.seed(42)
    csv_instance = load_default_instance_from_csv()
    if csv_instance is not None:
        schools, stations = csv_instance
        print("[DATA] loaded instance from CSV: stops-b_7.csv + time-b_7.csv")
    else:
        schools, stations = gen_instance_multi()
    
    best_sol = run_matheuristic(schools, stations)
    
    if best_sol.feasible:
        print_solution_pretty(best_sol, stations, schools)
        m = plot_routes_on_map(best_sol, stations, schools, title="Matheuristic (Set Partitioning)")
        os.makedirs("./data", exist_ok=True)
        m.save("./data/matheuristic_routes.html")
        print("已輸出地圖：./data/matheuristic_routes.html")
    else:
        print("求解失敗，請嘗試增加 ROUTE_POOL_SIZE 或檢查約束條件。")
