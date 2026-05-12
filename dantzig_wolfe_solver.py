import os
import random
from tqdm import tqdm
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    simulate_route, build_solution,
    BUS_COUNT_WEIGHT,
    TOTAL_TIME_WEIGHT,
    ROUTE_TIME_WEIGHT,
    FAIRNESS_WEIGHT,
    print_solution_pretty, plot_routes_on_map, route_cost,
    MAX_ROUTE_MIN, BUS_CAPACITY
)
from utils.instance_generator import gen_instance_multi
from utils.geo_utils import travel_minutes
from lns_solver import run_lns

# ---------- 參數設定 ----------
MAX_TOTAL_BUSES = 6
DW_ITERATIONS = 2000

# ===============================
# Dantzig-Wolfe Solver
# ===============================
class DantzigWolfeSolver:

    def __init__(self, schools: List[School], stations: List[Station]):
        self.schools = schools
        self.stations = stations
        self.st_dict = {s.idx: s for s in stations}
        
        # 定義群 (Group): 同一站點且同目的地學校的學生
        self.groups = []
        self.group_map = {} # For O(1) group lookup by (st_idx, sch_idx)
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

        self.routes = []
        self.route_pool_keys = set() # For O(1) route duplication check
        self.group_to_routes = [[] for _ in range(len(self.groups))] # For faster Master Problem modeling
        self.duals = [0.0] * len(self.groups)
        self.mu = 0.0
        self.alpha = 0.7  # dual stabilization
        self.best_feasible_sol = None

    # ===============================
    # Route Pool Management
    # ===============================
    def _add_route(self, r: Route) -> bool:
        """將路徑加入池中並進行去重檢查，返回是否成功加入"""
        g_ids = self._groups_in_route(r)
        g_set = frozenset(g_ids)
        if not g_set or g_set in self.route_pool_keys:
            return False
        
        route_idx = len(self.routes)
        r._g_ids = g_set # Cache for later modeling efficiency
        self.routes.append(r)
        self.route_pool_keys.add(g_set)
        
        # Update group to routes mapping index
        for g_id in g_ids:
            self.group_to_routes[g_id].append(route_idx)
        return True

    # ===============================
    # Initial columns
    # ===============================
    def initialize_columns(self):
        """使用啟發式解 (ACO) 初始化路徑池，確保 Master Problem 一開始就是可行的"""
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        # 取得一個高品質初始解，確保滿足車輛數限制
        init_sol = run_lns(self.schools, self.stations) #self._generate_random_greedy_solution()
        for r in init_sol.routes:
            self._add_route(r)
            
        # 檢查是否所有群體都被涵蓋
        # Use cached _g_ids for efficiency if available, otherwise compute
        covered_groups = set().union(*(r._g_ids for r in self.routes if hasattr(r, '_g_ids')))
        
        for g in self.groups:
            if g['id'] not in covered_groups:
                self._add_route(self._single_group_route(g))

    def _generate_random_greedy_solution(self) -> Solution:
        """使用隨機近鄰貪婪方式生成一個可行初始解"""
        unserved = [g['id'] for g in self.groups]
        routes = []
        
        while unserved and len(routes) < MAX_TOTAL_BUSES:
            curr_route_gids = []
            # 隨機挑選起點增加多樣性
            start_gid = random.choice(unserved)
            curr_route_gids.append(start_gid)
            unserved.remove(start_gid)
            
            while unserved:
                last_gid = curr_route_gids[-1]
                last_st_idx = self.groups[last_gid]['st_idx']
                last_coord = self.st_dict[last_st_idx].coord
                
                # 隨機選擇最近的 3 個候選站點之一，增加初始解多樣性
                candidates = sorted(unserved, key=lambda gid: travel_minutes(last_coord, self.st_dict[self.groups[gid]['st_idx']].coord))
                k = min(3, len(candidates))
                next_gid = random.choice(candidates[:k])
                
                # 測試加入該群組後是否仍符合約束 (載重與最大行駛時間)
                test_gids = curr_route_gids + [next_gid]
                test_route = self._build_route_from_group_ids(test_gids)
                load = sum(self.groups[x]['count'] for x in test_gids)
                
                if test_route and test_route.minutes <= MAX_ROUTE_MIN and load <= BUS_CAPACITY:
                    curr_route_gids.append(next_gid)
                    unserved.remove(next_gid)
                else:
                    break
            
            routes.append(self._build_route_from_group_ids(curr_route_gids))
            
        return build_solution(routes, self.stations)

    def _build_route_from_group_ids(self, gids: List[int]) -> Route:
        """根據群組 ID 序列構建 Route 物件，包含基本的 Drop 邏輯"""
        if not gids: return None
        path = [('g', gid) for gid in gids]
        target_schs = list(set(self.groups[gid]['sch_idx'] for gid in gids))
        last_st_idx = self.groups[gids[-1]]['st_idx']
        curr_pos = self.st_dict[last_st_idx].coord
        while target_schs:
            nxt_sch = min(target_schs, key=lambda s: travel_minutes(curr_pos, self.schools[s].coord))
            path.append(('s', nxt_sch))
            curr_pos = self.schools[nxt_sch].coord
            target_schs.remove(nxt_sch)
        return self._build_route(path)


    def _single_group_route(self, g):
        """為單一群體建立基礎可行路徑"""
        events = [
            ('pickup', (g['st_idx'], {g['sch_idx']: g['count']})),
            ('drop', g['sch_idx'])
        ]
        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(
            r, self.schools, self.st_dict
        )
        return r

    def _get_coord(self, n_type, n_idx):
        """根據類型獲取座標"""
        if n_type == 'st':
            return self.st_dict[n_idx].coord
        else:
            return self.schools[n_idx].coord

    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")

        m.setObjective(
            gp.quicksum(x[i] * route_cost(self.routes[i]) for i in range(len(self.routes))),
            GRB.MINIMIZE
        )

        constrs = {}
        for g_idx in range(len(self.groups)):
            idxs = self.group_to_routes[g_idx] # 使用預建索引提升效能
            # 每個群體必須被覆蓋剛好一次
            constrs[g_idx] = m.addConstr(gp.quicksum(x[i] for i in idxs) == 1, name=f"Cover_G{g_idx}")

        bus_constr = m.addConstr(gp.quicksum(x[i] for i in range(len(self.routes))) <= MAX_TOTAL_BUSES, name="MaxBus")

        m.optimize()

        if relax and m.status == GRB.OPTIMAL:
            self.duals = [constrs[g].Pi for g in range(len(self.groups))]
            self.mu = bus_constr.Pi

            # stabilization
            #for g_idx in range(len(self.groups)):
            #    self.duals[g_idx] = self.alpha * self.duals[g_idx] + (1 - self.alpha) * new_duals[g_idx]
            #self.mu = self.alpha * self.mu + (1 - self.alpha) * new_mu

        return m

    def _groups_in_route(self, r: Route) -> List[int]:
        """回傳此路徑包含的群體 ID 列表"""
        # 使用 group_map 進行 O(1) 匹配
        return [self.group_map[(d[0], d[1])] for d in r.pickup_detail]

    # ===============================
    # Pricing 
    # ===============================
    def pricing(self):
        best_route = None
        best_rc = 0.0 # 尋找負的縮減成本

        # 預先計算現有路徑的群體組成，用於快速去重判斷
        #existing_route_groups = [set(self._groups_in_route(r)) for r in self.routes]
        # 預先計算所有正的對偶值總和，用於動態剪枝
        total_pos_dual = sum(d for d in self.duals if d > 0)
        # 預先按對偶值從高到低排序群體，使 DFS 優先枚舉潛力較高的路徑
        sorted_groups = sorted(
            self.groups,
            key=lambda g: self.duals[g['id']],# / travel_minutes(self.st_dict[g['st_idx']].coord, self.schools[g['sch_idx']].coord),
            reverse=True
        )

        # DFS 共享 Buffer
        path = []               # 紀錄 (type, idx)
        onboard_groups = []     # 目前在車上的 group id
        visited_groups = [False] * len(self.groups)
        group_pickup_time = {}  # gid -> time_at_pickup
        finished_groups_info = [] # (sch_idx, st_idx, ivm) 用於公平性計算

        def dfs(curr_pos, time, ivm_acc, load, dual_acc, fairness_acc):
            nonlocal best_route, best_rc
            if best_rc < -0.0001:#*total_pos_dual:
                return

            # 1. 計算當前 Reduced Cost 並剪枝
            # RC = (Bus固定成本 + 時間成本 + 乘車時間成本 + 公平性成本) - (Dual和 + Mu)
            current_rc = (BUS_COUNT_WEIGHT + 
                          TOTAL_TIME_WEIGHT * time + 
                          ROUTE_TIME_WEIGHT * ivm_acc + 
                          FAIRNESS_WEIGHT * fairness_acc) - (dual_acc + self.mu)
            
            # 2. 動態剪枝邏輯：
            # 剩餘最大可能獲取的對偶值總和 = 總正對偶值 - 已獲得的對偶值
            # 如果 (當前 RC - 剩餘最大潛在 Dual 收益) 仍然大於最佳解，則剪枝
            if current_rc - (total_pos_dual - dual_acc)> best_rc:
                return

            # 2. 如果車輛已空，這是一個潛在的可行完整解
            if not onboard_groups and path:
                # 如果找到 RC 為負且不存在於現有路徑池中的新路徑，則將其設為最佳路徑並立即停止搜尋 (Early Exit)
                if current_rc < best_rc:
                    temp_route = self._build_route(path)
                    temp_g_set = frozenset(self._groups_in_route(temp_route))
                    if temp_g_set not in self.route_pool_keys: # 檢查是否為重複路徑
                        best_rc = current_rc
                        best_route = temp_route
                        return
                return

            # 3. 基礎邊界檢查
            if time > MAX_ROUTE_MIN or load > BUS_CAPACITY:
                return

            # 5. 擴展：前往學校 (Drop)
            onboard_sch_idxs = set(self.groups[gid]['sch_idx'] for gid in onboard_groups)
            for sch_idx in onboard_sch_idxs:
                sch_coord = self.schools[sch_idx].coord
                dist = travel_minutes(curr_pos, sch_coord)
                if time + dist > MAX_ROUTE_MIN: continue
                
                new_time = time + dist
                new_ivm = ivm_acc + (load * dist)
                
                # 處理下車與公平性計算
                dropping_gids = [gid for gid in onboard_groups if self.groups[gid]['sch_idx'] == sch_idx]
                dropped_load = sum(self.groups[gid]['count'] for gid in dropping_gids)
                
                local_fairness = 0.0
                temp_finished = []
                for gid in dropping_gids:
                    g = self.groups[gid]
                    g_ride = new_time - group_pickup_time[gid]
                    g_dist = travel_minutes(self.st_dict[g['st_idx']].coord, sch_coord)
                    
                    # 與「先前已下車」的群體比較
                    for f_sch, f_st, f_ride in finished_groups_info:
                        if f_sch == sch_idx:
                            f_dist = travel_minutes(self.st_dict[f_st].coord, sch_coord)
                            if (g_dist < f_dist - 0.01 and g_ride > f_ride + 0.01) or \
                               (f_dist < g_dist - 0.01 and f_ride > g_ride + 0.01):
                                local_fairness += 1.0
                    
                    # 與「同時下車」的其他群體比較 (避免重複計算，取 gid < gid2)
                    for gid2 in dropping_gids:
                        if gid >= gid2: continue
                        g2 = self.groups[gid2]
                        g2_ride = new_time - group_pickup_time[gid2]
                        g2_dist = travel_minutes(self.st_dict[g2['st_idx']].coord, sch_coord)
                        if (g_dist < g2_dist - 0.01 and g_ride > g2_ride + 0.01) or \
                           (g2_dist < g_dist - 0.01 and g2_ride > g_ride + 0.01):
                            local_fairness += 1.0
                            
                    temp_finished.append((sch_idx, g['st_idx'], g_ride))
                
                # Push Buffer (Drop)
                path.append(('s', sch_idx))
                for gid in dropping_gids: onboard_groups.remove(gid)
                finished_groups_info.extend(temp_finished)
                
                dfs(sch_coord, new_time, new_ivm, load - dropped_load, dual_acc, fairness_acc + local_fairness)
                
                # Pop Buffer (Drop)
                for _ in range(len(temp_finished)): finished_groups_info.pop()
                for gid in dropping_gids: onboard_groups.append(gid)
                path.pop()

            # 4. 擴展：撿起群體 (Pickup)
            for g in sorted_groups:
                if visited_groups[g['id']]: continue
                
                st_coord = self.st_dict[g['st_idx']].coord
                dist = travel_minutes(curr_pos, st_coord)
                
                if time + dist > MAX_ROUTE_MIN: continue
                if load + g['count'] > BUS_CAPACITY: continue

                # Push Buffer
                path.append(('g', g['id']))
                visited_groups[g['id']] = True
                onboard_groups.append(g['id'])
                group_pickup_time[g['id']] = time + dist
                
                new_time = time + dist
                new_ivm = ivm_acc + (load * dist) # 乘車時間累加：目前人數 * 此段行駛時間
                
                dfs(st_coord, new_time, new_ivm, load + g['count'], 
                    dual_acc + self.duals[g['id']], fairness_acc)
                
                # Pop Buffer
                del group_pickup_time[g['id']]
                onboard_groups.pop()
                visited_groups[g['id']] = False
                path.pop()

        # Pricing 起點：自由從各站點出發。
        for g in sorted_groups:
            st_coord = self.st_dict[g['st_idx']].coord
            # Push Buffer (起始群體)
            path.append(('g', g['id']))
            visited_groups[g['id']] = True
            onboard_groups.append(g['id'])
            group_pickup_time[g['id']] = 0.0
            
            dfs(st_coord, 0.0, 0.0, g['count'], self.duals[g['id']], 0.0)
            
            # Pop Buffer
            del group_pickup_time[g['id']]
            onboard_groups.pop()
            visited_groups[g['id']] = False
            path.pop()

            # 如果已經找到縮減成本為負的路徑，則根據 Early Exit 策略停止嘗試其他起始點
            if best_route and best_rc < -1e-4:
                break

        if best_route:
            tqdm.write(f"[Pricing] Found route with RC: {best_rc:.2f}")
        return best_route

    def _build_route(self, path: List[Tuple[str, int]]) -> Route:
        """根據路徑序列轉換為 Route 物件"""
        events = []
        for t, idx in path:
            if t == 'g':
                g = self.groups[idx]
                events.append(('pickup', (g['st_idx'], {g['sch_idx']: g['count']})))
            else:
                events.append(('drop', idx))

        r = Route(events=events)
        r.minutes, r.in_vehicle_minutes, r.pickup_detail, r.fairness_penalty = simulate_route(
            r, self.schools, self.st_dict
        )
        return r


    # ===============================
    # Main loop
    # ===============================
    def run(self) -> Solution:

        self.initialize_columns()

        for it in tqdm(range(DW_ITERATIONS), desc="DW/B&P Iterations"):

            m = self.solve_master(relax=True)
            if m.status != GRB.OPTIMAL:
                tqdm.write(f"[BP] Master problem not optimal at iteration {it}. Status: {m.status}")
                break

            # 每 10 次迭代輸出一次當前的 RMP 目標值
            if it % 10 == 0:
                tqdm.write(f"[DW Iter {it:03d}] 當前 RMP 目標值 (Relaxed): {m.objVal:.2f}")

            new_route = self.pricing()
            if new_route is None:
                tqdm.write(f"[DW Iter {it:03d}] 找不到具有負縮減成本的路徑，演算法收斂。")
                break
            if not self._add_route(new_route): # 使用 _add_route 進行去重檢查並嘗試加入
                tqdm.write(f"[DW Iter {it:03d}] 找到重複路徑，演算法收斂。") # 如果是重複路徑，則視為收斂
                break

        # final integer solve
        m = self.solve_master(relax=False)
        if m.status != GRB.OPTIMAL:
            return build_solution([], self.stations)

        x = m.getVars()
        selected = [self.routes[i] for i in range(len(self.routes)) if x[i].X > 0.5]
        return build_solution(selected, self.stations)

def run_dantzig_wolfe(schools: List[School], stations: List[Station]) -> Solution:
    solver = DantzigWolfeSolver(schools, stations)
    return solver.run()

if __name__ == "__main__":
    random.seed(42)
    schools, stations = gen_instance_multi()
    best_sol = run_dantzig_wolfe(schools, stations)
    print_solution_pretty(best_sol, stations, schools)
    m = plot_routes_on_map(best_sol, stations, schools, title="Dantzig-Wolfe Optimized")
    os.makedirs("./data", exist_ok=True)
    m.save("./data/dw_routes.html")
