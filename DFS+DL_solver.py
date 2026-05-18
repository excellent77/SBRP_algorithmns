import os
import random
from tqdm import tqdm
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.data_models import School, Station, Route, Solution
from utils.solution_utils import (
    simulate_route,
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
DW_ITERATIONS = 200
MAX_PRICING_STARTS = 18
MAX_PICKUP_BRANCHES = 10
MAX_ROUTE_EVENTS = 14



# =====================================================================
# 1. Deep Learning Policy Network Definition with Training Support
# =====================================================================
class PricingPolicyNet(nn.Module):
    def __init__(self, node_input_dim=4, hidden_dim=128, lr=1e-3):
        super(PricingPolicyNet, self).__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.state_encoder = nn.Linear(4, hidden_dim)
        self.attn_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Parameter(torch.rand(hidden_dim))
        
        # Optimizer for Reinforcement Learning
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, static_node_features, dynamic_state, action_mask):
        num_nodes = static_node_features.size(0)
        node_embeddings = self.node_encoder(static_node_features)
        state_embedding = self.state_encoder(dynamic_state).unsqueeze(0).repeat(num_nodes, 1)
        combined_context = torch.cat([node_embeddings, state_embedding], dim=1)
        scores = torch.tanh(self.attn_query(combined_context))
        logits = torch.matmul(scores, self.attn_v)
        logits = logits.masked_fill(action_mask == 0, -1e9)
        return F.softmax(logits, dim=-1)

    def update_policy(self, log_probs, reward):
        """
        REINFORCE Update: Loss = -log_prob * Reward
        In Column Generation, a 'Good' reward is a highly negative Reduced Cost.
        We want to minimize RC, so we maximize (-RC).
        """
        if not log_probs:
            return
        
        # Reward is the negative Reduced Cost (we want to maximize profit/incentive)
        # We normalize the reward to stabilize training
        R = torch.tensor(reward, dtype=torch.float32)
        
        policy_loss = []
        for lp in log_probs:
            # We use -R because the optimizer minimizes loss
            # Minimizing (-log_prob * -RC) = Minimizing (log_prob * RC)
            policy_loss.append(-lp * (-R)) 
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
        """使用啟發式解初始化路徑池，確保 Master Problem 一開始就是可行的"""
        print("[DW] 正在初始化路徑池 (Initial Columns)...")
        # 取得一個高品質初始解，確保滿足車輛數限制
        init_sol = run_lns(self.schools, self.stations, print_log=False)
        for r in init_sol.routes:
            self._add_route(r)
            
        # 檢查是否所有群體都被涵蓋
        covered_groups = set().union(*(r._g_ids for r in self.routes if hasattr(r, '_g_ids')))
        
        for g in self.groups:
            if g['id'] not in covered_groups:
                raise ValueError(f"初始路徑池未涵蓋所有群體！缺少群體 ID: {g['id']}")

    # ===============================
    # Master Problem (Set Partitioning)
    # ===============================
    def solve_master(self, relax=True):
        m = gp.Model("RMP")
        m.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if relax else GRB.BINARY

        x = m.addVars(len(self.routes), vtype=vtype, lb=0, ub=1, name="x")
        # 增加人工變數確保可行性 (Set Partitioning with Slack)，並引導對偶值
        z = m.addVars(len(self.groups), vtype=vtype, lb=0, name="z")
        
        penalty = 1e6

        m.setObjective(
            gp.quicksum(x[i] * route_cost(self.routes[i]) for i in range(len(self.routes))) +
            gp.quicksum(z[g] * penalty for g in range(len(self.groups))),
            GRB.MINIMIZE
        )

        constrs = {}
        for g_idx in range(len(self.groups)):
            idxs = self.group_to_routes[g_idx] # 使用預建索引提升效能
            # 每個群體必須被覆蓋剛好一次
            constrs[g_idx] = m.addConstr(gp.quicksum(x[i] for i in idxs) + z[g_idx] == 1, name=f"Cover_G{g_idx}")

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
    

    
    # =====================================================================
    # 2. Main Modified Pricing Function with REINFORCE Training
    # =====================================================================
    def pricing(self):
        n = len(self.groups)
        best_route = None
        best_rc = 0.0
        total_pos_dual = sum(max(0, d) for d in self.duals)

        # ... [Node Construction & Distance Matrix Setup - Same as before] ...
        school_offset = len(self.st_dict)
        station_nodes = {st_idx: i for i, st_idx in enumerate(self.st_dict)}
        school_nodes = {i: school_offset + i for i, sch in enumerate(self.schools)}
        total_nodes = school_offset + len(self.schools)
        
        # [Distance Matrix Setup Logic]
        coords = [None] * total_nodes
        for st_idx, nid in station_nodes.items(): coords[nid] = self.st_dict[st_idx].coord
        for sch_idx, nid in school_nodes.items(): coords[nid] = self.schools[sch_idx].coord
        dist_matrix = [[travel_minutes(coords[i], coords[j]) if i != j else 0.0 for j in range(total_nodes)] for i in range(total_nodes)]

        # =========================================================
        # DEEP LEARNING COMPONENT: Initialization
        # =========================================================
        static_features_list = []
        for i in range(total_nodes):
            node_type = 0.0 if i < school_offset else 1.0
            signal = float(self.duals[i]) if i < len(self.duals) else 0.0
            static_features_list.append([coords[i][0], coords[i][1], node_type, signal])
        static_node_features = torch.tensor(static_features_list, dtype=torch.float32)
        
        if not hasattr(self, 'policy_net'):
            self.policy_net = PricingPolicyNet(node_input_dim=4, hidden_dim=128)

        # ========================================================
        # TRAINING MODE: Rollouts with Stochastic Sampling
        # ========================================================
        for start_g in sorted(self.groups, key=lambda x: self.duals[x['id']], reverse=True)[:5]:
            gid = start_g['id']
            curr_node = station_nodes[start_g['st_idx']]
            
            # State trackers
            time, ivm_acc, load, dual_acc, fairness_acc = 0.0, 0.0, start_g['count'], self.duals[gid], 0.0
            visited_mask, onboard_mask = (1 << gid), (1 << gid)
            path = [('g', gid)]
            
            log_probs = [] # To store the log-probabilities of actions taken
            terminated = False
            
            while not terminated:
                current_rc = (BUS_COUNT_WEIGHT + TOTAL_TIME_WEIGHT * time + ROUTE_TIME_WEIGHT * ivm_acc - dual_acc - self.mu)
                
                # 1. Build Action Mask
                action_mask = torch.zeros(total_nodes)
                # Pickups
                for g in self.groups:
                    if not (visited_mask & (1 << g['id'])) and (load + g['count'] <= BUS_CAPACITY):
                        if time + dist_matrix[curr_node][station_nodes[g['st_idx']]] <= MAX_ROUTE_MIN:
                            action_mask[station_nodes[g['st_idx']]] = 1.0
                # Dropoffs
                for gid2 in range(n):
                    if onboard_mask & (1 << gid2):
                        sch_nd = school_nodes[self.groups[gid2]['sch_idx']]
                        if time + dist_matrix[curr_node][sch_nd] <= MAX_ROUTE_MIN:
                            action_mask[sch_nd] = 1.0

                if torch.sum(action_mask) == 0: break
                    
                # 2. Forward Pass
                dynamic_state = torch.tensor([time, float(load), current_rc, float(curr_node)], dtype=torch.float32)
                probs = self.policy_net(static_node_features, dynamic_state, action_mask)
                
                # 3. Stochastic Sampling for Exploration (Training Mode)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                next_node = action.item()
                
                # 4. State Transition
                dist = dist_matrix[curr_node][next_node]
                time += dist; ivm_acc += load * dist; curr_node = next_node
                
                if next_node < school_offset: # Pickup
                    tgt_g = next(g for g in self.groups if station_nodes[g['st_idx']] == next_node)
                    visited_mask |= (1 << tgt_g['id']); onboard_mask |= (1 << tgt_g['id'])
                    load += tgt_g['count']; dual_acc += self.duals[tgt_g['id']]; path.append(('g', tgt_g['id']))
                else: # Dropoff
                    tgt_sch_idx = next(s_idx for s_idx, n_idx in school_nodes.items() if n_idx == next_node)
                    dropping = [i for i in range(n) if (onboard_mask & (1 << i)) and self.groups[i]['sch_idx'] == tgt_sch_idx]
                    load -= sum(self.groups[i]['count'] for i in dropping)
                    for i in dropping: onboard_mask &= ~(1 << i)
                    path.append(('s', tgt_sch_idx))
                    
                if onboard_mask == 0 and path:
                    final_rc = (BUS_COUNT_WEIGHT + TOTAL_TIME_WEIGHT * time + ROUTE_TIME_WEIGHT * ivm_acc - dual_acc - self.mu)
                    
                    # 5. Feedback Loop (Reinforcement)
                    # We update the policy based on how much this route improves the Reduced Cost
                    if final_rc < 0:
                        self.policy_net.update_policy(log_probs, final_rc)
                    
                    if final_rc < best_rc:
                        temp_route = self._build_route(path)
                        if frozenset(self._groups_in_route(temp_route)) not in self.route_pool_keys:
                            best_rc, best_route = final_rc, temp_route
                    terminated = True

        # ========================================================
        # EXACT FALLBACK MECHANISM (Keep this to ensure convergence)
        # ========================================================
        if best_route is None or best_rc >= -1e-4:
            # 統計各群體目前在路徑池中被覆蓋的次數 (times)
            coverage_counts = [len(self.group_to_routes[g['id']]) for g in self.groups]
            sorted_groups = sorted(
                self.groups,
                key=lambda g:coverage_counts[g['id']], #self.duals[g['id']]
                #reverse=True
            )
            # =========================
            # DFS + DP
            # =========================
            memo = {}
            def dfs(
                curr_node,
                time,
                ivm_acc,
                load,
                dual_acc,
                fairness_acc,
                visited_mask,
                onboard_mask,
                path
            ):
                nonlocal best_route, best_rc

                if best_rc < -0.0001:
                    return
                if len(path) > MAX_ROUTE_EVENTS:
                    return

                current_rc = (
                    BUS_COUNT_WEIGHT
                    + TOTAL_TIME_WEIGHT * time
                    + ROUTE_TIME_WEIGHT * ivm_acc
                    + FAIRNESS_WEIGHT * fairness_acc
                    - dual_acc
                    - self.mu
                )

                if current_rc - (total_pos_dual - dual_acc) > best_rc:
                    return

                state = (
                    curr_node,
                    visited_mask,
                    onboard_mask,
                    load
                )

                prev_best = memo.get(state)
                if prev_best is not None and prev_best <= current_rc:
                    return

                memo[state] = current_rc
                if onboard_mask == 0 and path:
                    if current_rc < best_rc:
                        temp_route = self._build_route(path)
                        temp_g_set = frozenset(self._groups_in_route(temp_route))
                        if temp_g_set not in self.route_pool_keys: # 檢查是否為重複路徑
                            best_rc = current_rc
                            best_route = temp_route
                            return # 找到新的最佳路徑，立即停止 DFS 搜尋
                    #return
                

                if time > MAX_ROUTE_MIN:
                    return
                if load > BUS_CAPACITY:
                    return

                for gid in range(n):
                    if not (onboard_mask & (1 << gid)):
                        continue
                    g = self.groups[gid]
                    sch_idx = g['sch_idx']
                    # all groups to same school
                    dropping = []
                    for gid2 in range(n):
                        if onboard_mask & (1 << gid2):
                            if self.groups[gid2]['sch_idx'] == sch_idx:
                                dropping.append(gid2)

                    sch_node = school_nodes[sch_idx]
                    dist = dist_matrix[curr_node][sch_node]
                    new_time = time + dist

                    if new_time > MAX_ROUTE_MIN:
                        continue

                    dropped_load = sum(
                        self.groups[x]['count']
                        for x in dropping
                    )

                    new_onboard = onboard_mask
                    for x in dropping:
                        new_onboard &= ~(1 << x)

                    new_ivm = ivm_acc + load * dist
                    new_path = path + [('s', sch_idx)]

                    dfs(
                        sch_node,
                        new_time,
                        new_ivm,
                        load - dropped_load,
                        dual_acc,
                        fairness_acc,
                        visited_mask,
                        new_onboard,
                        new_path
                    )


                pickup_branches = 0
                for g in sorted_groups:
                    gid = g['id']
                    if visited_mask & (1 << gid):
                        continue
                    if load + g['count'] > BUS_CAPACITY:
                        continue

                    st_node = station_nodes[g['st_idx']]
                    dist = dist_matrix[curr_node][st_node]
                    new_time = time + dist

                    if new_time > MAX_ROUTE_MIN:
                        continue

                    new_visited = visited_mask | (1 << gid)
                    new_onboard = onboard_mask | (1 << gid)
                    new_ivm = ivm_acc + load * dist
                    new_path = path + [('g', gid)]

                    dfs(
                        st_node,
                        new_time,
                        new_ivm,
                        load + g['count'],
                        dual_acc + self.duals[gid],
                        fairness_acc,
                        new_visited,
                        new_onboard,
                        new_path
                    )
                    pickup_branches += 1
                    if pickup_branches >= MAX_PICKUP_BRANCHES:
                        break

            # =========================
            # Start DFS
            # =========================

            for g in sorted_groups[:MAX_PRICING_STARTS]:
                gid = g['id']
                st_node = station_nodes[g['st_idx']]
                dfs(
                    st_node,
                    0.0,
                    0.0,
                    g['count'],
                    self.duals[gid],
                    0.0,
                    (1 << gid),
                    (1 << gid),
                    [('g', gid)]
                )
                if best_route and best_rc < -1e-4:
                    break

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
            return Solution([], 0, 0, 0, 0, False)

        x = m.getVars()
        selected = [self.routes[i] for i in range(len(self.routes)) if x[i].X > 0.5]

        total_min = sum(r.minutes for r in selected)
        total_ivm = sum(r.in_vehicle_minutes for r in selected)
        total_fair = sum(r.fairness_penalty for r in selected)
        served = sum(sum(c for _, _, c, _ in r.pickup_detail) for r in selected)

        return Solution(selected, total_min, total_ivm, total_fair, served, True)

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
    m.save("./data/dfs_dl_routes.html")
