# Section 5 演算法實作：基於 Deep Learning 輔助 Pricing 的列生成方法

**專注於神經策略網路、強化學習回饋與精確 DFS fallback 架構**

### 1. 文件目的與適用範圍

本文件說明 `DFS+DL_solver.py` 的演算法架構。此方法在 Dantzig-Wolfe 列生成框架中加入神經網路策略，嘗試用學習式方法引導 pricing 子問題搜尋負縮減成本路線。

它不是以深度學習直接輸出最終解，而是用策略網路輔助路線生成，並保留 DFS fallback 以維持基本穩定性。

### 2. 整體架構

系統包含三個核心層：

1. **Restricted Master Problem**：Gurobi 求解欄位選擇問題並產生對偶值。
2. **Policy Pricing**：神經網路根據目前狀態選擇下一個 pickup/drop 行動。
3. **DFS Fallback**：若神經網路沒有找到有效負 reduced cost route，改用 DFS + DP 搜尋。

這種架構兼顧探索能力與可行性保底。

### 3. 策略網路設計（Policy Network）

策略網路輸入包含：

* 節點靜態特徵：座標、節點類型、對偶訊號。
* 動態狀態：目前時間、載重、reduced cost、目前節點。
* action mask：哪些 pickup/drop 行動目前合法。

輸出是所有候選節點的選擇機率。

### 4. Pricing Rollout 邏輯

每次 rollout 從一個高對偶 group 出發：

1. 建立合法 action mask。
2. 用 policy network 取得下一步機率分佈。
3. 依機率抽樣 pickup/drop 行動。
4. 更新時間、載重、對偶累積與 path。
5. 若車上清空，計算完整路線 reduced cost。

若路線 reduced cost 為負，則可加入欄位池。

### 5. 強化學習回饋（REINFORCE）

當 rollout 找到負 reduced cost route 時，演算法用該 reduced cost 作為回饋：

* 越負的 reduced cost 代表越有價值。
* 透過 log probability 更新策略網路。
* 目標是提高未來產生高收益路線的機率。

此設計讓 pricing 搜尋能逐步偏向較有效的路線結構。

### 6. 精確 Fallback 機制

神經網路可能因尚未訓練充分而找不到新欄位。因此本方法保留 DFS + DP fallback：

* 若 policy rollout 沒有找到負 reduced cost route。
* 則啟動傳統 DFS pricing。
* 使用 memo、容量、時間與事件數限制進行剪枝。

這使演算法不完全依賴學習模型。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: DL-assisted Dantzig-Wolfe Pricing for SBRP
Input: Schools, Stations
Output: Integer Route Selection

1. Initialize route pool using LNS.
2. For each DW iteration:
   a. Solve relaxed master problem.
   b. Extract dual values.
   c. Run policy-network pricing rollouts.
   d. If negative reduced-cost route found:
        Train policy by REINFORCE and add route.
      Else:
        Run DFS + DP fallback pricing.
   e. Add new route if available.
3. Solve final binary master problem.
4. Return selected routes.
```

### 8. 實作建議與注意事項

1. **DL 是輔助，不是保證**：必須保留 fallback，否則 pricing 容易不穩。
2. **action mask 很重要**：非法行動若未遮罩，模型會產生不可行路線。
3. **訓練訊號稀疏**：只有找到負 reduced cost route 時才有強回饋，因此需要控制 rollout 數與搜尋深度。
