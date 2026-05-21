# Section 5 演算法實作：基於 DFS + DP 記憶化的列生成 Pricing

**專注於狀態記憶化、縮減成本剪枝與 Dantzig-Wolfe 架構**

### 1. 文件目的與適用範圍

本文件說明 `DFS+DP_solver.py` 的演算法架構。此方法在 Dantzig-Wolfe 主問題下，使用 DFS 作為 pricing 子問題，並加入 DP-style memoization 記憶化，避免重複搜尋相同狀態。

它適合需要比純 DFS 更穩定、但仍希望保留可解釋搜尋邏輯的情境。

### 2. 欄位生成架構

整體架構分為兩層：

* **Master Problem**：在目前欄位池中選擇路線，求 LP relaxation 或最終整數解。
* **Pricing Problem**：根據 master 的對偶值，用 DFS 找新的負縮減成本路線。

每條 route 都是一個可被 master 選取的欄位。

### 3. 初始欄位與可行性

初始欄位由 LNS 產生，確保：

1. 所有 group 至少被某些 route 覆蓋。
2. Master problem 一開始有可行解。
3. 對偶值具有實際意義。

若初始欄位不足，後續 pricing 很難穩定運作。

### 4. DFS Pricing 狀態

DFS 狀態包含：

* 目前節點。
* 已花費時間。
* 累積乘車時間。
* 車上人數。
* 已收集的對偶值。
* 已拜訪與車上 group bitmask。
* 路徑事件序列。

每個狀態都對應一個部分路線。

### 5. DP 記憶化與剪枝

記憶化 key 通常包含：

```text
(current_node, visited_mask, onboard_mask, load)
```

若同一狀態曾經以更低 reduced cost 到達，新的較差狀態會被剪掉。

此外還有 reduced cost lower bound 剪枝：

* 若即使取得所有剩餘正對偶值也無法改善最佳 reduced cost，則停止擴展。

### 6. Drop / Pickup 擴展策略

DFS 每一步有兩種行動：

1. **Drop**：前往車上某批學生的學校，並放下同校學生。
2. **Pickup**：前往尚未拜訪且容量允許的 group。

演算法使用時間上限、容量上限、事件數上限與分支上限控制搜尋規模。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: DFS + DP Pricing for Dantzig-Wolfe SBRP
Input: Schools, Stations, Initial Columns
Output: Integer Route Selection

1. Initialize route pool with LNS routes.
2. For each DW iteration:
   a. Solve relaxed master problem.
   b. Extract dual values.
   c. Run DFS pricing:
      i. Expand drop and pickup decisions.
      ii. Apply DP memo pruning.
      iii. Track best negative reduced-cost route.
   d. Add new route if found.
3. Solve final binary master problem.
4. Return selected routes.
```

### 8. 實作建議與注意事項

1. **memo key 不宜過細**：過細會降低重用率，過粗則可能錯剪。
2. **對偶排序會影響速度**：優先嘗試高對偶或低覆蓋 group 通常較快找到新欄位。
3. **DP 不是完整動態規劃**：它主要扮演 DFS 搜尋中的 dominance pruning。
