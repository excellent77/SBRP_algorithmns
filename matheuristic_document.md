# Section 5 演算法實作：基於 Matheuristic 的路徑池整數規劃

**專注於啟發式路徑池生成與集合劃分模型架構**

### 1. 文件目的與適用範圍

本文件說明 `matheuristic_solver.py` 的演算法架構。Matheuristic 是「啟發式 + 數學規劃」的混合方法：先用啟發式方法產生一批高品質候選路線，再由 Gurobi 解集合劃分模型，從候選路線中選出總成本最低且覆蓋所有學生群組的組合。

此方法適合在純啟發式與完整精確模型之間取得折衷：路徑生成不求完備，但最終選路由整數規劃負責。

### 2. 路徑池設計（Route Pool）

演算法的核心資料結構是候選路徑池 `route_pool`。每條路線是一個完整 `Route`，包含：

* pickup/drop 事件序列。
* 模擬後的行駛時間。
* 學生乘車時間。
* 對應服務的 group 集合。

主模型不直接決定車輛如何行駛，而是在已產生的 route pool 中選擇路線。

### 3. 路徑池生成策略

路徑池由兩種來源構成：

1. **LNS 解抽取**：反覆執行 LNS，將其中的可行路線加入候選池。
2. **隨機站點切割**：隨機排列站點，依容量限制切成多條路線，補充多樣性。

這種混合策略讓 route pool 同時包含高品質路線與結構較多樣的路線。

### 4. 群組覆蓋映射（Group Coverage）

每條路線會被轉換為其服務的 group ID 集合：

* group 定義為「同一站點、同一學校」的學生需求。
* `pickup_detail` 用於回推路線覆蓋哪些 group。
* 最終整數模型會用這些覆蓋關係建立約束。

此映射是連接路徑層與數學模型層的核心。

### 5. 集合劃分模型（Set Partitioning）

Matheuristic 的精確階段建立二元變數：

* `x_r = 1`：選擇第 r 條候選路線。
* `x_r = 0`：不選擇該路線。

主要約束：

1. **群組覆蓋約束**：每個 group 必須被剛好一條被選路線覆蓋。
2. **車輛數限制**：選取路線數不得超過最大公車數。

目標函數為最小化被選路線的總成本。

### 6. 成本函數（Objective Function）

每條候選路線使用共同的 `route_cost`：

$$Cost_r = \text{BusCost} + \text{TravelTimeCost} + \text{InVehicleTimeCost} + \text{FairnessCost}$$

整數規劃目標為：

$$\min \sum_r Cost_r x_r$$

此設計讓模型在選路時同時考慮派車數、總行駛時間、乘車時間與公平性。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: Matheuristic for SBRP
Input: Schools, Stations, Route Pool Size
Output: Best Selected Route Set

1. Define student groups by station-school pairs.
2. Generate route pool:
   a. Run LNS repeatedly and collect feasible routes.
   b. Add random split routes to improve diversity.
   c. Remove duplicate routes.
3. Build group-to-route coverage map.
4. Solve set partitioning model with Gurobi:
   a. Each group covered exactly once.
   b. Number of selected routes <= bus limit.
5. Convert selected routes into Solution.
```

### 8. 實作建議與注意事項

1. **路徑池品質決定上限**：若 route pool 沒有好路線，Gurobi 也無法選出好解。
2. **必須檢查未覆蓋 group**：若某些 group 沒有任何候選路線，模型會 infeasible。
3. **去重很重要**：大量重複路線會增加模型大小，但不會增加搜尋能力。
