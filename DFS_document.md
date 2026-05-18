# Section 5 演算法實作：基於 DFS 路徑枚舉的校車路徑池模型

**專注於深度優先搜尋枚舉、多樣路徑池與集合劃分選路**

### 1. 文件目的與適用範圍

本文件說明 `DFS_solver.py` 的演算法架構。此方法先使用 DFS 枚舉大量可行路線，形成候選路徑池，再由 Gurobi 解集合劃分模型，選出覆蓋所有學生群組且成本最低的路線組合。

它不使用 Dantzig-Wolfe 的對偶值 pricing，而是採取一次性路徑池生成方式。

### 2. 群組與狀態表示

學生需求被拆成 group：

* 每個 group 對應一個 `(station, school)` 需求。
* DFS 狀態使用 bitmask 表示已拜訪 group 與車上 group。
* path 使用 `('g', gid)` 與 `('s', school_idx)` 表示 pickup/drop 順序。

此設計讓 DFS 能快速判斷哪些學生已上車、哪些仍在車上。

### 3. 基礎可行欄位

在 DFS 枚舉前，演算法先加入每個 group 的單獨直達路線：

```text
station -> school
```

這些基礎欄位確保每個 group 至少有被覆蓋的機會，降低整數模型 infeasible 的風險。

### 4. DFS 路徑枚舉邏輯

DFS 在每個狀態下有兩類擴展：

1. **Drop 分支**：前往車上學生的目的學校，並一次放下同校學生。
2. **Pickup 分支**：選擇尚未拜訪且容量允許的 group 上車。

每次擴展都會檢查：

* 路線時間是否超過上限。
* 車上載重是否超過容量。
* 路徑長度是否超過搜尋保護限制。

### 5. 路徑池管理

DFS 每找到一條完整路線，會：

1. 轉換為 `Route`。
2. 透過 `simulate_route` 計算路線資訊。
3. 用 group set 去重。
4. 建立 group-to-route 索引，加速後續建模。

路徑池大小受 `ROUTE_POOL_SIZE` 控制，以避免完全枚舉造成組合爆炸。

### 6. 整數規劃選路模型

路徑池建立後，模型建立二元變數：

* `x_r = 1` 代表選擇路線 r。

約束包含：

* 每個 group 被剛好覆蓋一次。
* 選取路線數不超過車輛上限。

目標函數最小化選取路線的總 `route_cost`。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: DFS Route Pool + IP for SBRP
Input: Schools, Stations, Route Pool Size
Output: Selected Feasible Routes

1. Build groups from station-school demands.
2. Add single-group direct routes.
3. For each group as DFS start:
   a. Explore drop branches.
   b. Explore bounded pickup branches.
   c. Store complete feasible routes.
4. Stop when route pool reaches target size.
5. Solve set partitioning model.
6. Return selected routes and optionally merge routes.
```

### 8. 實作建議與注意事項

1. **DFS 必須有分支限制**：不限制 pickup 分支會造成排列組合爆炸。
2. **路徑池不是越大越好**：過大會讓 Gurobi 建模與求解變慢。
3. **適合作為 baseline**：此方法結構直觀，適合比較其他進階方法的效果。
