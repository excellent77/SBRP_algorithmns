# Section 5 演算法實作：基於分段枚舉與 Master IP 的 Dantzig-Wolfe 2

**專注於分段路徑枚舉、兩校路線組合與保底可行欄位**

### 1. 文件目的與適用範圍

本文件說明 `dantzig_wolfe_2_solver.py` 的演算法架構。此版本不是標準逐輪 pricing 的列生成，而是先依照路線型態枚舉可行 segment，再將 segment 組合成候選完整路線，最後由 Gurobi 解 master IP 選路。

它可視為「結構化路徑枚舉 + 集合劃分」的 Dantzig-Wolfe 變體。

### 2. 資料抽象與 Group 定義

此版本將專案資料轉成簡化節點：

* `A`、`B` 代表兩所學校。
* 站點用字串編號表示。
* `Group` 表示某站點前往某學校的一批學生。
* `Segment` 表示從起點到終點的一段接送路徑。
* `InternalRoute` 表示 master problem 可選的一條候選路線。

這些抽象讓路線可以依學校方向分段枚舉。

### 3. 分段枚舉（Bounded-Length Segment Enumeration）

`enumerate_segments` 使用 DFS 生成：

```text
start -> selected stops -> end school
```

每個 segment 必須符合：

* 總時間不超過路線上限。
* 載重不超過容量。
* pickup 分支數與 segment 數量受限制。
* 後續接站可依距離學校的方向性剪枝。

此設計避免完整路線一次枚舉造成爆炸。

### 4. 路線型態設計

候選路線主要包含：

1. **A-only route**：接 A 校學生並送至 A。
2. **B-only route**：接 B 校學生並送至 B。
3. **B→A composition route**：先服務 B 校，再從 B 出發接 A 校學生並送至 A。
4. **LNS route**：由 LNS 產生的完整可行路線，作為保底欄位。

加入 LNS route 的目的，是確保 master problem 至少有一組完整可行解。

### 5. 路線組合與可行性

組合 B→A 路線時，會檢查：

* 兩段不能使用相同中間站點。
* 兩段不能接同一 group。
* 合併總時間不超過上限。
* 兩段各自載重不超過容量。

由於 B 校學生在 B 已下車，後續接 A 校學生時載重不重疊，因此載重檢查應使用兩段最大值，而不是兩段相加。

### 6. Master IP 模型

Master problem 使用二元變數：

* `x_r = 1` 代表選擇第 r 條 `InternalRoute`。

約束包含：

* 每個 group 必須被剛好覆蓋一次。
* 選擇路線數不得超過最大公車數。

目標函數最小化所有被選路線成本。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: Dantzig-Wolfe 2 Segment Pool for SBRP
Input: Schools, Stations
Output: Selected Feasible Routes

1. Convert project data into Group and simplified node labels.
2. Build distance matrix.
3. Add LNS feasible routes as safety columns.
4. Enumerate A-only segments and add routes.
5. Enumerate B-only segments and add routes.
6. Compose B segments with A segments to build B->A routes.
7. Build set partitioning master IP.
8. Solve with Gurobi and convert selected InternalRoutes back to ProjectRoute.
```

### 8. 實作建議與注意事項

1. **必須保證 route pool 覆蓋所有 group**：否則 `cover_g == 1` 會直接 infeasible。
2. **B-only 欄位不可省略**：只做 B→A 組合會漏掉只服務 B 的合理路線。
3. **分段上限會影響品質**：枚舉上限越小速度越快，但可能漏掉好路線。
