# Section 5 演算法實作：基於近校約束 DFS Pricing 的 Dantzig-Wolfe 方法

**專注於近校剪枝、對偶導引與列生成 pricing 架構**

### 1. 文件目的與適用範圍

本文件說明 `DFS_near_solver.py` 的演算法架構。此方法屬於 Dantzig-Wolfe / Branch-and-Price 風格：主問題負責選路，pricing 子問題使用 DFS 產生負縮減成本的新路線。

與一般 DFS pricing 不同，本版本加入「越接越靠近目標學校」的近校約束，用來縮小搜尋空間。

### 2. 主問題與欄位結構

主問題是一個集合劃分模型：

* 每條 route 是一個欄位。
* 每個 group 必須被覆蓋一次。
* 路線總數不得超過公車數上限。

求解 LP relaxation 後，模型會輸出 group 覆蓋約束的對偶值，供 pricing 使用。

### 3. 初始欄位策略

演算法先使用 LNS 產生高品質可行解，將其中路線加入欄位池。

初始欄位的作用是：

* 讓 restricted master problem 一開始可行。
* 提供合理的對偶值。
* 避免從空集合開始造成 infeasible。

### 4. Pricing DFS 狀態設計

Pricing 子問題用 DFS 尋找負縮減成本路線。狀態包含：

* 目前節點。
* 目前時間與車上載重。
* 已拜訪 group bitmask。
* 車上 group bitmask。
* 累積對偶收益。
* 路徑事件序列。

DFS 會根據目前 reduced cost 判斷是否值得繼續擴展。

### 5. 近校剪枝邏輯（Near-School Rule）

本版本的核心剪枝是：

```text
下一個 pickup group 必須比目前 group 更接近自己的目標學校
```

此規則帶來兩個效果：

1. 降低 pickup 排列數量。
2. 鼓勵路線沿著往學校方向收斂，避免繞遠。

它犧牲部分完整性，但能顯著降低 DFS pricing 的搜尋負擔。

### 6. Reduced Cost 評估

Pricing 目標是尋找：

$$RC = RouteCost - \sum_{g \in r} \pi_g - \mu < 0$$

其中：

* `RouteCost` 為路線實際成本。
* `π_g` 為 group 覆蓋約束對偶值。
* `μ` 為車輛數限制對偶值。

若找到負 reduced cost route，該路線會被加入 master problem。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: DFS-Near Dantzig-Wolfe for SBRP
Input: Schools, Stations
Output: Best Integer Route Selection

1. Generate initial columns using LNS.
2. Repeat for DW iterations:
   a. Solve restricted master LP.
   b. Read dual values.
   c. Run DFS pricing with near-school pruning.
   d. If negative reduced-cost route found:
        Add route to column pool.
      Else:
        Stop column generation.
3. Solve final integer master problem.
4. Return selected routes.
```

### 8. 實作建議與注意事項

1. **近校規則是啟發式剪枝**：速度提升明顯，但可能漏掉部分高品質繞行路線。
2. **應快取距離矩陣**：pricing 每輪都會被呼叫，重算距離會拖慢整體。
3. **限制 DFS 深度與分支**：對 50 個以上 group 的實例，分支控制是必要條件。
