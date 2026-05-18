# Section 5 演算法實作：基於 Dantzig-Wolfe 列生成的校車路徑優化

**專注於主問題、Pricing 子問題與縮減成本路線生成**

### 1. 文件目的與適用範圍

本文件說明 `dantzig_wolfe_solver.py` 的演算法架構。Dantzig-Wolfe 方法將 SBRP 拆成兩部分：主問題負責從候選路線中選路，pricing 子問題負責根據對偶值產生新的高價值路線。

此方法適合路線組合龐大、無法一次列舉所有可行路線的情況。

### 2. 問題分解設計

每條可行校車路線被視為一個欄位（Column）：

* 欄位覆蓋若干 group。
* 欄位有自己的 route cost。
* 主問題選擇欄位集合以覆蓋所有 group。

完整路線空間由 pricing 動態補充，而不是一次全部產生。

### 3. 初始欄位建構

演算法先使用 LNS 產生初始可行解，並將其中每條路線加入欄位池。

若某些 group 沒被初始欄位覆蓋，會補入單 group 直達路線，以確保 restricted master problem 可行。

### 4. Restricted Master Problem

Master problem 是集合劃分模型：

* `x_r` 表示是否選擇 route r。
* 每個 group 必須被剛好覆蓋一次。
* 總路線數不得超過車輛上限。

在列生成迭代中，先求 LP relaxation，取得 group 覆蓋約束與車輛數約束的對偶值。

### 5. Pricing 子問題

Pricing 目標是尋找 reduced cost 為負的新 route：

$$RC(r) = Cost(r) - \sum_{g \in r}\pi_g - \mu$$

若找到 `RC < 0` 的路線，代表加入此欄位能改善 master LP。

本實作用 DFS 生成候選路線，依據對偶值排序 group，使高潛力路線較早被搜尋。

### 6. DFS Pricing 架構

DFS 狀態包含：

* 目前位置。
* 時間、載重、乘車時間。
* 車上 group。
* 已拜訪 group。
* 累積對偶值與公平性懲罰。

擴展行動包含 pickup 與 drop，並透過容量、時間、路線事件數與 reduced cost bound 進行剪枝。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: Dantzig-Wolfe Column Generation for SBRP
Input: Schools, Stations
Output: Best Integer Route Set

1. Generate initial columns using LNS and fallback direct routes.
2. For iteration = 1 to DW_ITERATIONS:
   a. Solve relaxed restricted master problem.
   b. Read dual values.
   c. Run DFS pricing to find negative reduced-cost route.
   d. If found:
        Add route to column pool.
      Else:
        Stop column generation.
3. Solve final master problem as binary IP.
4. Return selected routes.
```

### 8. 實作建議與注意事項

1. **初始欄位必須可行**：否則對偶值無法穩定引導 pricing。
2. **pricing 是主要瓶頸**：DFS 分支與深度控制會直接影響速度。
3. **最終仍需整數求解**：LP 收斂不等於整數最佳，最後要解 binary master。
