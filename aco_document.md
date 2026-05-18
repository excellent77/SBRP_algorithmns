# Section 5 演算法實作：基於蟻群演算法的校車路徑建構

**專注於費洛蒙導引、啟發式選站與可行路徑構造邏輯**

### 1. 文件目的與適用範圍

本文件說明 `aco_solver.py` 如何利用蟻群演算法（Ant Colony Optimization, ACO）求解多校校車路徑問題。ACO 的核心不是先建立完整數學模型，而是讓多隻「螞蟻」反覆建構可行路線，並根據解的品質更新費洛蒙，使後續搜尋更偏向高品質的站點選擇。

此方法適合作為快速產生可行解、初始化其他演算法，或在中小規模實例中取得穩定啟發式解。

### 2. 解的表示方式（Route Construction）

ACO 不使用染色體或欄位變數，而是直接建構 `Route.events`：

* **Pickup Event**：代表車輛到站點接某些學校的學生。
* **Drop Event**：代表車輛前往某所學校下車。
* **完整路線**：由多個 pickup/drop 事件組成，並透過 `simulate_route` 計算時間、乘車時間與公平性懲罰。

每隻螞蟻會逐步選擇下一個站點或學校，直到所有學生被服務，或達到車輛數、容量、時間限制。

### 3. 初始化策略：費洛蒙與啟發式資訊

演算法使用兩類資訊導引搜尋：

1. **費洛蒙矩陣 (`tau`)**：記錄過去較佳解中常出現的選擇，代表經驗偏好。
2. **啟發式資訊 (`eta`)**：通常與距離或旅行時間反比，鼓勵選擇較近的站點。

在起始階段，費洛蒙通常為均勻值，使早期搜尋具有較高探索性。

### 4. 路徑生成邏輯（Ant Construction）

每隻螞蟻建構解時，會重複執行以下決策：

1. **判斷是否應下車**：若車上乘客接近容量上限，或繼續接站會導致路線超時，則優先前往學校下車。
2. **選擇下一站點**：根據費洛蒙強度與距離啟發值計算機率，隨機抽樣下一個服務站點。
3. **容量與時間檢查**：只接受不違反容量上限與路線時間上限的 pickup/drop。
4. **路線收尾**：若車上仍有學生，會將其送往對應學校，形成完整 route。

這種「建構式」流程讓 ACO 能自然處理多校與多站點需求。

### 5. 費洛蒙更新機制

每輪迭代後，演算法會根據螞蟻產生的解更新費洛蒙：

* **揮發（Evaporation）**：降低所有舊費洛蒙，避免過早收斂。
* **增強（Deposit）**：對成本較低、品質較好的解增加費洛蒙。
* **菁英加權（Elite Reinforcement）**：較佳解對費洛蒙的影響更大，使搜尋逐漸集中。

此機制在「探索」與「利用」之間取得平衡。

### 6. 成本函數與可行性評估

每個候選解會透過共同成本函數評估：

$$Cost = \text{BusCount} \times f_b + \text{TotalTime} \times \alpha + \text{InVehicleTime} \times \beta + \text{FairnessPenalty} \times \gamma$$

其中主要考量：

* 使用車輛數。
* 所有路線總行駛時間。
* 學生總乘車時間。
* 同校學生間的不公平懲罰。

不可行解會被排除或給予極高成本。

### 7. 演算法流程（Algorithm Pipeline）

```text
Algorithm: ACO for SBRP
Input: Schools, Stations, Ant Count, Iterations
Output: Best Feasible Solution

1. Initialize pheromone matrix tau and heuristic matrix eta.
2. For each iteration:
   a. For each ant:
      i. Construct routes by probabilistic station/drop decisions.
      ii. Simulate routes and evaluate solution cost.
   b. Select the best solutions of this iteration.
   c. Evaporate pheromone globally.
   d. Deposit pheromone based on high-quality routes.
   e. Update the global best solution.
3. Return the best feasible solution.
```

### 8. 實作建議與注意事項

1. **避免過早收斂**：若解太快集中，可提高揮發率或降低費洛蒙權重。
2. **建構可行性優先**：ACO 的優勢在快速產生可行路線，應讓容量與時間檢查盡早剪枝。
3. **可作為初始化器**：ACO 產生的解可供 LNS、Dantzig-Wolfe 或 Matheuristic 作為初始欄位或起始解。
