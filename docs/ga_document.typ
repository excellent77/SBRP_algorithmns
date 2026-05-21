#set page(paper: "a4", margin: auto)
#set text(font: "Noto Sans CJK TC", size: 12pt)
#show heading: it=> block(
     below:12pt,
     it
)
#show heading.where(level: 1): it => block(width: 100%, below:12pt, align(center, it))

= 基於遺傳演算法

*專注於啟發式搜索與染色體解碼邏輯*

=== 1. 文件目的與適用範圍

本文件詳細說明如何利用遺傳演算法（GA）求解多校校車路徑問題（Multi-School SBRP）。與 Section 4 的列生成（Column Generation）不同，本方法不依賴對偶值，而是透過模擬生物演化的過程（交配、突變、選擇），在龐大的解空間中進行全局搜索。

=== 2. 染色體編碼設計（Chromosome Encoding）

為了有效處理車輛數與路徑分配，我們採用「群組序列 + 切分標記」的編碼方式：

- / 基因座 (Gene):
- / 正整數 ($0, 1, dots, n-1$): 代表學生群組（Group）的 ID。
- / 切分標記 (Delimiter, $-1$): 代表兩條不同路徑的邊界。


- / 染色體長度: $|G| + (m-1)$，其中 $|G|$ 為群組總數，$m$ 為最大可用公車數。
- / 範例: 若有 5 個群組與 3 輛車，染色體 `[3, 0, -1, 1, 4, -1, 2]` 代表：
   -  / 第一輛車服務: Group 3, 0
   - / 第二輛車服務: Group 1, 4
   - / 第三輛車服務: Group 2



=== 3. 初始化策略：隨機貪婪

為避免隨機初始化產生過多不可行解, 我們採用以下策略:

+ / 隨機最近鄰: 隨機選取起始點，並從最近的 $K$ 個站點中隨機挑選下一個站點，生成具有地理聚集性的群組序列。
+ / 貪婪切分遍歷序列: ，若加入下一個群組會導致路徑違反載重上限（40人）或時間上限（60分鐘），則強制插入切分標記 $-1$。

=== 4. 解碼邏輯與路徑生成

將染色體轉換為具體路徑的過程包含兩個關鍵步驟:

+ / 合併上車事件: 將染色體片段中相同站點的群組合併為單一次上車動作。
+ / 貪婪下車順序: 從最後一個上車點出發, 每次選擇距離當前位置最近的目標學校作為下一個下車點，直到所有群組皆送達。
+ / 合理性檢查: 若解碼出的任何路徑違反容量或時間限制，則標記該個體為不可行（Feasible = False）。

=== 5. 進化算子 (Evolutionary Operators)

==== A. 錦標賽選擇 (Tournament Selection)

每次隨機選取 $k=3$ 個個體，根據其 *Solution Cost* 進行評比，勝出者進入下一代或參與交配。

==== B. 片段交配 (Segment-based Crossover)

不使用傳統的單點或雙點交配，而是針對 VRP 問題設計的特殊算子：

+ 從父母 A 與 B 中各隨機提取一條*完整路徑片段*作為子代的開頭。
+ *Greedy Packing*：將剩餘未被服務的群組，依隨機順序嘗試填入現有路徑，若填不下則開啟新路徑。

==== C. 突變算子 (Mutation)

- / 併車突變: 將中間的切分點 $-1$ 移至末尾，嘗試將兩條路徑合併。
- / 位置交換: 隨機交換染色體中的兩個位置（可能是群組或標記）。
- / 插值突變: 將某一對群組移出原路徑，隨機插入到另一位置。

=== 6. 適應度函數 (Fitness Function)

適應度直接對應總成本函數：


Cost = \text{BusCount} \times f_b + \text{TotalTime} \times \alpha + \text{InVehicleTime} \times \beta + \text{FairnessPenalty} \times \gamma


其中：

- / BusCount: 使用的公車數量。
- / TotalTime: 所有車輛總行駛分鐘。
- / FairnessPenalty: 同一學校不同學生的乘車時間不公平性懲罰。

=== 7. 演算法流程 (Algorithm Pipeline)

```text
Algorithm: GA for SBRP
Input: Schools, Stations, Population Size (1000), Generations (300)
Output: Best Feasible Solution

1. Initialize Population using Randomized Nearest Neighbor.
2. For generation g = 1 to 300:
   a. Decode each chromosome and calculate Cost.
   b. Keep Elite 100 individuals.
   c. While population < 1000:
      i. Select parents via Tournament.
      ii. Apply Segment-based Crossover.
      iii. Apply Mutation (Merge, Swap, or Insertion).
   d. Replace old population with elites and offspring.
   e. Track and log the Best Solution.
3. Return Best Solution.

```

=== 8. 實作建議與注意事項

+ / 菁英保留: 在 SBRP 這種強約束問題中，菁英保留（本實作取前 100 名）對於維持收斂穩定性至關重要。
+ / 多樣性與變異: 由於路徑合併的邏輯強烈依賴 $-1$ 的位置，建議維持較高的突變率（本實作設為 0.6）以跳出區域解。
+ / 地圖驗證: 建議在每 50 代結束後，利用 `plot_routes_on_map` 視覺化最優路徑，觀察路徑是否出現重疊或不合理的跨區行為，以此調整 $K$ 值或權重。