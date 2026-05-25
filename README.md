# 校車路徑問題 (SBRP) 求解器

本儲存庫提供了多種演算法來解決校車路徑問題 (SBRP)。目標是為校車找到最佳或接近最佳的路徑，以便從指定站點接送學生並將其送到各自的學校，在遵守約束條件（如巴士容量、最大行駛時間）的同時，最小化成本（如巴士數量、總行駛時間）。

## 已實現的演算法

1.  **精確求解器 (`test.py`)**: 測試用可忽略
2.  **遺傳演算法 (GA) 求解器 (`ga_solver.py`)**: 一種啟發式方法，使用交配和突變等遺傳算子使解的種群跨代進化。
3.  **貪婪求解器 (`greedy_solver.py`)**: 一種簡單的啟發式方法，透過做出局部最佳選擇來增量構建路徑。常用於為其他演算法生成初始解。
4.  **大鄰域搜索 (LNS) 求解器 (`lns_solver.py`)**: 一種迭代啟發式方法，重複破壞當前解的一部分並使用啟發式方法修復它，旨在跳出局部最佳解。
5.  **Dantzig-Wolfe 分解求解器 (`dantzig-wolfe_solver.py`)**: 一種精確方法，使用列生成 (Column Generation) 解決大規模線性規劃，將問題分解為主問題和定價問題。

## 安裝方式

1.  **複製儲存庫：**
    ```bash
    git clone https://github.com/your-username/SBRP_algorithmns.git
    cd SBRP_algorithmns
    ```

2.  **安裝 Python 依賴項：**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Gurobi 安裝（精確與 Dantzig-Wolfe 求解器所需）：**
    `gurobipy` 是 Gurobi Optimizer 的接口，這是一款商業軟體。您需要：
    *   從 Gurobi 官網下載並安裝 Gurobi Optimizer。
    *   獲取有效的 Gurobi 授權（學術授權通常是免費的）。
    *   按照 Gurobi 的說明設定環境變數和授權。

## 數據格式

求解器需要兩個 CSV 檔案作為輸入：

*   `stops-*.csv`: 定義學生取貨站點及其需求。
    *   欄位：`idx`（站點 ID）、`name`（站點名稱）、`orig_idx`（用於行駛時間矩陣查找的原始索引）、`school_idx`（目標學校 ID）、`count`（該校學生人數）。
*   `time-*.csv`: 一個方陣，代表所有相關原始索引（站點和學校）之間的行駛時間。
    *   `(i, j)` 條目代表從原始索引 `i` 到原始索引 `j` 的行駛時間。

需自行建立`./data/` 目錄並放入數據檔案。

## 如何執行求解器

每個求解器都可以直接從其 Python 檔案執行。您可以修改各檔案內的 `if __name__ == "__main__":` 區塊來更改輸入數據或參數。

**用法範例：**

*   **精確求解器：**
    ```bash
    python test.py
    ```
*   **遺傳演算法求解器：**
    ```bash
    python ga_solver.py
    ```
*   **貪婪求解器：**
    ```bash
    python greedy_solver.py
    ```
*   **大鄰域搜索求解器：**
    ```bash
    python lns_solver.py
    ```
*   **蟻群最佳化求解器：**
    ```bash
    python aco_solver.py
    ```
*   **Dantzig-Wolfe 求解器：**
    ```bash
    python dantzig-wolfe_solver.py
    ```

## 輸出結果

所有求解器都將在終端機中印出格式解摘要，包括總成本、巴士數量、總車內時間以及每條路徑的詳情