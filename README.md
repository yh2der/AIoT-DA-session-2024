# AIoT-DA-Session-2024
This repo will consist of homework of "Artificial Intelligence of Things"

## 作業連結
- [Hw1](./Hw1)
  - [Hw1-1](./Hw1/Hw1-1)
  - [Hw1-1_prompt](./Hw1/Hw1-1/PROMPT.md)
  - [Hw1-2](./Hw1/Hw1-2)
  - [Hw1-2_prompt](./Hw1/Hw1-2/PROMPT.md)
- [Hw2](./Hw2)
  - [Result](./Hw2/RESULT.md)
- [Hw3](./Hw3)
  - [Result](./Hw3/RESULT.md)

## Anaconda 安裝和 Conda 虛擬環境使用指南

## 1. 下載和安裝 Anaconda

### 1.1 下載 Anaconda
1. 訪問 [Anaconda 官方網站](https://www.anaconda.com/products/distribution)
2. 選擇適合你操作系統的版本（Windows、macOS 或 Linux）

### 1.2 安裝 Anaconda
- **Windows**:
  1. 運行下載的 .exe 文件
  2. 按照安裝向導的指示進行操作

- **macOS**:
  1. 打開下載的 .pkg 文件
  2. 按照指示安裝

- **Linux**:
  1. 打開終端
  2. 運行以下命令（替換 `path_to_file` 為實際文件路徑）:
     ```
     bash ~/path_to_file/Anaconda3-2023.XX-Linux-x86_64.sh
     ```
  3. 按照提示完成安裝

### 1.3 驗證安裝
安裝完成後，打開新的終端窗口，運行:
```
conda --version
```
如果顯示 conda 的版本號，說明安裝成功。

### 1.4 初始化 conda
在某些系統中，你可能需要初始化 conda。在終端中運行:
```
conda init
```
然後重新啟動終端。

## 2. 使用 Conda 虛擬環境

### 2.1 創建新的虛擬環境
```
conda create --name myenv python=3.8
```
這會創建一個名為 "myenv" 的新環境，使用 Python 3.8。

### 2.2 激活虛擬環境
```
conda activate myenv
```

### 2.3 在虛擬環境中安裝包
```
conda install package_name
```

### 2.4 查看已安裝的包
```
conda list
```

### 2.5 退出虛擬環境
```
conda deactivate
```

### 2.6 刪除虛擬環境
```
conda remove --name myenv --all
```

### 2.7 列出所有虛擬環境
```
conda env list
```

## 3. 提示和技巧

- 創建環境時指定 Python 版本很重要，因為不同的專案可能需要不同的 Python 版本。
- 在激活環境後安裝包，確保包只安裝在特定環境中，不影響其他環境。
