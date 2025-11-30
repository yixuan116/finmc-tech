# MPI Monte Carlo Demo - 逐行解释

## 文件结构概览

这个文件实现了基于 MPI（Message Passing Interface）的分布式内存并行 Monte Carlo 模拟。

---

## 第一部分：导入和常量定义 (Lines 1-46)

### Lines 1-24: 文档字符串
```python
"""
MPI Monte Carlo benchmark demo (HPC, Step 8).
...
"""
```
- **作用**: 说明文件用途和使用方法
- **关键信息**: 
  - 这是一个可选的 MPI 演示，不用于主流程
  - 展示如何将 Monte Carlo 工作负载分配到多个 MPI ranks
  - 通过多进程消息传递实现并行

### Lines 26-31: 导入库
```python
from mpi4py import MPI      # MPI 并行计算库
import numpy as np          # 数值计算
from pathlib import Path    # 路径处理
import csv                  # CSV 文件读写
import sys                  # 系统相关（未直接使用）
import argparse             # 命令行参数解析
```

### Lines 33-39: 默认参数
```python
TOTAL_SIMS = 1000000        # 总模拟次数：100万
DEFAULT_N_STEPS = 12        # 默认时间步数：12个月（1年）
MU = 0.01                   # 年化收益率：1%
SIGMA = 0.40                # 年化波动率：40%
S0 = 100.0                  # 初始价格：$100
STEPS_PER_YEAR = 12         # 每年步数：12（月度）
```

### Lines 41-46: 多时间范围配置
```python
HORIZONS = {
    "1Y": 12,   # 1年 = 12个月
    "3Y": 36,   # 3年 = 36个月
    "5Y": 60,   # 5年 = 60个月
}
```

---

## 第二部分：Monte Carlo 核心函数 (Lines 48-70)

### `run_mc_paths()` - 单个 Rank 的 Monte Carlo 模拟

```python
def run_mc_paths(n_sims, n_steps, mu, sigma, s0, steps_per_year=12):
```
**作用**: 在单个 MPI rank 上运行 Monte Carlo 模拟（向量化实现）

#### Line 49: 函数说明
```python
"""Simple sequential Monte Carlo kernel for a single rank."""
```
- 这是**单个 rank** 的模拟函数，不是全局并行
- 全局并行是通过多个 rank 各自运行这个函数实现的

#### Line 50: 随机数生成器
```python
rng = np.random.default_rng()  # Seed can be rank-dependent if needed
```
- 创建随机数生成器
- **注意**: 注释提到可以设置 rank 相关的 seed（当前未实现）
- 如果所有 rank 用相同 seed，会产生相同随机数（通常不希望这样）

#### Line 51: 计算每步波动率
```python
sigma_step = sigma / np.sqrt(steps_per_year)
```
- **公式**: `σ_step = σ_annual / √(steps_per_year)`
- **例子**: 年化波动率 40%，月度步数 12 → `σ_step = 0.40 / √12 ≈ 0.1155`
- 这是将年化波动率转换为每步波动率的标准方法

#### Line 54: 预分配终端价格数组
```python
terminals = np.zeros(n_sims)
```
- 预分配数组存储每条路径的最终价格
- **注意**: 这个变量定义了但未使用（实际返回的是 `paths[:, -1]`）

#### Lines 58-59: 初始化价格路径矩阵
```python
paths = np.zeros((n_sims, n_steps + 1))  # 形状: (n_sims, n_steps+1)
paths[:, 0] = s0                         # 所有路径的初始价格 = S0
```
- `paths[i, t]` = 第 i 条路径在时间 t 的价格
- 第一列（索引 0）全部设为初始价格 `S0 = 100.0`

#### Line 61: 生成随机数矩阵
```python
Z = rng.standard_normal((n_sims, n_steps))
```
- 生成标准正态分布随机数矩阵
- 形状: `(n_sims, n_steps)`
- 每个元素 `Z[i, t]` 是路径 i 在时间步 t 的随机冲击

#### Line 64: 计算每步收益率
```python
rets = mu + sigma_step * Z
```
- **GBM 模型**: `r_t = μ + σ_step × Z_t`
- `mu` 是每步的期望收益率（年化 1% / 12 ≈ 0.000833）
- `sigma_step * Z` 是随机波动项

#### Line 65: 防止负价格
```python
rets = np.clip(rets, -0.99, None)
```
- 限制收益率不低于 -99%
- **原因**: 如果 `1 + rets < 0`，价格会变成负数（不符合金融逻辑）
- 确保 `1 + rets ≥ 0.01`，即价格最多下跌 99%

#### Line 68: 计算累积价格路径
```python
paths[:, 1:] = s0 * np.cumprod(1.0 + rets, axis=1)
```
- **关键公式**: `S_{t+1} = S_t × (1 + r_t)`
- `np.cumprod(1.0 + rets, axis=1)`: 沿时间轴（axis=1）累积乘积
- **例子**: 
  - `1 + rets[0, 0] = 1.01` → `paths[0, 1] = 100 × 1.01 = 101`
  - `1 + rets[0, 1] = 0.98` → `paths[0, 2] = 101 × 0.98 = 98.98`
- 从 `S0` 开始，所以 `paths[:, 1:]` 从索引 1 开始

#### Line 70: 返回最终价格
```python
return paths[:, -1]
```
- 返回所有路径的最终价格（最后一列）
- 形状: `(n_sims,)`，每个元素是一条路径的终端价格

---

## 第三部分：MPI 并行基准测试函数 (Lines 72-131)

### `run_single_mpi_benchmark()` - 单次 MPI 基准测试

```python
def run_single_mpi_benchmark(n_steps: int, total_sims: int, output_dir: Path) -> float:
```
**作用**: 运行一次完整的 MPI 并行 Monte Carlo 基准测试

#### Lines 92-94: 获取 MPI 通信器和进程信息
```python
comm = MPI.COMM_WORLD      # 全局通信器（所有进程）
rank = comm.Get_rank()     # 当前进程的 rank（0, 1, 2, 3...）
size = comm.Get_size()     # 总进程数（例如：4）
```
- **`MPI.COMM_WORLD`**: 包含所有 MPI 进程的通信器
- **`rank`**: 当前进程的唯一标识符（0 到 size-1）
- **`size`**: 总进程数（由 `mpirun -n 4` 指定）

#### Lines 96-100: 计算每个 rank 的工作负载
```python
local_n_sims = total_sims // size        # 每个 rank 的基础工作量
if rank == size - 1:                     # 最后一个 rank
    local_n_sims += total_sims % size     # 处理余数
```
- **例子**: `total_sims = 1,000,000`, `size = 4`
  - `local_n_sims = 1,000,000 // 4 = 250,000`（前 3 个 rank）
  - `rank 3` (最后一个): `250,000 + (1,000,000 % 4) = 250,000 + 0 = 250,000`
- **如果余数不为 0**: 例如 `1,000,001` 个模拟
  - Rank 0-2: 各 250,000
  - Rank 3: 250,000 + 1 = 250,001

#### Lines 102-104: 同步并开始计时
```python
comm.Barrier()           # 所有进程在此等待，确保同步
t0 = MPI.Wtime()         # 获取当前时间（高精度）
```
- **`Barrier()`**: 同步点，所有进程必须到达这里才能继续
- **作用**: 确保所有进程同时开始计算，公平计时
- **`MPI.Wtime()`**: MPI 提供的高精度计时函数（类似 `time.time()`）

#### Line 107: 并行执行 Monte Carlo 模拟
```python
local_terminals = run_mc_paths(local_n_sims, n_steps, MU, SIGMA, S0, STEPS_PER_YEAR)
```
- **每个 rank 独立运行**: 在自己的进程空间内计算
- **无通信**: 此时各 rank 之间不交换数据
- **结果**: 每个 rank 得到 `local_n_sims` 条路径的终端价格

#### Lines 109-113: 等待所有进程完成并结束计时
```python
comm.Barrier()           # 等待所有进程完成
t1 = MPI.Wtime()         # 结束时间
local_time = t1 - t0     # 当前 rank 的运行时间
```
- **第二个 Barrier**: 确保所有进程都完成计算
- **`local_time`**: 每个 rank 的本地运行时间（可能不同）

#### Line 116: 收集最大运行时间
```python
max_time = comm.reduce(local_time, op=MPI.MAX, root=0)
```
- **`comm.reduce()`**: 归约操作，将所有 rank 的数据合并
- **`op=MPI.MAX`**: 取最大值（找出最慢的 rank）
- **`root=0`**: 结果只返回给 rank 0
- **为什么用 MAX**: 并行程序的运行时间 = 最慢进程的时间（木桶效应）

#### Lines 118-129: Rank 0 写入结果到 CSV
```python
if rank == 0:  # 只有 master rank 写入文件
    csv_path = output_dir / "hpc_benchmark_mpi.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, mode='a', newline='') as f:  # 追加模式
        writer = csv.writer(f)
        if not file_exists:
            # 第一次写入：创建表头
            writer.writerow(["backend", "mode", "n_sims", "n_steps", "n_ranks", "time_sec"])
        
        # 写入一行结果
        writer.writerow(["mpi_python", "mpi_parallel", total_sims, n_steps, size, max_time])
```
- **只有 rank 0 写入**: 避免多个进程同时写文件造成冲突
- **追加模式 (`'a'`)**: 多次运行会追加新行，不会覆盖
- **CSV 格式**: 
  - `backend`: "mpi_python"
  - `mode`: "mpi_parallel"
  - `n_sims`: 总模拟数（1,000,000）
  - `n_steps`: 时间步数（12, 36, 60...）
  - `n_ranks`: 进程数（4）
  - `time_sec`: 最大运行时间（秒）

#### Line 131: 返回值
```python
return max_time if rank == 0 else 0.0
```
- **Rank 0**: 返回 `max_time`（用于打印和后续处理）
- **其他 rank**: 返回 `0.0`（因为只有 rank 0 知道 `max_time`）

---

## 第四部分：主函数 (Lines 133-218)

### `main()` - 程序入口

#### Lines 134-150: 定义命令行参数
```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--steps", type=int, default=DEFAULT_N_STEPS, ...)
parser.add_argument("--sims", type=int, default=TOTAL_SIMS, ...)
parser.add_argument("--multi-horizon", action="store_true", ...)
```
- **`--steps`**: 指定时间步数（默认 12）
- **`--sims`**: 指定总模拟数（默认 1,000,000）
- **`--multi-horizon`**: 标志位，如果提供则运行所有时间范围

#### Lines 152-155: 获取 MPI 信息
```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```
- 再次获取 MPI 信息（在 main 函数中）

#### Lines 157-170: Rank 0 解析参数并广播
```python
if rank == 0:
    args = parser.parse_args()      # 只有 rank 0 解析命令行
    n_steps = args.steps
    total_sims = args.sims
    multi_horizon = args.multi_horizon
else:
    n_steps = None                   # 其他 rank 先设为 None
    total_sims = None
    multi_horizon = None

# 广播参数到所有 rank
n_steps = comm.bcast(n_steps, root=0)
total_sims = comm.bcast(total_sims, root=0)
multi_horizon = comm.bcast(multi_horizon, root=0)
```
- **为什么只有 rank 0 解析**: 
  - 避免多个进程同时读取命令行造成混乱
  - 通常只有 rank 0 有标准输入/输出
- **`comm.bcast()`**: 广播操作
  - `root=0`: 从 rank 0 发送数据
  - 所有其他 rank 接收相同的数据
  - **结果**: 所有 rank 都有相同的 `n_steps`, `total_sims`, `multi_horizon`

#### Line 172-173: 创建输出目录
```python
output_dir = Path("results/step8")
output_dir.mkdir(parents=True, exist_ok=True)
```
- 所有 rank 都执行（虽然只有 rank 0 会写文件）
- `parents=True`: 如果父目录不存在则创建
- `exist_ok=True`: 如果目录已存在不报错

#### Lines 175-193: 多时间范围模式
```python
if multi_horizon:
    if rank == 0:
        print(f"Starting Multi-Horizon MPI Monte Carlo Demo with {size} ranks...")
        # ... 打印信息
    
    for horizon_label, horizon_steps in HORIZONS.items():  # 遍历 1Y, 3Y, 5Y
        if rank == 0:
            print(f"--- Running {horizon_label} ({horizon_steps} steps) ---")
        
        max_time = run_single_mpi_benchmark(horizon_steps, total_sims, output_dir)
        
        if rank == 0:
            print(f"✓ {horizon_label} finished in {max_time:.4f} seconds\n")
```
- **逻辑**: 循环运行 1Y、3Y、5Y 三个时间范围
- **每个时间范围**: 调用一次 `run_single_mpi_benchmark()`
- **结果**: CSV 文件中会有 3 行（每行一个时间范围）

#### Lines 194-215: 单时间范围模式（默认）
```python
else:
    if rank == 0:
        # 根据 n_steps 确定标签
        horizon_label = f"{n_steps}M"
        if n_steps == 12:
            horizon_label = "1Y"
        elif n_steps == 36:
            horizon_label = "3Y"
        # ...
        print(f"Starting MPI Monte Carlo Demo with {size} ranks...")
    
    max_time = run_single_mpi_benchmark(n_steps, total_sims, output_dir)
    
    if rank == 0:
        print(f"✓ MPI Parallel finished in {max_time:.4f} seconds")
```
- **默认行为**: 只运行一次，使用 `--steps` 指定的时间范围
- **如果没有 `--multi-horizon`**: 走这个分支

#### Lines 217-218: 程序入口
```python
if __name__ == "__main__":
    main()
```
- 标准 Python 入口点
- 当直接运行脚本时执行 `main()`

---

## 关键 MPI 概念总结

### 1. **通信器 (Communicator)**
- `MPI.COMM_WORLD`: 包含所有进程的全局通信器

### 2. **Rank 和 Size**
- `rank`: 进程标识符（0 到 size-1）
- `size`: 总进程数

### 3. **同步操作**
- `Barrier()`: 所有进程等待，直到全部到达同步点

### 4. **通信操作**
- `bcast()`: 广播（一个进程发送，所有进程接收）
- `reduce()`: 归约（所有进程发送，一个进程接收并合并）

### 5. **工作分配策略**
- **数据并行**: 将 1,000,000 条路径分配给 4 个进程
- **负载均衡**: 每个进程处理大致相同数量的路径
- **余数处理**: 最后一个 rank 处理余数

---

## 执行流程示例（4 个进程，1,000,000 次模拟）

```
时间轴:
t0: 所有进程启动
    Rank 0: 解析命令行参数
    Rank 1-3: 等待

t1: Rank 0 广播参数
    Rank 0-3: 都收到 n_steps=12, total_sims=1,000,000

t2: 计算工作分配
    Rank 0: local_n_sims = 250,000
    Rank 1: local_n_sims = 250,000
    Rank 2: local_n_sims = 250,000
    Rank 3: local_n_sims = 250,000

t3: Barrier() - 所有进程同步

t4: 开始计时 (MPI.Wtime())
    Rank 0-3: 各自运行 run_mc_paths(250,000, 12, ...)
    [并行执行，无通信]

t5: 完成计算
    Rank 0: local_time = 0.25 秒
    Rank 1: local_time = 0.24 秒
    Rank 2: local_time = 0.26 秒  ← 最慢
    Rank 3: local_time = 0.25 秒

t6: Barrier() - 等待所有进程完成

t7: Reduce(MAX) - 收集最大时间
    Rank 0: 收到 max_time = 0.26 秒
    Rank 1-3: 不接收结果

t8: Rank 0 写入 CSV
    "mpi_python,mpi_parallel,1000000,12,4,0.26"

t9: 所有进程结束
```

---

## 与 Numba/OpenMP 的对比

| 特性 | MPI | Numba/OpenMP |
|------|-----|--------------|
| **内存模型** | 分布式（每个进程独立内存） | 共享内存（所有线程共享） |
| **通信** | 显式消息传递 | 隐式共享变量 |
| **扩展性** | 可跨多台机器 | 单机多核 |
| **开销** | 较高（进程间通信） | 较低（线程间共享） |
| **适用场景** | 大规模集群、多节点 | 单机多核、细粒度并行 |

---

## 常见问题

### Q1: 为什么只有 rank 0 写文件？
**A**: 避免多个进程同时写文件造成数据竞争和文件损坏。

### Q2: 为什么用 `reduce(MAX)` 而不是 `reduce(SUM)`？
**A**: 并行程序的运行时间 = 最慢进程的时间。如果 rank 2 需要 0.26 秒，整个程序就需要 0.26 秒。

### Q3: 如果 `total_sims` 不能被 `size` 整除怎么办？
**A**: 余数分配给最后一个 rank（`rank == size - 1`），确保所有模拟都被处理。

### Q4: 为什么需要两个 `Barrier()`？
**A**: 
- 第一个：确保所有进程同时开始（公平计时）
- 第二个：确保所有进程都完成（才能计算 max_time）

### Q5: 随机数生成器会冲突吗？
**A**: 当前代码中，所有 rank 使用相同的默认 seed，可能产生相同的随机数。理想情况下应该：
```python
rng = np.random.default_rng(seed=42 + rank)  # 每个 rank 不同的 seed
```

