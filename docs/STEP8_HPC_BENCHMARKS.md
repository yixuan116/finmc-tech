# Step 8 HPC Multi-Horizon Benchmarks Guide

本指南说明如何运行 Step 8 的 HPC 基准测试，覆盖 1Y/3Y/5Y 时间范围。

## 快速开始

### 方法 1: 使用自动化脚本（推荐）

```bash
# 运行所有基准测试
./run_step8_hpc_benchmarks.sh
```

这个脚本会自动：
1. 运行 Python NumPy vs Numba 基准测试（1Y/3Y/5Y）
2. 运行 MPI 基准测试（如果可用）
3. 运行 OpenMP C 基准测试（如果已编译）
4. 生成汇总表

### 方法 2: 手动运行各个步骤

#### Step 1: Python NumPy vs Numba 基准测试

```bash
# 运行多时间范围基准测试（1Y, 3Y, 5Y）
python -m finmc_tech.simulation.scenario_mc --hpc-multi-horizon --n 50000
```

**输出文件**: `results/step8/hpc_benchmark_paths_1y_3y_5y.csv`

**说明**:
- `--n 50000`: 每个时间范围的模拟次数（可根据需要调整）
- 默认运行 1Y (12步), 3Y (36步), 5Y (60步)
- 比较 NumPy baseline 和 Numba parallel 的性能

#### Step 2: MPI 基准测试（可选）

**前提条件**: 需要安装 `mpi4py` 和 MPI 运行时

```bash
# 运行多时间范围 MPI 基准测试
mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --multi-horizon
```

**输出文件**: `results/step8/hpc_benchmark_mpi.csv`

**说明**:
- `-n 4`: MPI 进程数（可根据系统调整）
- `--multi-horizon`: 运行所有时间范围（1Y, 3Y, 5Y）
- 如果不提供 `--multi-horizon`，默认只运行 12 步（1Y）

**单次运行示例**:
```bash
# 只运行 3Y (36步)
mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --steps 36
```

#### Step 3: OpenMP C 基准测试（可选）

**前提条件**: 需要编译 C 代码

```bash
# 编译（Linux/macOS）
gcc -O3 -fopenmp finmc_tech/hpc_demos/openmp_mc_demo.c -o openmp_mc_demo

# macOS 可能需要：
gcc -O3 -Xpreprocessor -fopenmp -lomp finmc_tech/hpc_demos/openmp_mc_demo.c -o openmp_mc_demo

# 运行（默认运行所有时间范围）
./openmp_mc_demo
```

**输出文件**: `results/step8/hpc_benchmark_openmp.csv`

**说明**:
- 默认行为：自动运行所有时间范围（1Y, 3Y, 5Y）
- 如果指定 `--steps`，只运行该时间范围

**单次运行示例**:
```bash
# 只运行 5Y (60步)
./openmp_mc_demo --steps 60
```

#### Step 4: 生成汇总表

```bash
# 生成汇总表（需要先运行前面的基准测试）
python -m finmc_tech.simulation.scenario_mc --hpc-summary
```

**输出文件**: `results/step8/hpc_benchmark_summary_1y_3y_5y.csv`

**说明**:
- 汇总所有基准测试结果
- 计算每个时间范围的最佳并行时间和加速比
- 需要先运行至少 Python 基准测试

## 输出文件说明

所有结果保存在 `results/step8/` 目录：

### 1. `hpc_benchmark_paths_1y_3y_5y.csv`
Python NumPy vs Numba 基准测试结果

**列**:
- `backend`: "baseline_numpy" 或 "numba_parallel"
- `time_sec`: 运行时间（秒）
- `speedup_vs_baseline`: 相对于 baseline 的加速比
- `horizon`: 时间范围标签（"1Y", "3Y", "5Y"）
- `n_steps`: 步数（12, 36, 60）

### 2. `hpc_benchmark_mpi.csv`
MPI 基准测试结果

**列**:
- `backend`: "mpi_python"
- `mode`: "mpi_parallel"
- `n_sims`: 模拟次数
- `n_steps`: 步数（12, 36, 60）
- `n_ranks`: MPI 进程数
- `time_sec`: 运行时间（秒）

### 3. `hpc_benchmark_openmp.csv`
OpenMP C 基准测试结果

**列**:
- `backend`: "openmp_c"
- `mode`: "sequential" 或 "openmp_parallel"
- `n_sims`: 模拟次数
- `n_steps`: 步数（12, 36, 60）
- `n_ranks`: 线程数（通常为 1，因为 OpenMP 自动管理）
- `time_sec`: 运行时间（秒）

### 4. `hpc_benchmark_summary_1y_3y_5y.csv`
汇总表

**列**:
- `horizon`: 时间范围（"1Y", "3Y", "5Y"）
- `n_steps`: 步数
- `baseline_time`: NumPy baseline 时间
- `best_parallel_time`: 最佳并行时间（Numba/MPI/OpenMP 中最快的）
- `best_backend`: 最佳并行后端
- `speedup`: 加速比（baseline_time / best_parallel_time）

## 运行顺序建议

### 最小运行（只运行 Python 基准测试）

```bash
# 1. Python 基准测试
python -m finmc_tech.simulation.scenario_mc --hpc-multi-horizon --n 50000

# 2. 生成汇总（只有 Python 结果）
python -m finmc_tech.simulation.scenario_mc --hpc-summary
```

### 完整运行（所有基准测试）

```bash
# 使用自动化脚本
./run_step8_hpc_benchmarks.sh
```

或手动运行：

```bash
# 1. Python 基准测试
python -m finmc_tech.simulation.scenario_mc --hpc-multi-horizon --n 50000

# 2. MPI 基准测试（如果可用）
mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --multi-horizon

# 3. OpenMP 基准测试（如果已编译）
./openmp_mc_demo

# 4. 生成汇总
python -m finmc_tech.simulation.scenario_mc --hpc-summary
```

## 性能调优建议

1. **模拟次数 (`--n`)**: 
   - 默认 50,000 次模拟通常足够
   - 可以增加到 100,000 或更多以获得更准确的结果
   - 减少到 10,000 可以快速测试

2. **MPI 进程数 (`-n`)**: 
   - 根据 CPU 核心数调整
   - 通常使用 4-8 个进程

3. **OpenMP 线程数**: 
   - 由环境变量 `OMP_NUM_THREADS` 控制
   - 例如: `export OMP_NUM_THREADS=4`

## 故障排除

### MPI 未找到
```bash
# 检查是否安装
pip install mpi4py

# 检查 MPI 运行时
which mpirun
```

### OpenMP 编译错误
```bash
# macOS 可能需要安装 libomp
brew install libomp

# 然后使用：
gcc -O3 -Xpreprocessor -fopenmp -lomp finmc_tech/hpc_demos/openmp_mc_demo.c -o openmp_mc_demo
```

### 结果文件不存在
确保先运行相应的基准测试，汇总功能需要至少 Python 基准测试的结果。

## 下一步

运行完基准测试后，可以：
1. 查看汇总表了解性能对比
2. 分析不同时间范围的性能特征
3. 根据结果选择最适合的并行化方案

