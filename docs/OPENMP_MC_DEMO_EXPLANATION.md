# OpenMP Monte Carlo Demo - 逐行解释

## 文件结构概览

这个文件实现了基于 OpenMP（Open Multi-Processing）的共享内存并行 Monte Carlo 模拟，使用 C 语言编写。

**关键特点**:
- **共享内存模型**: 所有线程共享同一内存空间
- **编译器指令**: 使用 `#pragma omp parallel for` 实现并行
- **对比基准**: 同时运行顺序版本和并行版本，计算加速比

---

## 第一部分：头文件和常量定义 (Lines 1-48)

### Lines 1-24: 文件头注释
```c
/*
 * OpenMP Monte Carlo Benchmark Demo (HPC, Step 8).
 * ...
 */
```
- **作用**: 说明文件用途、编译方法和运行示例
- **关键信息**:
  - 单线程循环对应 NumPy 向量化 baseline
  - OpenMP 并行循环对应 Numba `@njit(parallel=True)` + `prange`
  - macOS 编译需要特殊参数（`-Xpreprocessor -fopenmp -lomp`）

### Lines 26-34: 标准库头文件
```c
#include <stdio.h>    // 标准输入输出（printf, fprintf）
#include <stdlib.h>   // 标准库（malloc, free, rand, atoi）
#include <math.h>     // 数学函数（sqrt, log, cos, M_PI）
#include <time.h>     // 时间函数（未直接使用，但通常用于计时）
#include <omp.h>      // OpenMP 库（omp_get_wtime, #pragma omp）
#include <string.h>   // 字符串函数（strcmp, strncpy）
#include <sys/stat.h> // 文件状态（stat）
#include <sys/types.h>// 系统类型定义
#include <errno.h>    // 错误码（未直接使用）
```
- **`omp.h`**: OpenMP 的核心头文件，提供并行编程接口

### Lines 36-42: 默认参数（宏定义）
```c
#define DEFAULT_N_SIMS 1000000    // 默认模拟次数：100万
#define DEFAULT_N_STEPS 12        // 默认时间步数：12个月（1年）
#define MU 0.01                   // 年化收益率：1%
#define SIGMA 0.40                // 年化波动率：40%
#define S0 100.0                 // 初始价格：$100
#define STEPS_PER_YEAR 12         // 每年步数：12（月度）
```
- **宏定义**: 编译时替换，无类型检查
- **与 MPI 版本一致**: 使用相同的参数值，便于对比

### Lines 44-47: 多时间范围配置
```c
#define NUM_HORIZONS 3
static const int HORIZON_STEPS[] = {12, 36, 60};
static const char* HORIZON_LABELS[] = {"1Y", "3Y", "5Y"};
```
- **`static const`**: 文件内静态常量，不可修改
- **数组**: 存储 1Y、3Y、5Y 三个时间范围的步数和标签

---

## 第二部分：辅助函数 (Lines 49-104)

### `rand_normal()` - Box-Muller 变换生成正态分布随机数

#### Lines 49-56: Box-Muller 算法
```c
double rand_normal() {
    double u1 = ((double) rand() / RAND_MAX);
    double u2 = ((double) rand() / RAND_MAX);
    // Avoid log(0)
    if (u1 < 1e-9) u1 = 1e-9;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
```

**作用**: 将均匀分布随机数转换为标准正态分布

**算法原理** (Box-Muller 变换):
1. 生成两个独立的均匀分布随机数 `u1`, `u2` ∈ [0, 1)
2. 如果 `u1` 太小（接近 0），`log(u1)` 会很大，设为最小值避免数值问题
3. 公式: `Z = √(-2·ln(u1)) · cos(2π·u2)`
   - 结果 `Z` 服从标准正态分布 N(0, 1)

**为什么需要这个函数**:
- C 标准库的 `rand()` 只生成均匀分布
- Monte Carlo 需要正态分布随机数（模拟价格波动）

**注意**: 这个函数**不是线程安全的**（使用全局 `rand()`），在并行版本中会使用 `rand_r()` 替代。

---

### `append_csv_row()` - 写入 CSV 行

#### Lines 58-82: CSV 文件写入函数
```c
void append_csv_row(const char *filename, const char *backend, const char *mode, 
                   int n_sims, int n_steps, int n_ranks, double time_sec) {
    FILE *fp = fopen(filename, "a");  // 追加模式打开
    if (!fp) {
        // 如果追加失败，尝试创建新文件
        fp = fopen(filename, "w");
        if (!fp) {
            fprintf(stderr, "Error opening file %s\n", filename);
            return;
        }
        // 新文件：写入表头
        fprintf(fp, "backend,mode,n_sims,n_steps,n_ranks,time_sec\n");
    } else {
        // 检查文件是否为空
        fseek(fp, 0, SEEK_END);  // 移动到文件末尾
        long size = ftell(fp);    // 获取文件大小
        if (size == 0) {
            // 空文件：写入表头
            fprintf(fp, "backend,mode,n_sims,n_steps,n_ranks,time_sec\n");
        }
    }
    
    // 写入数据行
    fprintf(fp, "%s,%s,%d,%d,%d,%.6f\n", backend, mode, n_sims, n_steps, n_ranks, time_sec);
    fclose(fp);
}
```

**逻辑流程**:
1. **尝试追加模式 (`"a"`)**: 如果文件存在，追加内容
2. **如果失败**: 尝试创建模式 (`"w"`)
3. **检查文件大小**: 如果文件为空，写入 CSV 表头
4. **写入数据行**: 格式化为 CSV 格式

**CSV 格式**:
```
backend,mode,n_sims,n_steps,n_ranks,time_sec
openmp_c,sequential,1000000,12,1,0.721792
openmp_c,openmp_parallel,1000000,12,1,0.226030
```

**注意**: `n_ranks` 在 OpenMP 中始终为 1（因为 OpenMP 使用线程，不是进程）。

---

### `parse_args()` - 命令行参数解析

#### Lines 84-104: 参数解析函数
```c
void parse_args(int argc, char *argv[], int *n_sims, int *n_steps) {
    *n_sims = DEFAULT_N_SIMS;   // 默认值：100万
    *n_steps = DEFAULT_N_STEPS;  // 默认值：12
    
    for (int i = 1; i < argc; i++) {  // 跳过 argv[0]（程序名）
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *n_steps = atoi(argv[i + 1]);  // 转换为整数
            i++;  // 跳过下一个参数（已处理）
        } else if (strcmp(argv[i], "--sims") == 0 && i + 1 < argc) {
            *n_sims = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--steps N] [--sims N]\n", argv[0]);
            // ... 打印帮助信息
            exit(0);
        }
    }
}
```

**支持的参数**:
- `--steps N`: 指定时间步数（默认 12）
- `--sims N`: 指定模拟次数（默认 1,000,000）
- `--help` 或 `-h`: 显示帮助信息

**例子**:
```bash
./openmp_mc_demo --steps 36 --sims 500000
# n_steps = 36, n_sims = 500000
```

---

## 第三部分：主函数 (Lines 106-246)

### `main()` - 程序入口

#### Lines 107-108: 解析命令行参数
```c
int n_sims, n_steps_arg;
parse_args(argc, argv, &n_sims, &n_steps_arg);
```
- 获取用户指定的模拟次数和时间步数

#### Lines 110-117: 检查是否提供了 `--steps` 参数
```c
int steps_provided = 0;
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--steps") == 0) {
        steps_provided = 1;
        break;
    }
}
```
- **目的**: 区分"用户指定了时间步数"和"使用默认值"
- **如果 `steps_provided == 1`**: 只运行指定的时间范围
- **如果 `steps_provided == 0`**: 运行所有时间范围（1Y, 3Y, 5Y）

#### Lines 119-125: 设置输出路径和检查目录
```c
const char *csv_path = "results/step8/hpc_benchmark_openmp.csv";

struct stat st = {0};
if (stat("results/step8", &st) == -1) {
    printf("Warning: results/step8 directory may not exist.\n");
}
```
- **`stat()`**: 检查目录是否存在
- **注意**: 这里只检查，不创建目录（依赖用户提前创建）

#### Lines 127-145: 确定要运行的时间范围
```c
int horizons_to_run[NUM_HORIZONS];
int num_to_run = 0;

if (steps_provided) {
    // 用户指定了 --steps: 只运行指定的时间范围
    horizons_to_run[0] = n_steps_arg;
    num_to_run = 1;
    printf("Starting OpenMP Monte Carlo Demo...\n");
    printf("Sims: %d, Steps: %d\n", n_sims, n_steps_arg);
} else {
    // 没有指定 --steps: 运行所有时间范围
    for (int i = 0; i < NUM_HORIZONS; i++) {
        horizons_to_run[i] = HORIZON_STEPS[i];  // {12, 36, 60}
    }
    num_to_run = NUM_HORIZONS;  // 3
    printf("Starting Multi-Horizon OpenMP Monte Carlo Demo...\n");
    printf("Sims: %d\n", n_sims);
}
```

**两种模式**:
1. **单时间范围模式**: `./openmp_mc_demo --steps 36` → 只运行 3Y
2. **多时间范围模式**: `./openmp_mc_demo` → 运行 1Y, 3Y, 5Y

#### Lines 147-149: 预分配内存
```c
double *terminals_seq = malloc(n_sims * sizeof(double));  // 顺序版本结果
double *terminals_par = malloc(n_sims * sizeof(double));  // 并行版本结果
```
- **`malloc()`**: 动态分配内存
- **大小**: `n_sims × sizeof(double)` 字节
- **重用**: 所有时间范围共享同一块内存（避免重复分配）

#### Lines 151-239: 循环运行每个时间范围
```c
for (int h = 0; h < num_to_run; ++h) {
    int n_steps = horizons_to_run[h];
    // ... 运行基准测试 ...
}
```

#### Lines 153-169: 查找时间范围标签
```c
const char *hz_label = NULL;

// 在 HORIZON_LABELS 中查找对应的标签
for (int i = 0; i < NUM_HORIZONS; i++) {
    if (HORIZON_STEPS[i] == n_steps) {
        hz_label = HORIZON_LABELS[i];  // "1Y", "3Y", 或 "5Y"
        break;
    }
}

if (hz_label == NULL) {
    // 如果找不到（用户指定了自定义步数），生成标签
    char label_buf[32];
    snprintf(label_buf, sizeof(label_buf), "%dM", n_steps);
    hz_label = label_buf;  // 例如 "120M"
}
```
- **目的**: 为输出显示生成友好的标签（"1Y" 而不是 "12"）

---

### 顺序版本基准测试 (Lines 173-195)

#### Lines 174-175: 初始化
```c
printf("\n1. Running Sequential Baseline...\n");
srand(42);  // 设置随机数种子（可重现性）
```

#### Line 177: 开始计时
```c
double start_seq = omp_get_wtime();
```
- **`omp_get_wtime()`**: OpenMP 提供的高精度计时函数
- **返回**: 从某个固定时间点开始的秒数（类似 `time.time()`）

#### Lines 179-191: 顺序 Monte Carlo 循环
```c
// [HPC-OpenMP] Baseline: single-threaded loop over Monte Carlo paths.
for (int i = 0; i < n_sims; ++i) {
    double s_t = S0;  // 初始价格
    double sigma_step = SIGMA / sqrt((double)STEPS_PER_YEAR);  // 每步波动率
    
    for (int t = 0; t < n_steps; ++t) {
        double eps = rand_normal();  // 标准正态分布随机数
        double r_t = MU + sigma_step * eps;  // 收益率
        if (r_t < -0.99) r_t = -0.99;  // 防止价格变负
        s_t = s_t * (1.0 + r_t);  // 更新价格
    }
    terminals_seq[i] = s_t;  // 保存最终价格
}
```

**算法流程**:
1. **外层循环**: 遍历每条 Monte Carlo 路径（`i = 0` 到 `n_sims-1`）
2. **初始化**: `s_t = S0 = 100.0`
3. **计算每步波动率**: `σ_step = σ_annual / √12 ≈ 0.1155`
4. **内层循环**: 模拟每个时间步
   - 生成随机冲击 `ε ~ N(0, 1)`
   - 计算收益率 `r_t = μ + σ_step × ε`
   - 限制收益率 ≥ -99%（防止价格变负）
   - 更新价格 `S_{t+1} = S_t × (1 + r_t)`
5. **保存结果**: `terminals_seq[i] = s_t`（最终价格）

**GBM 模型**: `S_{t+1} = S_t × exp((μ - 0.5σ²)dt + σ√dt·Z)`
- 这里使用简化版本: `S_{t+1} = S_t × (1 + r_t)`，其中 `r_t = μ + σ_step·Z`

#### Lines 193-195: 结束计时并打印
```c
double end_seq = omp_get_wtime();
double time_seq = end_seq - start_seq;
printf("✓ Sequential finished in %.4f seconds\n", time_seq);
```

---

### OpenMP 并行版本基准测试 (Lines 197-234)

#### Lines 198-201: 初始化
```c
printf("\n2. Running OpenMP Parallel...\n");
srand(42);  // 重置种子（注意：并行版本使用 rand_r，这个调用影响不大）

double start_par = omp_get_wtime();
```

#### Lines 203-229: OpenMP 并行循环
```c
// [HPC-OpenMP] Enable path-level parallelism over Monte Carlo simulations.
// This mirrors the Numba `@njit(parallel=True)` + `prange` pattern in Python.
#pragma omp parallel for
for (int i = 0; i < n_sims; ++i) {
    // 每个线程需要自己的随机数状态
    unsigned int seed = i;  // 每条路径使用不同的种子
    double s_t = S0;
    double sigma_step = SIGMA / sqrt((double)STEPS_PER_YEAR);
    
    for (int t = 0; t < n_steps; ++t) {
        // 线程安全的随机数生成
        double u1 = ((double) rand_r(&seed) / RAND_MAX);
        double u2 = ((double) rand_r(&seed) / RAND_MAX);
        if (u1 < 1e-9) u1 = 1e-9;
        double eps = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        
        double r_t = MU + sigma_step * eps;
        if (r_t < -0.99) r_t = -0.99;
        s_t = s_t * (1.0 + r_t);
    }
    terminals_par[i] = s_t;
}
```

**关键点**:

1. **`#pragma omp parallel for`**:
   - **编译器指令**: 告诉编译器并行化这个循环
   - **自动工作分配**: OpenMP 运行时将循环迭代分配给多个线程
   - **隐式同步**: 循环结束后，所有线程同步

2. **线程安全的随机数**:
   - **`rand_r(&seed)`**: 可重入的随机数生成器（线程安全）
   - **`seed = i`**: 每条路径使用不同的种子，确保独立性
   - **Box-Muller 变换**: 在循环内部实现（避免调用非线程安全的 `rand_normal()`）

3. **内存访问**:
   - **读取**: `S0`, `MU`, `SIGMA`（所有线程共享，只读，安全）
   - **写入**: `terminals_par[i]`（每个线程写入不同的 `i`，无竞争）

4. **与顺序版本的差异**:
   - 顺序版本: 调用 `rand_normal()`（使用全局 `rand()`，非线程安全）
   - 并行版本: 内联 Box-Muller 变换 + `rand_r()`（线程安全）

#### Lines 231-234: 结束计时并计算加速比
```c
double end_par = omp_get_wtime();
double time_par = end_par - start_par;
printf("✓ OpenMP Parallel finished in %.4f seconds\n", time_par);
printf("  Speedup: %.2fx\n", time_seq / time_par);
```

**加速比计算**: `speedup = time_seq / time_par`
- **例子**: 顺序 0.72 秒，并行 0.23 秒 → 加速比 3.13×

#### Lines 236-238: 保存结果到 CSV
```c
append_csv_row(csv_path, "openmp_c", "sequential", n_sims, n_steps, 1, time_seq);
append_csv_row(csv_path, "openmp_c", "openmp_parallel", n_sims, n_steps, 1, time_par);
```
- **每个时间范围**: 写入两行（顺序 + 并行）
- **`n_ranks = 1`**: OpenMP 使用线程，不是进程，所以始终为 1

#### Lines 241-245: 清理和退出
```c
printf("\nAll results appended to: %s\n", csv_path);

free(terminals_seq);  // 释放内存
free(terminals_par);
return 0;
```

---

## 关键 OpenMP 概念总结

### 1. **共享内存模型**
- 所有线程共享同一内存空间
- 可以直接访问全局变量和数组
- 需要小心数据竞争（race condition）

### 2. **编译器指令 (Pragmas)**
- `#pragma omp parallel for`: 并行化 for 循环
- 编译器自动生成并行代码
- 运行时动态创建线程池

### 3. **工作分配**
- OpenMP 自动将循环迭代分配给线程
- **默认策略**: 静态分配（每个线程处理大致相等的迭代次数）
- **例子**: 1,000,000 次迭代，4 个线程 → 每个线程约 250,000 次

### 4. **线程安全**
- **问题**: `rand()` 不是线程安全的（使用全局状态）
- **解决**: 使用 `rand_r(&seed)`，每个线程有自己的种子
- **种子策略**: `seed = i`（路径索引），确保可重现性

### 5. **计时函数**
- `omp_get_wtime()`: 高精度计时（类似 `MPI.Wtime()`）
- 返回从某个固定时间点开始的秒数

---

## 执行流程示例（4 个线程，1,000,000 次模拟，12 步）

```
时间轴:
t0: 程序启动
    - 解析命令行参数
    - 分配内存（terminals_seq, terminals_par）

t1: 开始顺序版本
    - srand(42) 设置种子
    - start_seq = omp_get_wtime()
    - 单线程顺序执行 1,000,000 次循环
    - 每次循环：模拟 12 步价格路径
    - end_seq = omp_get_wtime()
    - time_seq = 0.72 秒

t2: 开始并行版本
    - srand(42) 重置种子（影响不大）
    - start_par = omp_get_wtime()
    - OpenMP 创建 4 个线程（假设 4 核 CPU）

t3: 并行执行（4 个线程）
    Thread 0: 处理 i = 0 到 249,999
    Thread 1: 处理 i = 250,000 到 499,999
    Thread 2: 处理 i = 500,000 到 749,999
    Thread 3: 处理 i = 750,000 到 999,999
    [所有线程同时运行，共享内存]

t4: 所有线程完成
    - 隐式同步（#pragma omp parallel for 自动处理）
    - end_par = omp_get_wtime()
    - time_par = 0.23 秒

t5: 计算加速比
    - speedup = 0.72 / 0.23 = 3.13×

t6: 写入 CSV
    - 追加两行到 hpc_benchmark_openmp.csv

t7: 释放内存，程序结束
```

---

## 与 MPI/Numba 的对比

| 特性 | OpenMP (C) | MPI (Python) | Numba (Python) |
|------|------------|--------------|----------------|
| **内存模型** | 共享内存 | 分布式内存 | 共享内存 |
| **并行单位** | 线程 | 进程 | 线程（通过 LLVM） |
| **通信** | 隐式共享变量 | 显式消息传递 | 隐式共享变量 |
| **代码修改** | 编译器指令 | 显式通信调用 | 装饰器 + prange |
| **扩展性** | 单机多核 | 可跨多台机器 | 单机多核 |
| **开销** | 低（线程创建） | 高（进程间通信） | 中等（JIT 编译） |
| **适用场景** | 单机多核、细粒度并行 | 大规模集群 | 单机多核、Python 生态 |

### 代码对比

**OpenMP (C)**:
```c
#pragma omp parallel for
for (int i = 0; i < n_sims; ++i) {
    // 并行循环体
}
```

**Numba (Python)**:
```python
@njit(parallel=True)
def mc_numba_parallel(...):
    for i in prange(n_sims):  # prange = parallel range
        # 并行循环体
```

**MPI (Python)**:
```python
# 每个 rank 处理 local_n_sims 条路径
local_terminals = run_mc_paths(local_n_sims, ...)
# 需要显式通信收集结果
```

---

## 常见问题

### Q1: 为什么顺序版本和并行版本的结果可能不同？
**A**: 
- **随机数生成**: 顺序版本使用 `rand_normal()`（全局 `rand()`），并行版本使用 `rand_r(&seed)`
- **种子策略**: 并行版本 `seed = i`，顺序版本使用全局种子序列
- **数值误差**: 浮点运算顺序不同可能导致微小差异

### Q2: `#pragma omp parallel for` 如何工作？
**A**:
1. 编译器识别这个指令
2. 生成代码创建线程池（通常等于 CPU 核心数）
3. 将循环迭代分配给线程（静态分配）
4. 线程并行执行各自的迭代
5. 循环结束后自动同步所有线程

### Q3: 为什么 `n_ranks = 1`？
**A**: OpenMP 使用**线程**（threads），不是**进程**（processes）。`n_ranks` 是 MPI 的概念（进程数），OpenMP 中始终为 1（单进程多线程）。

### Q4: 如何控制线程数？
**A**: 
- **环境变量**: `export OMP_NUM_THREADS=8`
- **运行时设置**: `omp_set_num_threads(8)`
- **默认**: 通常等于 CPU 核心数

### Q5: `rand_r()` 和 `rand()` 的区别？
**A**:
- **`rand()`**: 使用全局状态，**不是线程安全的**
- **`rand_r(&seed)`**: 使用传入的种子指针，**线程安全的**
- **并行代码必须使用 `rand_r()`**，否则可能产生竞争条件

### Q6: 为什么并行版本的随机数生成在循环内部？
**A**: 
- `rand_normal()` 函数使用全局 `rand()`，不是线程安全的
- 在循环内部内联 Box-Muller 变换，使用 `rand_r(&seed)`，确保线程安全

### Q7: 内存分配在哪里？
**A**: 
- **Line 148-149**: 在循环外分配一次，所有时间范围共享
- **优点**: 避免重复分配/释放的开销
- **大小**: `n_sims × sizeof(double)` 字节（例如 1M × 8 = 8MB）

### Q8: 如何编译和运行？
**A**:
```bash
# Linux
gcc -O3 -fopenmp openmp_mc_demo.c -o openmp_mc_demo

# macOS (with Homebrew libomp)
clang -O3 -I/opt/homebrew/opt/libomp/include \
      -L/opt/homebrew/opt/libomp/lib \
      -Xpreprocessor -fopenmp -lomp \
      openmp_mc_demo.c -o openmp_mc_demo

# 运行
./openmp_mc_demo                    # 多时间范围（1Y, 3Y, 5Y）
./openmp_mc_demo --steps 36         # 单时间范围（3Y）
./openmp_mc_demo --steps 60 --sims 500000  # 自定义参数
```

---

## 性能优化建议

### 1. **线程数调优**
```bash
export OMP_NUM_THREADS=8  # 根据 CPU 核心数设置
./openmp_mc_demo
```

### 2. **编译器优化**
```bash
gcc -O3 -march=native -fopenmp ...  # -march=native 针对当前 CPU 优化
```

### 3. **内存对齐**
- 当前代码使用 `malloc()`，可能未对齐
- 可以使用 `aligned_alloc()` 提高缓存性能

### 4. **随机数生成器**
- 当前使用 `rand_r()`，性能一般
- 高性能场景可以使用更快的生成器（如 PCG、xoshiro）

---

## 总结

OpenMP 提供了**最简单**的并行化方式：
- **一行代码**: `#pragma omp parallel for`
- **自动管理**: 线程创建、工作分配、同步
- **共享内存**: 无需显式通信
- **性能**: 接近手写多线程代码

这使得 OpenMP 成为**单机多核并行**的理想选择，特别适合 Monte Carlo 这种**数据并行**的工作负载。

