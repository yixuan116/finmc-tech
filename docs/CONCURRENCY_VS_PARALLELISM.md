# Concurrency vs Parallelism 在你的项目中的应用

## 你的代码中实际上两者都在用！

### 1. **Path-Level: Parallelism (并行) - Numba `prange`**

**场景**: 单个场景内的 Monte Carlo 路径计算

```python
# mc_numba_parallel 函数中
for i in prange(n_sims):  # 并行循环
    for t in range(horizon_steps):
        # 计算单个路径
```

**为什么用 Parallelism？**
- ✅ **CPU 密集型任务**: 纯计算，没有 I/O 等待
- ✅ **数据独立**: 每个路径的计算完全独立
- ✅ **可扩展**: 路径越多，并行优势越明显（你的测试显示 500K 时达到 10.9x 加速）
- ✅ **真正的并行**: 多个 CPU 核心同时计算不同路径

**适用场景**: 
- 单个场景需要运行大量 Monte Carlo 模拟（10K - 500K+）
- 计算密集型，没有 I/O 操作

---

### 2. **Scenario-Level: Concurrency (并发) - ThreadPoolExecutor**

**场景**: 多个不同场景（BASE, MACRO_STRESS, VIX_CRASH 等）同时运行

```python
# run_scenario_forecast 函数中
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(_run_single_scenario_task, ...): scenario_name
        for scenario_name, shock_spec in scenarios.items()
    }
```

**为什么用 Concurrency？**
- ✅ **任务级并行**: 每个场景是独立的任务
- ✅ **粗粒度分解**: 场景之间没有数据依赖
- ✅ **资源利用**: 当一个场景在等待时，其他场景可以运行
- ✅ **Python GIL 友好**: ThreadPoolExecutor 在 I/O 或释放 GIL 的操作中有效

**适用场景**:
- 需要同时运行多个不同的场景（通常 4-5 个场景）
- 每个场景内部已经用 Numba 并行化了

---

## 为什么不用 Concurrency 替代 Parallelism？

### ❌ 如果用 ThreadPoolExecutor 做路径级并行：

```python
# 不好的做法
def run_paths_with_threads(n_sims):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_single_path, i) for i in range(n_sims)]
        # 问题：创建 10K-500K 个线程/任务，开销巨大！
```

**问题**:
1. **线程开销**: 创建 10K+ 线程成本太高
2. **GIL 限制**: Python 的 GIL 会限制真正的并行执行
3. **内存开销**: 每个线程需要独立栈空间
4. **调度开销**: 操作系统线程调度会成为瓶颈

### ✅ 为什么 Numba `prange` 更好：

```python
# 好的做法
@njit(parallel=True)
for i in prange(n_sims):  # 编译为原生并行代码
    # 直接利用 CPU 核心，无 GIL，无线程开销
```

**优势**:
1. **零开销**: 编译为原生机器码，无 Python 解释器开销
2. **无 GIL**: 绕过 Python 的全局解释器锁
3. **SIMD**: 可以利用 CPU 的向量指令
4. **自动优化**: Numba 自动处理负载均衡

---

## 你的项目中的两层并行化架构

```
┌─────────────────────────────────────────┐
│  Scenario-Level Concurrency             │
│  (ThreadPoolExecutor)                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ BASE    │ │ MACRO   │ │ VIX     │  │
│  │ Scenario│ │ STRESS  │ │ CRASH   │  │
│  └────┬────┘ └────┬────┘ └────┬────┘  │
│       │            │            │       │
│       ▼            ▼            ▼       │
│  ┌───────────────────────────────────┐  │
│  │ Path-Level Parallelism (Numba)   │  │
│  │ ┌──┐ ┌──┐ ┌──┐ ... ┌──┐         │  │
│  │ │P1│ │P2│ │P3│     │Pn│         │  │
│  │ └──┘ └──┘ └──┘     └──┘         │  │
│  │ 每个路径并行计算 (prange)        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 为什么这样设计？

1. **场景级 Concurrency**:
   - 场景数量少（4-5 个），适合线程池
   - 场景之间完全独立，无数据共享
   - 每个场景内部已经并行化，线程池只是协调

2. **路径级 Parallelism**:
   - 路径数量大（10K-500K），需要真正的并行
   - CPU 密集型，Numba 编译后性能最优
   - 避免线程开销，直接利用 CPU 核心

---

## 什么时候应该用 Concurrency？

### ✅ 适合 Concurrency 的场景：

1. **I/O 密集型任务**:
   ```python
   # 下载多个文件
   with ThreadPoolExecutor() as executor:
       futures = [executor.submit(download_file, url) for url in urls]
   ```

2. **粗粒度任务分解**:
   ```python
   # 处理多个独立的数据文件
   with ThreadPoolExecutor() as executor:
       futures = [executor.submit(process_file, f) for f in files]
   ```

3. **你的场景级并行**:
   ```python
   # 运行多个独立的 Monte Carlo 场景
   with ThreadPoolExecutor() as executor:
       futures = [executor.submit(run_scenario, s) for s in scenarios]
   ```

### ❌ 不适合 Concurrency 的场景：

1. **细粒度 CPU 密集型循环**:
   ```python
   # 不好：创建太多线程
   for i in range(100000):
       thread = Thread(target=compute, args=(i,))
       thread.start()
   ```

2. **需要真正并行计算的场景**:
   ```python
   # 应该用 Numba prange 或 MPI
   for i in prange(n_sims):  # 真正的并行
       compute_path(i)
   ```

---

## 总结

### 你的项目中的最佳实践：

| 层级 | 策略 | 工具 | 原因 |
|------|------|------|------|
| **路径级** | Parallelism | Numba `prange` | CPU 密集型，大规模（10K-500K），需要真正并行 |
| **场景级** | Concurrency | ThreadPoolExecutor | 粗粒度任务（4-5 个场景），独立任务 |

### 关键区别：

- **Parallelism (并行)**: 多个 CPU 核心**同时**执行计算
  - 你的路径计算：500K 路径 → 10.9x 加速
  
- **Concurrency (并发)**: 多个任务**交替**执行，看起来同时
  - 你的场景计算：4-5 个场景并发运行

### 为什么不用 Concurrency 做路径级？

1. **开销太大**: 创建 10K+ 线程成本高
2. **GIL 限制**: Python 线程受 GIL 限制
3. **性能差**: Numba 并行比线程快 10x+

**结论**: 你的架构设计是正确的！在不同层级使用不同的并行化策略是最优解。

