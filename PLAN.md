# TurboQuant Memory Skill — 最终设计方案 v2

> 综合 Claude Opus 自审 + GPT-4.1 审核 + GPT-5.4 深度审核后的定稿

## 目标

用 TurboQuant 核心算法压缩 OpenClaw memory 的 embedding 向量，降低存储、加速搜索。

## 核心算法流程（修订版）

```
原始 embedding (float32, 3072维, L2归一化)
    → ① 存储 norm (fp16)，L2 normalize
    → ② Pad 到 4096 维
    → ③ 随机符号 × Fast Walsh-Hadamard Transform
    → ④ 随机采样回 3072 维 + 尺度修正 √(4096/3072)
    → ⑤ Lloyd-Max 标量量化（预计算码本，无需 per-block scale/zero-point）
    → ⑥ [可选] SRHT sketch 残差纠错（m=256, 仅 32 bytes）
    → 压缩向量存储 + 批量近似内积搜索
```

## 关键设计决策（v2 更新）

### 1. Hadamard 采样策略（GPT-5.4 方案B）
- pad 到 4096 → FWHT → **随机采样** 3072 维（非截取前 3072）
- 尺度修正 √(4096/3072) 保持二阶矩
- 只需存 4096 个符号 + 3072 个采样索引（可复现，固定 seed）

### 2. QJL 改用 SRHT 低维 sketch（核心改进）
- **不再用 d×d 稠密矩阵**（原方案 37MB）
- 改用 SRHT sketch 投影到 m=256 维
- 存储：256 bit = 32 bytes（vs 原来 37MB）
- 修正公式：`score += α * (sign_bits @ S·query)`

### 3. Lloyd-Max 码本预计算
- 标准正态的最优码本是常数
- 对 4/6/8 bit 直接 hardcode，运行时零计算
- 去掉 scipy 依赖

### 4. 每向量 scale（新增）
- 旋转后不同向量动态范围有差异
- 存一个 fp16 scale，重构时乘回
- 额外开销仅 2 bytes/向量

### 5. 搜索策略：两阶段重排
- Stage 1：sqlite-vec 原生检索 top-N（200~1000）
- Stage 2：TurboQuant 对候选集做精确重排
- 避免纯 Python 扫全库

### 6. 搜索批量化
- Block decode (512 向量/块) + numpy matmul
- 残差修正：unpack sign bits → int8 矩阵 → batch matmul
- 预计算 `Q @ query` 和 `S @ query` 各一次

## 项目结构（基于现有 skill 改进）

```
skills/turboquant-memory/
├── SKILL.md
├── references/
│   └── algorithm.md
├── scripts/
│   ├── turboquant.py           # 核心算法（改进版）
│   │   ├── SRHTRotate           # 替代 QR 分解
│   │   ├── SRHTSketch           # 替代 d×d QJL 矩阵
│   │   ├── TurboQuantMSE        # 保留，换旋转实现
│   │   ├── TurboQuantProd       # 保留，换 QJL 实现
│   │   └── 预计算 Lloyd-Max 码本
│   ├── memory_quantize.py       # OpenClaw 集成（改进版）
│   │   ├── OpenClawVecReader    # 新增：读取 sqlite-vec vec0
│   │   ├── detect_vec0_tables() # 新增：自动检测 vec0 表
│   │   └── batch_search()       # 新增：批量化搜索
│   └── validate.py              # 新增：分布验证工具
│       ├── 均值/方差/偏度/峰度
│       ├── Jarque-Bera 检验
│       ├── 分位数误差
│       ├── 维间去相关检查
│       └── 量化误差 + recall@k
```

## 与现有代码的关系

保留的部分：
- TurboQuantMSE / TurboQuantProd 的 API 设计和类结构
- Lloyd-Max 迭代逻辑（简化为纯 numpy + 可选 hardcode）
- pack_indices / unpack_indices 位打包
- 测试套件框架（扩展更多测试）

替换的部分：
- `np.linalg.qr(randn(d,d))` → `SRHTRotate`（pad+FWHT+采样）
- `rng.randn(d,d)` QJL → `SRHTSketch`（m=256 低维投影）
- `scipy.stats.norm` → 纯 numpy pdf/cdf（Abramowitz-Stegun 近似）
- 通用 schema 检测 → OpenClaw vec0 专用 reader

新增的部分：
- 分布验证工具 validate.py
- 每向量 fp16 scale
- 两阶段重排搜索
- 批量化 search（block decode + matmul）
- 持久化缓存格式（codes.bin + residual_bits.bin + scales.npy + meta.json）

## 内存对比

| 组件 | 原方案 | 改进后 |
|------|--------|--------|
| 旋转矩阵 | 37MB (d×d float64) | 16KB (4096 signs + 3072 indices) |
| QJL 矩阵 | 37MB (d×d float32) | 16KB (同上，共享 FWHT) |
| 码本 | 运行时计算 | 硬编码，<1KB |
| **总计** | **~74MB** | **<50KB** |

## 存储格式（每向量）

| 组件 | 大小 |
|------|------|
| norm (fp16) | 2 bytes |
| scale (fp16) | 2 bytes |
| MSE codes (4-bit packed) | 1,536 bytes |
| QJL signs (256-bit packed) | 32 bytes |
| **总计** | **~1,572 bytes** |
| 原始 float32 | 12,288 bytes |
| **压缩比** | **~7.8x** |

## 分阶段实施

### Phase 0：分布验证（2h）
1. 从 sqlite-vec 提取所有 embedding
2. 应用 SRHT 旋转
3. 统计验证：均值/方差/偏度/峰度/JB/分位数/去相关
4. 输出报告，决定是否继续

### Phase 1：核心算法改进（4h）
1. SRHTRotate 类（pad+FWHT+采样）
2. SRHTSketch 类（低维残差投影）
3. 纯 numpy Lloyd-Max + hardcode 码本
4. 更新 TurboQuantMSE/Prod 使用新组件
5. 扩展测试套件

### Phase 2：OpenClaw 集成（3h）
1. vec0 表检测和 reader
2. 批量化搜索引擎
3. 持久化缓存格式
4. benchmark 对比

### Phase 3：封装为 Skill（1h）
1. 更新 SKILL.md
2. CLI 接口（compress / search / benchmark / validate）
3. 发布到 ClawHub

## 依赖

- `numpy`（唯一依赖）
- Python 3.9+

## 参考

- TurboQuant: https://arxiv.org/abs/2504.19874
- PolarQuant: https://arxiv.org/abs/2502.02617
- GPT-5.4 审核意见: 2026-03-27
- 现有实现: clawhub.ai/sunnyztj/turboquant-memory
