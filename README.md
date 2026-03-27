# TurboQuant Memory

> Near-optimal vector quantization for memory/RAG embedding compression.  
> Based on [TurboQuant (Google, ICLR 2026)](https://arxiv.org/abs/2504.19874).

**5-8x compression · 98%+ recall · numpy only · no GPU**

---

## ✨ Features

- **98% Recall@1** at 6.4x compression (5-bit, tested on Gemini embeddings)
- **Blockwise Hadamard rotation** — fully invertible, O(d log d), <50KB memory
- **Lloyd-Max optimal quantization** — precomputed codebooks, zero training
- **Any embedding model** — Gemini, OpenAI, Cohere, sentence-transformers, etc.
- **Any dimension ≥ 128** — auto-splits into power-of-2 blocks
- **OpenClaw sqlite-vec** native support (auto-detects `vec0` tables)
- **Distribution validation** tool to verify your data before quantizing
- **Single dependency**: `numpy`

## 📊 Benchmark

Tested on real Gemini embedding-001 vectors (3072-dim, L2-normalized, 112 vectors):

| Bits | MSE | Cosine | Recall@1 | Bytes/vec | Compression |
|------|-----|--------|----------|-----------|-------------|
| 3 | 1.1e-5 | 0.982 | 88% | 1,160 | 10.6x |
| 4 | 3.2e-6 | 0.995 | 92% | 1,544 | 8.0x |
| **5** | **8.2e-7** | **0.999** | **98%** | **1,928** | **6.4x** |
| 6 | 2.2e-7 | 1.000 | 96%+ | 2,312 | 5.3x |
| 7 | 8e-8 | 1.000 | 100% | 2,696 | 4.6x |
| 8 | 3e-8 | 1.000 | 98%+ | 3,080 | 4.0x |

**Recommended: 5-bit** (best quality/compression tradeoff)

## 🚀 Quick Start

```bash
# Run tests
python3 scripts/turboquant.py

# Validate on your data
python3 scripts/validate.py --db /path/to/memory.sqlite --auto-detect --bits 5

# Benchmark + migrate
python3 scripts/memory_quantize.py --db /path/to/memory.db --bits 5 --benchmark
python3 scripts/memory_quantize.py --db /path/to/memory.db --bits 5 --migrate
```

```python
from turboquant import TurboQuantMSE
import numpy as np

tq = TurboQuantMSE(dim=3072, bits=5)

# Compress
stored = tq.quantize(embedding)       # float32 → compact representation
reconstructed = tq.dequantize(stored)  # → float32 (cosine similarity 0.999)

# Search (asymmetric: query is exact float, database is quantized)
q_rot = tq.rotation.apply(query_vec)
for doc in quantized_db:
    score = doc['norm'] * doc['scale'] * np.dot(q_rot, tq.codebook[doc['indices']])
```

## 🔧 How It Works

```
Input embedding (float32, e.g. 3072-dim)
  → Blockwise Hadamard rotation (3 × 1024, fully invertible)
  → Per-vector scale normalization
  → Lloyd-Max scalar quantization (precomputed codebook)
  → Packed bit representation (5-bit = 1,928 bytes vs 12,288 original)
```

Key design choices:
- **Blockwise Hadamard** over SRHT subsampling (ablation showed subsampling creates an irreversible MSE floor)
- **Lloyd-Max** over uniform quantization (optimal for the post-rotation Gaussian-like distribution)
- **MSE mode** over Prod/QJL mode (QJL adds complexity without benefit at <10k vectors)

## 📁 Files

| File | Description |
|------|-------------|
| `scripts/turboquant.py` | Core algorithm (BlockwiseHadamard + Lloyd-Max + QJL) |
| `scripts/memory_quantize.py` | SQLite/sqlite-vec integration, benchmark, migration |
| `scripts/validate.py` | Distribution validation + quantization quality report |
| `references/algorithm.md` | Full algorithm explanation |

## 📦 Install (OpenClaw Skill)

```bash
clawhub install turboquant-memory
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- [Google Research Blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## License

MIT

---

# TurboQuant Memory（中文说明）

> 基于 [TurboQuant（Google，ICLR 2026）](https://arxiv.org/abs/2504.19874) 的近最优向量量化，用于 memory/RAG embedding 压缩。

**5-8 倍压缩 · 98%+ 召回率 · 仅依赖 numpy · 无需 GPU**

## ✨ 特性

- **98% Recall@1**，6.4 倍压缩（5-bit，Gemini embedding 实测）
- **分块 Hadamard 旋转** — 完全可逆，O(d log d)，内存 <50KB
- **Lloyd-Max 最优量化** — 预计算码本，零训练开销
- **支持任意 embedding 模型** — Gemini、OpenAI、Cohere、sentence-transformers 等
- **支持任意维度 ≥ 128** — 自动分块为 2 的幂
- **OpenClaw sqlite-vec 原生支持**（自动检测 `vec0` 表）
- **分布验证工具** — 量化前验证数据是否适合
- **唯一依赖**：`numpy`

## 📊 性能测试

在真实 Gemini embedding-001 向量上测试（3072 维，L2 归一化，112 个向量）：

| 位宽 | MSE | 余弦相似度 | Recall@1 | 每向量大小 | 压缩比 |
|------|-----|-----------|----------|-----------|--------|
| 3 | 1.1e-5 | 0.982 | 88% | 1,160 B | 10.6x |
| 4 | 3.2e-6 | 0.995 | 92% | 1,544 B | 8.0x |
| **5** | **8.2e-7** | **0.999** | **98%** | **1,928 B** | **6.4x** |
| 6 | 2.2e-7 | 1.000 | 96%+ | 2,312 B | 5.3x |
| 7 | 8e-8 | 1.000 | 100% | 2,696 B | 4.6x |

**推荐：5-bit**（性价比最优）

## 🔧 工作原理

```
输入 embedding（float32，如 3072 维）
  → 分块 Hadamard 旋转（3 × 1024，完全可逆）
  → 逐向量 scale 归一化
  → Lloyd-Max 标量量化（预计算码本）
  → 紧凑位打包（5-bit：1,928 字节 vs 原始 12,288 字节）
```

核心设计：
- **分块 Hadamard** 替代 SRHT 子采样（消融实验证明子采样造成不可逆 MSE 地板）
- **Lloyd-Max** 替代均匀量化（对旋转后的近高斯分布是最优的）
- **MSE 模式**为默认（QJL 模式在 <10k 向量时无明显优势）

## 📦 安装（OpenClaw 技能）

```bash
clawhub install turboquant-memory
```

## 参考文献

- [TurboQuant: 在线向量量化与近最优失真率](https://arxiv.org/abs/2504.19874)（ICLR 2026）
- [PolarQuant: 极坐标变换量化 KV 缓存](https://arxiv.org/abs/2502.02617)（AISTATS 2026）
- [Google Research 博客：TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## 许可证

MIT
