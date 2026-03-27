# TurboQuant Algorithm Reference

## Overview

TurboQuant (ICLR 2026, Google Research) is a near-optimal vector quantization algorithm that compresses high-dimensional vectors to 2-4 bits per coordinate with provably minimal distortion. It is **data-oblivious** (no training/calibration needed) and works **online** (each vector quantized independently).

Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## How It Works

**Random Rotation → Predictable Distribution → Optimal Scalar Quantization**

1. Multiply input vector by a fixed random orthogonal matrix Π (from QR decomposition of random Gaussian). This makes every coordinate follow the same Beta distribution ≈ N(0, 1/d), regardless of the original vector.

2. Since coordinates are now ~independent with a known distribution, apply a precomputed Lloyd-Max optimal scalar quantizer to each coordinate independently. This is provably the best b-bit scalar quantizer for that distribution.

3. For unbiased inner product estimation (needed for search), add a 1-bit QJL (Quantized Johnson-Lindenstrauss) correction on the residual. This eliminates the bias that MSE-optimal quantizers introduce in dot products.

## Two Modes

### TurboQuant_mse (Algorithm 1)
- Uses all b bits for MSE-optimal quantization
- Best for: storage compression, nearest-neighbor by L2 distance
- MSE distortion: ≈ 0.36 (b=1), 0.117 (b=2), 0.03 (b=3), 0.009 (b=4)

### TurboQuant_prod (Algorithm 2) — Recommended for search
- Uses (b-1) bits for MSE + 1 bit for QJL residual correction
- Guarantees **unbiased** inner product estimation
- Best for: cosine similarity search, retrieval, RAG
- IP distortion: ≈ 0.047/d at b=4 (for d=768: correlation > 0.98)

## Choosing Bit-Width

| Bits | Compression | MSE | IP Correlation (d=768) | Recommended For |
|------|-------------|-----|----------------------|-----------------|
| 2 | ~16x | 0.117 | ~0.80 | Maximum compression, approximate search |
| 3 | ~8x | 0.03 | ~0.93 | Good balance for large datasets |
| 4 | ~8x | 0.009 | ~0.86 | Maximum compression, some recall loss |
| 5 | ~6x | 0.003 | ~0.92 | Good balance for large datasets |
| **6** | **~5x** | **0.001** | **~0.98** | **Default — best quality/compression tradeoff** |

## Dimension Considerations

- **d ≥ 128**: Algorithm works well, Gaussian approximation is accurate
- **d = 768** (Gemini, many sentence transformers): Sweet spot
- **d = 1536** (OpenAI text-embedding-3-large): Excellent, even higher correlation
- **d = 3072** (some code embeddings): Near-perfect at b=4

Higher dimensions → better quantization quality (concentration of measure).

## Asymmetric Search

The key performance trick: keep queries in full float32, only quantize the database.
- Query rotation: q_rot = Π · q (one matrix multiply per query)
- Score computation: dot product with codebook centroids + QJL correction
- No need to dequantize stored vectors → saves memory and computation

## Theoretical Guarantees

- MSE within **2.7x** of information-theoretic lower bound (Shannon)
- Inner product estimator is **mathematically unbiased** (Theorem 2)
- **Zero indexing time** — no offline training, no codebook learning from data
- Provably near-optimal across ALL bit-widths and dimensions
