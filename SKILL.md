---
name: turboquant-memory
description: >
  Compress and accelerate vector search in memory/RAG systems using TurboQuant
  (ICLR 2026) — near-optimal vector quantization with 6-8x compression and 98%+
  search accuracy. Use when: (1) optimizing embedding storage size, (2) speeding
  up semantic search over large vector collections, (3) user mentions "compress
  embeddings", "quantize vectors", "memory optimization", "faster search",
  "TurboQuant", "vector compression", or "embedding compression", (4) reducing
  memory footprint of RAG systems. Works with any embedding model (Gemini, OpenAI,
  Cohere, local) and any dimension of 128 or higher. No GPU required.
---

# TurboQuant Memory

Compress embedding vectors 6-8x with under 2% search accuracy loss using TurboQuant vector quantization.

## Quick Start

### 1. Benchmark existing memory system

```bash
python3 scripts/turboquant.py
```

Runs built-in tests: MSE distortion, inner product accuracy, recall, compression ratio.

### 2. Quantize a memory database

```bash
python3 scripts/memory_quantize.py --db /path/to/memory.db --bits 6 --benchmark
```

Auto-detects SQLite databases with embedding columns, quantizes all vectors, reports recall metrics.

### 3. Integrate into existing code

```python
from turboquant import TurboQuantProd

# Initialize once (deterministic — same seed = same quantization)
tq = TurboQuantProd(dim=768, bits=6)

# Quantize embeddings for storage
stored = tq.quantize(embedding_vector)  # float32 → compressed

# Search: query stays float32, database is quantized
results = tq.search(query_vec, quantized_database, top_k=10)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | auto-detect | Embedding dimension (768, 1536, 3072, etc.) |
| `bits` | 6 | Total bits per coordinate. 6 = best quality (98% recall, 5.3x). 5 = balanced (92%, 6.3x). 4 = aggressive (86%, 7.9x). |
| `seed_rot` | 42 | Rotation matrix seed. Same seed = reproducible quantization. |
| `seed_qjl` | 137 | QJL projection seed. |

Environment variables: `TURBOQUANT_BITS`, `TURBOQUANT_SEED`

## When to Use Which Mode

- **TurboQuantProd** (default): For search/retrieval. Unbiased inner product estimation.
- **TurboQuantMSE**: For pure storage compression when you need to reconstruct vectors.

## Algorithm Details

See [references/algorithm.md](references/algorithm.md) for the full algorithm explanation, theoretical guarantees, and tuning guide.

## Compatibility

- Python 3.9+, numpy, scipy
- Any embedding dimension ≥ 128
- Any embedding model (Gemini, OpenAI, Cohere, sentence-transformers, etc.)
- SQLite, PostgreSQL (via memory_quantize.py), or direct API use
