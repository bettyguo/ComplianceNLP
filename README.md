# ComplianceNLP

**Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

**ComplianceNLP** is an end-to-end system for automated regulatory compliance monitoring in financial services. It monitors regulatory changes, extracts structured obligations, and identifies compliance gaps against institutional policies across three major frameworks: **SEC**, **MiFID II**, and **Basel III**.

The system integrates three components:

1. **KG-Augmented RAG Pipeline** — Grounds generations in a Regulatory Knowledge Graph (RKG) of 12,847 provisions with hybrid dense+sparse retrieval and KG-based re-ranking.
2. **Multi-Task Obligation Extraction** — Jointly trains NER, deontic classification, and cross-reference resolution over a shared LEGAL-BERT encoder.
3. **Compliance Gap Analysis** — Maps extracted obligations to internal policies with severity-aware scoring, powered by a distilled LLaMA-3-8B generator.

### Key Results

| Metric | Value |
|--------|-------|
| Gap Detection F1 (δ=0.6) | **87.7** |
| NER F1 | **91.3** |
| Grounding Accuracy | **94.2%** (r=0.83 vs. human) |
| End-to-End F1 (error propagation) | **83.4** |
| Production Recall (4-month parallel run) | **96.0%** |
| Analyst Efficiency Gain | **3.1×** sustained |
| Inference Speedup (KD + Medusa) | **2.8×** |

---

## 🏭 Main Work

### Deployment Context

| Aspect | Details |
|--------|---------|
| Status | Parallel-run deployment (Phase 2) |
| Duration | 4 months (Oct 2025 – Jan 2026) |
| Scale | 9,847 regulatory updates processed |
| Users | 12 compliance analysts across 3 regulatory teams |
| Latency | P50: 659ms (generator), P99: 1,082ms |
| Infrastructure | AWS (A100 80GB), Neo4j, FAISS |
| Throughput | ~2,400 docs/month |

### Production Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Processing Time/Update | 47 min | 15 min | 3.1× faster |
| Gaps Flagged Correctly | — | 90.7% precision | — |
| Estimated Recall | — | 96.0% | — |
| System Uptime | — | 99.2% | — |
| Monthly Cost | $45K | $48.2K (Phase 2) | Projected −55% at Phase 4 |

### System Architecture

```
[Regulatory Documents] → [Chunking + Embedding] → [Vector Store]
                                                  → [Regulatory KG (12,847 nodes)]
                                                        ↓
[Regulatory NER] → [Deontic Classifier] → [Cross-Ref Resolver] → [Structured Obligations]
                                                                        ↓
[Internal Policies] → [Obligation–Policy Alignment] → [Gap Severity Scoring] → [Gap Report]
```

### Lessons Learned

1. **Structural knowledge outperforms embeddings for cross-references.** KG-based relationships improved cross-reference resolution from 72.3 to 89.1 F1 (+16.8), the single most impactful design decision.
2. **Formulaic language enables efficient speculative decoding.** Regulatory text's constrained vocabulary yields 91.3% Medusa acceptance rates vs. 82.7% on general text (H=2.31 vs. 3.87 bits).
3. **Analysts trust recall more than F1.** A single missed compliance gap erodes institutional trust. Recall-optimized thresholds and confidence score display are essential.
4. **GRC integration is harder than model development.** Integration with legacy GRC platforms consumed ~3 months—comparable to the entire model development cycle.
5. **Organizational adoption requires staged trust-building.** Plan for a 2–3 month trust calibration period with transparent weekly performance reporting.

---

## Installation

### Requirements

- Python 3.10–3.11
- CUDA 12.1+ (for GPU inference)
- Neo4j 5.x (for knowledge graph)
- 16GB+ GPU memory (A100 recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/anonymous/ComplianceNLP.git
cd ComplianceNLP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Or install serving dependencies only
pip install -e ".[serve]"
```

### Quick Start

```bash
# Run obligation extraction on a regulatory document
python -m compliance_nlp.extraction.run --input data/sample_regulation.txt

# Run gap analysis
python -m compliance_nlp.gap_analysis.run \
    --obligations outputs/obligations.json \
    --policies data/sample_policies/

# Start serving API
uvicorn compliance_nlp.serving.server:app --host 0.0.0.0 --port 8080

# Run latency benchmark
python -m compliance_nlp.evaluation.latency_benchmark

# Verify ACL Industry Track compliance
python scripts/verify_acl_industry_compliance.py
```

---

## Repository Structure

```
ComplianceNLP/
├── src/compliance_nlp/
│   ├── __init__.py
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py          # RegObligation and GapBench loaders
│   │   └── preprocessing.py     # Regulatory text preprocessing
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── legal_bert.py        # Multi-task LEGAL-BERT encoder
│   │   └── gap_generator.py     # LLaMA-3-based gap analysis
│   ├── knowledge_graph/         # Regulatory Knowledge Graph
│   │   ├── __init__.py
│   │   ├── schema.py            # RKG node/edge types
│   │   ├── builder.py           # KG construction pipeline
│   │   └── query.py             # Graph traversal and scoring
│   ├── retrieval/               # Hybrid retrieval pipeline
│   │   ├── __init__.py
│   │   ├── dense.py             # Dense bi-encoder retrieval
│   │   ├── sparse.py            # BM25 sparse retrieval
│   │   ├── hybrid.py            # Weighted hybrid retrieval
│   │   └── kg_reranker.py       # KG-based re-ranking
│   ├── extraction/              # Multi-task obligation extraction
│   │   ├── __init__.py
│   │   ├── ner.py               # CRF-based regulatory NER
│   │   ├── deontic.py           # Deontic modality classifier
│   │   ├── crossref.py          # Cross-reference resolver
│   │   ├── multitask.py         # Joint training module
│   │   └── run.py               # Extraction entry point
│   ├── gap_analysis/            # Compliance gap analysis
│   │   ├── __init__.py
│   │   ├── alignment.py         # Obligation–policy alignment
│   │   ├── severity.py          # Gap severity scoring
│   │   ├── grounding.py         # MiniCheck grounding verification
│   │   ├── report.py            # Gap report generation
│   │   └── run.py               # Gap analysis entry point
│   ├── optimization/            # Production optimization
│   │   ├── __init__.py
│   │   ├── distillation.py      # MiniLLM knowledge distillation
│   │   └── medusa_heads.py      # Medusa speculative decoding heads
│   ├── serving/                 # Production serving
│   │   ├── __init__.py
│   │   └── server.py            # FastAPI server with Prometheus
│   ├── evaluation/              # Evaluation and benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py           # Quality metrics (F1, EM, etc.)
│   │   └── latency_benchmark.py # Production latency benchmarking
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── reproducibility.py   # Seed setting, determinism
├── configs/                     # Configuration files
│   ├── extraction.yaml          # Extraction training config
│   ├── gap_analysis.yaml        # Gap analysis config
│   ├── distillation.yaml        # Knowledge distillation config
│   ├── serving.yaml             # Serving configuration
│   └── medusa.yaml              # Medusa heads training config
├── scripts/                     # Utility scripts
│   ├── train_extraction.py      # Train multi-task extraction model
│   ├── train_gap_analysis.py    # Train/distill gap analysis model
│   ├── train_medusa.py          # Train Medusa speculative heads
│   ├── build_knowledge_graph.py # Build RKG from regulatory feeds
│   └── evaluate.py              # Run full evaluation suite
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_retrieval.py
│   ├── test_gap_analysis.py
│   └── test_knowledge_graph.py
├── docs/                        # Documentation
│   └── COMPUTE_RESOURCES.md
├── examples/                    # Example inputs and outputs
│   ├── sample_regulation.txt
│   └── sample_output.json
├── kubernetes/                  # K8s deployment configs
│   └── deployment.yaml
├── Dockerfile.serving           # Production serving container
├── pyproject.toml               # Project configuration
├── requirements.txt             # Core dependencies
├── requirements-serving.txt     # Serving-only dependencies
├── requirements-dev.txt         # Development dependencies
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## Training

### 1. Build Knowledge Graph

```bash
python scripts/build_knowledge_graph.py \
    --sec-dir data/sec_edgar/ \
    --mifid-dir data/eurlex/ \
    --basel-dir data/bis_pdf/ \
    --output-dir outputs/knowledge_graph/ \
    --neo4j-uri bolt://localhost:7687
```

### 2. Train Multi-Task Extraction Model

```bash
python scripts/train_extraction.py \
    --config configs/extraction.yaml \
    --data-dir data/regobligation/ \
    --output-dir outputs/extraction_model/ \
    --seed 42
```

### 3. Knowledge Distillation (70B → 8B)

```bash
python scripts/train_gap_analysis.py \
    --config configs/distillation.yaml \
    --teacher meta-llama/Meta-Llama-3-70B-Instruct \
    --student meta-llama/Meta-Llama-3-8B-Instruct \
    --data-dir data/compliance_instructions/ \
    --output-dir outputs/distilled_model/
```

### 4. Train Medusa Speculative Decoding Heads

```bash
python scripts/train_medusa.py \
    --config configs/medusa.yaml \
    --base-model outputs/distilled_model/ \
    --data-dir data/regulatory_corpus/ \
    --output-dir outputs/medusa_model/
```

---

## Evaluation

```bash
# Full evaluation suite
python scripts/evaluate.py \
    --config configs/gap_analysis.yaml \
    --extraction-model outputs/extraction_model/ \
    --gap-model outputs/distilled_model/ \
    --data-dir data/ \
    --output-dir outputs/evaluation/

# Latency benchmark
python -m compliance_nlp.evaluation.latency_benchmark \
    --model outputs/medusa_model/ \
    --batch-size 1 \
    --num-samples 500 \
    --warmup 50
```

---

## Serving

```bash
# Local serving
uvicorn compliance_nlp.serving.server:app --host 0.0.0.0 --port 8080

# Docker
docker build -f Dockerfile.serving -t compliancenlp:latest .
docker run --gpus all -p 8080:8080 compliancenlp:latest

# Kubernetes
kubectl apply -f kubernetes/deployment.yaml
```

---

## Hardware Requirements

| Component | Training | Inference |
|-----------|----------|-----------|
| GPU | 2× A100 80GB (distillation) | 1× A100 80GB |
| CPU | 32 cores | 8 cores |
| RAM | 128 GB | 32 GB |
| Storage | 500 GB SSD | 100 GB SSD |
| Neo4j | 16 GB RAM | 8 GB RAM |

**Total training compute:** ~180 GPU-hours on A100 80GB (~$720 at $4/GPU-hr).

---

## Reproducibility

While production data cannot be released, we provide:

- [x] Model architecture and training code
- [x] Inference and serving code
- [x] Benchmark evaluation scripts
- [x] Latency benchmarking utilities
- [x] Knowledge graph construction pipeline
- [x] Configuration files with all hyperparameters
- [x] Sample evaluation data
- [ ] Training data (proprietary — RegObligation to be released upon publication)
- [ ] GapBench (proprietary — anonymized version in progress)

```bash
# Reproduce main results (with released data)
python scripts/evaluate.py --config configs/gap_analysis.yaml --seed 42

# Run latency benchmark
python -m compliance_nlp.evaluation.latency_benchmark

# Start serving
uvicorn compliance_nlp.serving.server:app --host 0.0.0.0 --port 8080
```

---

## Limitations

1. Coverage is limited to three regulatory frameworks (~48% of annual updates).
2. Primary evaluation dataset (GapBench, 423 examples) derives from a single institution.
3. English-language texts only.
4. User study (12 analysts, 96 updates) is modest and unblinded.
5. Production metrics are from parallel-run operation where all outputs receive human review.
6. The 96.0% production recall is an estimate with irreducible structural uncertainty.
7. P99 latency of 1,082ms exceeds sub-second target.

---

## Responsible NLP Checklist

- **Limitations:** See Limitations section above and paper appendix.
- **Risks:** System is a decision-support tool; all high-severity findings require human review.
- **Compute:** ~180 GPU-hours on A100 80GB for training; ~$48.2K/month serving cost.
- **Human Evaluation:** 2 compliance experts annotated 200 samples for grounding validation (κ=0.87).
- **AI Assistants:** Claude was used for code review and documentation drafting. All outputs were reviewed by authors.

