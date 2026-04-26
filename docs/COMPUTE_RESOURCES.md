# Compute Resources

This document details the computational resources used for training, evaluation, and deployment of ComplianceNLP, as required by the ACL Responsible NLP Checklist (Section C).

## Training Compute

| Component | Hardware | GPU-Hours | Estimated Cost |
|-----------|----------|-----------|----------------|
| Multi-task extraction (LEGAL-BERT) | 1× A100 80GB | ~24 hrs | ~$96 |
| Knowledge distillation (70B→8B) | 2× A100 80GB | ~80 hrs | ~$420 |
| Medusa head training | 1× A100 80GB | ~16 hrs | ~$64 |
| KG construction + embedding | 1× A100 80GB | ~8 hrs | ~$32 |
| Hyperparameter search | 2× A100 80GB | ~50 hrs | ~$200 |
| **Total Training** | | **~180 hrs** | **~$720** |

Cost estimated at $4/GPU-hour for A100 80GB on AWS (p4d.24xlarge spot pricing).

## Inference/Serving Compute

| Component | Hardware | Monthly Cost |
|-----------|----------|--------------|
| Model serving (2 replicas) | 2× A100 80GB | ~$8,100/mo |
| Neo4j (KG storage) | 16GB RAM instance | ~$1,200/mo |
| FAISS index hosting | 32GB RAM instance | ~$800/mo |
| Monitoring + logging | Standard instances | ~$1,100/mo |
| **Total Serving** | | **~$11,200/mo** |

At current volume (~2,400 docs/month), cost is approximately **$4.67/document**.

## Per-Document Processing Cost

| Stage | Avg. Time | Compute |
|-------|-----------|---------|
| Obligation extraction | ~30s | GPU |
| KG retrieval + re-ranking | ~15s | CPU + GPU |
| Gap analysis generation | ~2.5 min | GPU |
| Grounding verification | ~30s | GPU |
| **End-to-end** | **~4.2 min** | |

## Environmental Impact

Using the ML CO2 Impact tool methodology:
- Training: ~180 GPU-hours on A100 → estimated ~25 kg CO2eq
- Inference: ~2,400 docs/month → estimated ~15 kg CO2eq/month
- Location: AWS us-east-1 (Virginia), PUE ~1.2

## Retraining Requirements

- Distributional shift retraining: ~8 hours on 2× A100 (estimated 2-3 events per 4 months)
- Steady-state maintenance: ~4-6 hours/week for KG review queue
- Model update frequency: ~1 per 6-8 weeks
