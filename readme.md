# Efficient Visual Document Retrieval

This repository explores token compression for multi-vector visual document retrieval models (e.g., ColQwen). The core idea is to reduce the number of document tokens by a factor of `mf` — so `mf=10` means the compressed index uses 1/10 of the original tokens — while preserving retrieval quality through teacher-student distillation.

We experiment with various distillation objectives including listwise ranking loss, score-preserving loss, InfoNCE, RankNet, and LambdaLoss.

## Method

Document embeddings from a pretrained ColQwen model serve as the teacher. A compressed student embedding (with fewer tokens) is optimized so that its query-document score matrix approximates the teacher's. Scoring follows the ColBERT-style MaxSim late interaction:

```
score(q, d) = sum_{t in q} max_{s in d} (q_t · d_s)
```

Attention and image masks are applied to exclude padding tokens during scoring.

## Structure

```
mainv1.py / mainv1_pseudoQ.py   # epoch-based training with SPL loss
mainv2_*.py                     # step-based training, various loss functions
mainv3_*.py                     # further experiments (hard tokens, mixup, noise)
criterion.py                    # loss implementations
makeQ.py / makeQ.sh             # pseudo query generation
ProxyQ/                         # pseudo query module
Qdatasets/                      # query dataloader
evaluator/                      # Recall@k, NDCG@k
utils/                          # shared utilities
```

## Usage

**Pseudo query generation (optional):**
```bash
bash makeQ.sh
```

**Training with SPL loss (v1):**
```bash
python mainv1.py \
  --datasets {dataset} \
  --mfs 5 10 25 50 \
  --data_root data/split_features/colqwen \
  --init_root /path/to/init_features \
  --out_root results \
  --epochs 15 \
  --lr 1e-3
```

**Training with listwise + score-preserving loss (v2):**
```bash
python mainv2_iter_liscore.py \
  --datasets {dataset} \
  --mfs 5 10 25 50 \
  --query_root ProxyQ/results/colqwen \
  --teacher_root /path/to/teacher_features \
  --init_root /path/to/init_features \
  --out_root results \
  --max_steps 23460 \
  --eval_every 200 \
  --lr 1e-3 \
  --k 40 \
  --temp 0.1
```

Checkpoints and logs are saved under `results/{name}/mf{k}/{dataset}/`. Metrics (Recall@1, NDCG@5, latency) are logged to TensorBoard and JSON.

## Dataset Setup

Register dataset file paths in `utils/mapping.py`:

```python
DATASETMAP = {
    "your_dataset": {
        "train":        "your_dataset_train.npz",
        "test":         "your_dataset_test.npz",
        "pseudoQ":      "your_dataset_pseudoQ.npz",
        "split_before": "your_dataset_full.npz",
        "mf5":          "your_dataset_mf5.npz",
        "mf10":         "your_dataset_mf10.npz",
        "mf25":         "your_dataset_mf25.npz",
        "mf50":         "your_dataset_mf50.npz",
    },
}
```

## References

Built on top of ColPali/ColQwen:
```bibtex
@misc{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Manuel Faysse et al.},
  year={2024},
  eprint={2407.01449},
  archivePrefix={arXiv}
}
```
