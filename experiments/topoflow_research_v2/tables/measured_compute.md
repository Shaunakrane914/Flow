# Measured compute and storage

| Model | Parameters | Weight MiB | Median forward ms | P95 forward ms |
|---|---:|---:|---:|---:|
| GAT standardized | 15,425 | 0.0588 | 1.8779 | 2.9284 |
| GAT physics + geometry | 15,618 | 0.0596 | 1.7185 | 2.6733 |
| SAGE physics + geometry | 27,650 | 0.1055 | 1.0977 | 1.6110 |
| Uniform GAT + geometry | 15,618 | 0.0596 | 1.8763 | 2.6622 |
| Pore-set + geometry | 15,106 | 0.0576 | 0.4365 | 0.6504 |

All 24 held-out natural crops are timed using trained fold checkpoints, 10 warmups and 30 forward repeats per crop. Forward measurements exclude graph extraction, normalization, loading and prediction inversion. Parameter storage excludes activations and Python/library memory. No CNN resource comparison is established.
