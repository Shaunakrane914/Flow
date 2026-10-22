# Controlled GAT ablation

Four development scans are evaluated with nested parent-held-out validation. The physical-normalization factor changes both node/target dimensional scaling, while the descriptor factor independently enables the matched graph-level descriptor head. The original GAT encoder and candidate grid are held fixed. The two compatible V2 cells are reused and the two missing cells are newly trained.

| Normalization | Descriptors | Parent-balanced log RMSE |
|---|---|---:|
| Standard | No | 1.5528 |
| Standard | Yes | 0.8933 |
| Physical | No | 0.9504 |
| Physical | Yes | 0.8311 |

The primary four-cell comparison concerns GAT design choices. Reused normalized descriptor controls are included in the prediction CSV for context. This is development cross-validation, not the separately locked external evaluation.
