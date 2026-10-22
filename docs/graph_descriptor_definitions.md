# TopoFlow graph descriptor definitions

Checked against the bulk benchmark implementation on 23 October 2026.
The manuscript describes both vectors because the neural models and ExtraTrees
do not use the same preprocessing.

Let φ denote image porosity, d pore equivalent diameter, V pore volume, Np the
number of graph pores, Ne the number of directed graph edges, and D pore degree.
All logarithms below are base ten. Means and standard deviations are population
statistics across graph pores. Each physical throat contributes two directed
edges, so Ne is twice the number of represented undirected throats. Artificial
boundary pores enter the hydraulic reference but are excluded from these graphs.

| Index | Raw descriptor | Transformed geometry descriptor |
| --- | --- | --- |
| 1 | φ | log φ |
| 2 | mean(log d) | log(1 − φ) |
| 3 | std(log d) | std(log d) |
| 4 | mean(log V) | mean(log V) − 3 log d₀ |
| 5 | std(log V) | std(log V) |
| 6 | log Np | log Np |
| 7 | log Ne | log Ne |
| 8 | mean D | mean D |
| 9 | std D | std D |
| 10 | std(d/2) / mean(d/2) | std(d/2) / mean(d/2) |

The implementation computes log d₀ as the mean of the canonical log-diameter
node feature. Decoding and float32 storage can introduce rounding differences
between that value and the raw descriptor's mean(log d).

ExtraTrees uses the raw vector. Physics ridge uses the transformed vector with
an offset of 2 log d₀ subtracted from the target. Neural descriptor branches use
the transformed vector even when node and target scale normalization is disabled.
Feature and target standardization are fitted only on the training partition.

The tenth feature is pore-radius CV, not degree CV. The two porosity transforms
are deliberate and belong in the description of the transformed vector.

Implementation sources:

- [Raw features](../src/experiment_data.py)
- [Transformed features and neural preprocessing](../src/research_evaluation.py)
- [Ridge and ExtraTrees inputs](../scripts/run_research_v2.py)
- [Graph extraction and exclusion of artificial boundary pores](../src/research_pnm.py)
- [Frozen follow-up sources](../experiments/topoflow_followup/source_snapshot/src/research_evaluation.py)
