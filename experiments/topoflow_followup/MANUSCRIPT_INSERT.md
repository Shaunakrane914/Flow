# Throat-aware intervention study

## Abstract

We investigate whether pore-network graph models predict permeability changes
caused by localized throat restrictions. An existing node-only graph attention
network is invariant to throat narrowing when pore features and adjacency remain
fixed. Controlled pore-network simulations confirm that this invariance misses
substantial permeability changes. We introduce an edge-aware prototype with a
learned hydraulic-importance head, auxiliary dissipation supervision, and a
bounded single-throat response readout. The pilot uses eight development graphs
from four natural scans and four external graphs from four additional sandstone
scans. Across five training seeds, the guided GAT predicts fractional
permeability loss with a mean absolute error of 4.57 percentage points, compared
with 4.89 for unguided edge-aware GAT and 6.08 for a no-response control. Its mean
importance-ranking correlation with pore-network sensitivity is 0.422. These
results characterize a limited intervention pilot under a cylindrical-throat
pore-network approximation; they do not establish real-clogging accuracy or
population-level architectural superiority.

## Method

For a fixed-pressure linear hydraulic network, the sensitivity of effective
permeability to a throat conductance is its normalized hydraulic dissipation:

\[
s_e=\frac{\partial\log k}{\partial\log g_e}
=\frac{g_e(p_u-p_v)^2}{Q\Delta P}.
\]

The dissipation fractions sum to one over the full network. We verify this
identity numerically and check the derivative against finite differences.
Interventions change one throat diameter while holding its length fixed. A
diameter multiplier \(r_d\) gives a conductance multiplier \(r_g=r_d^4\).
Full blockage is represented by throat deletion; a nonpercolating outcome has
zero permeability rather than an arbitrary finite logarithmic label.

The edge-aware prototype receives dimensionless pore and throat geometry,
coordinates, inlet/outlet flags and connectivity. A graph encoder produces node
representations, and a symmetric edge head predicts a global hydraulic
importance distribution \(\hat{s}_e\) and a bounded leverage parameter
\(\hat{\ell}_e\). We impose \(\hat{s}_e\geq0\),
\(\sum_e\hat{s}_e=1\), and
\(0\leq\hat{\ell}_e\leq1-\hat{s}_e\). Predicted fractional permeability
loss for one restricted throat is

\[
\widehat{1-k'/k}
=-\frac{(r_g-1)\hat{s}_e}
 {1+(r_g-1)\hat{\ell}_e}.
\]

With exact sensitivity and Dirichlet-network leverage, this readout follows
from a rank-one update of the linear network system. We verify the finite-change
identity against fresh solves before constructing training labels. The identity
is used as a known physical constraint, not presented as a new mathematical
result. The guided model additionally receives the PNM dissipation distribution
as a training target. Pressures, flows, sensitivities and reference permeability
are absent from the model's prediction inputs. The global importance head is
distinct from neighborhood-normalized GAT attention coefficients.

## Data and evaluation

The broader permeability experiment contains 24 development crops from Berea,
C1, C2 and Estaillades, followed by a locked external test containing 32 crops
from S1, S3, S5 and S9. All scans come from the official Imperial College
collection. The four-case GAT ablation independently varies physical
normalization and graph descriptors, with the same encoder and nested parent
selection protocol.

The intervention prototype uses two predeclared accepted crops from each
development scan and one from each external scan. Within each graph, three
highest-dissipation and three other random internal throats are evaluated at
10%, 25% and 50% diameter reduction. All derived restrictions remain grouped
with their parent scan. A single fixed optimizer configuration is used, epochs
are selected on development parents only, and five final training seeds are
reported. Four external graphs remain four test specimens regardless of the
number of throats, doses or seeds.

## Results

The original GAT formulations predict exactly zero change for partial throat
restrictions because those interventions alter none of their inputs. At 50%
diameter reduction, the mean permeability losses over the three
highest-dissipation throats are 13.7%, 44.5%, 42.9% and 29.9% in the selected
Berea, C1, C2 and Estaillades graphs, respectively.

| Prototype | External response MAE in percentage points | Seed SD in points | Mean importance Spearman correlation |
|---|---:|---:|---:|
| Guided edge-aware GAT | 4.573 | 0.377 | 0.422 |
| Unguided edge-aware GAT | 4.888 | 0.208 | 0.209 |
| Edge-aware mean aggregator | 5.003 | 0.138 | 0.301 |
| Guided edge-aware mean aggregator* | 5.033 | 0.389 | 0.363 |
| Edge-only MLP | 5.287 | 0.217 | 0.323 |
| Guided edge-only MLP* | 4.840 | 0.125 | 0.291 |
| No-response control | 6.078 | 0.000 | Undefined |
| Conductance-proxy control | 6.105 | 0.000 | 0.089 |

The guided GAT reduces mean absolute response error by 24.8% relative to the
no-response control. Its improvement over unguided GAT is small and varies by
parent; the exploratory exact parent sign-flip comparison gives p=0.625. The
no-response control is better on the randomly chosen low-impact throats, so the
overall improvement should not be interpreted as uniform superiority across
intervention regimes. Seed standard deviation measures optimization variation,
not uncertainty over independent rocks.

*The two equally guided controls were added after inspecting the first
external prototype results to examine whether supervision, rather than graph
attention, explained the improvement. They are post hoc exploratory controls.
The original guided-GAT model settings were not retuned on those external
responses.*

## Scope and relation to prior work

The intervention targets come from one fixed cylindrical-throat PNM closure.
Complete-image consistency checks on S1 and S5 yield PNM values approximately
one quarter of the published directional image-reference values; unmatched
boundary conventions prevent interpreting these as matched DNS validation.
No published full-image permeability is assigned to a crop, and no new LBM or
physical clogging experiment is claimed.

[RTG-GNN](https://doi.org/10.1016/j.geoen.2024.213358) already addresses
pore/throat topology in permeability prediction.
[Zhao and Xiao](https://doi.org/10.1016/j.jhydrol.2026.135602) already study
learned conductances and physically consistent geometric sensitivities.
Accordingly, the investigated contribution is the controlled intervention
evaluation and the geometry-only learned importance readout, rather than edge
features or sensitivity analysis alone. The current pilot does not establish
that native attention coefficients explain hydraulic control or that the
method is the first of its kind.
