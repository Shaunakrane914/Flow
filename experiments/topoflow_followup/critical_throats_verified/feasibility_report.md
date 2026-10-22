# Critical throat feasibility experiment

The existing GAT is exactly insensitive to partial throat diameter changes because those changes alter none of its inputs. The same limitation holds for the V2 physics-and-descriptor GAT. Full throat removal changes connectivity and is evaluated separately.

One predeclared accepted crop per natural parent is tested. Within each graph, three high-dissipation and three other random internal throats are restricted by 10%, 25%, 50% and 100%. The 100% intervention deletes a throat; nonpercolating outcomes have k=0 and no log label.

PNM hydraulic dissipation fractions sum to one. Analytic conductance sensitivities are checked by finite differences, all saved base labels are checked against fresh solves, and effective permeability is checked to be nonincreasing under conductance restrictions.

| Parent | Selection | Diameter reduction | Mean permeability loss |
|---|---|---:|---:|
| parent_berea | high_dissipation | 10% | 2.04% |
| parent_berea | high_dissipation | 25% | 5.92% |
| parent_berea | high_dissipation | 50% | 13.70% |
| parent_berea | high_dissipation | 100% | 18.76% |
| parent_berea | random | 10% | 0.00% |
| parent_berea | random | 25% | 0.00% |
| parent_berea | random | 50% | 0.00% |
| parent_berea | random | 100% | 0.00% |
| parent_c1 | high_dissipation | 10% | 7.85% |
| parent_c1 | high_dissipation | 25% | 21.67% |
| parent_c1 | high_dissipation | 50% | 44.51% |
| parent_c1 | high_dissipation | 100% | 55.19% |
| parent_c1 | random | 10% | 0.00% |
| parent_c1 | random | 25% | 0.00% |
| parent_c1 | random | 50% | 0.00% |
| parent_c1 | random | 100% | 0.00% |
| parent_c2 | high_dissipation | 10% | 11.99% |
| parent_c2 | high_dissipation | 25% | 27.54% |
| parent_c2 | high_dissipation | 50% | 42.95% |
| parent_c2 | high_dissipation | 100% | 47.46% |
| parent_c2 | random | 10% | 0.00% |
| parent_c2 | random | 25% | 0.00% |
| parent_c2 | random | 50% | 0.00% |
| parent_c2 | random | 100% | 0.00% |
| parent_estaillades | high_dissipation | 10% | 9.16% |
| parent_estaillades | high_dissipation | 25% | 20.02% |
| parent_estaillades | high_dissipation | 50% | 29.87% |
| parent_estaillades | high_dissipation | 100% | 32.61% |
| parent_estaillades | random | 10% | 0.02% |
| parent_estaillades | random | 25% | 0.03% |
| parent_estaillades | random | 50% | 0.04% |
| parent_estaillades | random | 100% | 0.05% |

Attention rankings compare local GAT coefficients with global PNM sensitivity. These are different normalizations; correlation is an exploratory faithfulness diagnostic. Conductance-only and uniform-incoming rankings are controls. The high-dissipation intervention selector uses a base PNM solve and is an oracle experimental selector, not a solver-free prediction method.

This is a four-graph, four-parent feasibility study. Perturbations do not increase the number of independent specimens, and all physical conclusions are conditional on the PNM closure. No real clogging experiment or voxel-resolved validation was performed.
