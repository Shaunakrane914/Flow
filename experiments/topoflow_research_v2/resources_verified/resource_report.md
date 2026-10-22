# Measured compute costs

| Stage | Crops | Median ms | P95 ms |
|---|---:|---:|---:|
| snow2 | 4 | 4057.211 | 4716.774 |
| pnm_solve | 4 | 1.854 | 7.063 |
| graph_conversion | 4 | 0.518 | 0.826 |
| total_extraction_and_reference | 4 | 4059.531 | 4724.664 |
| gat_standard_forward | 24 | 1.878 | 2.928 |
| gat_physics_forward | 24 | 1.719 | 2.673 |
| sage_physics_forward | 24 | 1.098 | 1.611 |
| uniform_physics_forward | 24 | 1.876 | 2.662 |
| pore_set_physics_forward | 24 | 0.436 | 0.650 |

Measurements are serial CPU runs at one Torch/BLAS thread. Extraction uses one fixed crop per natural parent; p95 across four crops is descriptive and unstable. Model forwarding uses 30 repeated calls per held-out crop after 10 warmups. Input file loading and feature normalization are excluded from forward latency. The complete extraction-and-reference row includes an unnecessary PNM label solve for deployment; it is not an end-to-end inference measurement.

Peak process RSS, actual parameter bytes and graph tensor bytes are different quantities. No dense CNN was measured, so these data do not establish a CNN memory advantage.
