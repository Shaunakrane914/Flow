# Measured compute costs

| Stage | Crops | Median ms | P95 ms |
|---|---:|---:|---:|
| snow2 | 4 | 3997.104 | 4746.570 |
| pnm_solve | 4 | 1.995 | 6.784 |
| graph_conversion | 4 | 0.382 | 0.771 |
| total_extraction_and_reference | 4 | 3999.480 | 4754.125 |
| gat_standard_forward | 24 | 1.993 | 3.048 |
| gat_physics_forward | 24 | 1.863 | 2.723 |
| sage_physics_forward | 24 | 1.087 | 1.702 |
| uniform_physics_forward | 24 | 1.903 | 2.741 |
| pore_set_physics_forward | 24 | 0.437 | 0.668 |

Measurements are serial CPU runs at one Torch/BLAS thread. Extraction uses one fixed crop per natural parent; p95 across four crops is descriptive and unstable. Model forwarding uses 30 repeated calls per held-out crop after 10 warmups. Input file loading and feature normalization are excluded from forward latency. The complete extraction-and-reference row includes an unnecessary PNM label solve for deployment; it is not an end-to-end inference measurement.

Peak process RSS, actual parameter bytes and graph tensor bytes are different quantities. No dense CNN was measured, so these data do not establish a CNN memory advantage.
