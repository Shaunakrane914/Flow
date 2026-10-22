# Paper comparison table

Metrics use natural rocks only: four parent scans and 24 accepted crops. Each crop appears in exactly one outer test fold. Neural settings and epoch counts are selected inside the training parents. All labels are PNM approximations.

| Model | Parent RMSE mean +/- seed SD | Log MSE | Log RMSE | Log MAE | Log R2 | Linear R2 | Factor-two accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Porosity KC | 0.7525 +/- 0.0000 | 0.4887 | 0.6990 | 0.5254 | 0.2537 | -3.3437 | 37.5% |
| Physics ridge | 0.7881 +/- 0.0000 | 0.6514 | 0.8071 | 0.6883 | 0.0051 | -11.6206 | 16.7% |
| Diameter-scaled KC | 0.8133 +/- 0.0000 | 0.6181 | 0.7862 | 0.5652 | 0.0560 | -15.5409 | 54.2% |
| GAT physics + geometry | 0.8311 +/- 0.0200 | 0.7207 | 0.8488 | 0.7207 | -0.1007 | -11.3928 | 22.5% |
| ExtraTrees descriptors | 0.8327 +/- 0.0124 | 0.7270 | 0.8525 | 0.7521 | -0.1104 | -0.1541 | 12.5% |
| Pore-set + geometry | 0.8462 +/- 0.0856 | 0.7232 | 0.8474 | 0.6654 | -0.1044 | -17.0516 | 32.5% |
| Uniform GAT + geometry | 0.8659 +/- 0.1314 | 0.6680 | 0.8137 | 0.6377 | -0.0202 | -1.0120 | 30.8% |
| SAGE physics + geometry | 0.8713 +/- 0.0781 | 0.7760 | 0.8782 | 0.7361 | -0.1851 | -4.9386 | 20.0% |
| Train mean | 0.9224 +/- 0.0000 | 0.7722 | 0.8787 | 0.6762 | -0.1793 | -0.1775 | 20.8% |
| Inner-selected model | 0.9943 +/- 0.1069 | 0.9601 | 0.9767 | 0.8387 | -0.4663 | -17.3654 | 17.5% |
| Inner-selected GAT | 1.4565 +/- 0.0825 | 2.3957 | 1.5462 | 1.2633 | -2.6589 | -11.5479 | 13.3% |
| GAT standardized | 1.5528 +/- 0.0393 | 2.6153 | 1.6167 | 1.3305 | -2.9942 | -0.2249 | 16.7% |
