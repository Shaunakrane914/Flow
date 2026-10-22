# Locked external scan evaluation

The model is fitted to all 24 development crops from Berea, C1, C2 and Estaillades.
Hyperparameters and epochs are selected by leaving out development parents only.
The four additional scans S1, S3, S5 and S9 supply the locked external test crops.
Synthetic media are excluded. See metrics.csv and per_parent_metrics.csv for the
complete results, including the fixed GAT and development-selected strategies.

The external scans are four named sandstone datasets from one public collection.
Their geological specimen independence has not been independently certified.
All labels remain the same cylindrical-throat PNM approximation; external scan
validation does not independently validate that physical approximation. Published
full-volume references are recorded separately and are not crop labels.
