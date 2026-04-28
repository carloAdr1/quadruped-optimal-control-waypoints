# Controller Comparison Results

| controller   | trajectory   | disturbance   |   velocity_rmse |   xy_rmse |   xy_max_error |   final_xy_error |   mean_grf_norm |   resets |
|:-------------|:-------------|:--------------|----------------:|----------:|---------------:|-----------------:|----------------:|---------:|
| pmp          | line         | none          |        0.420794 |  0.152322 |       0.57585  |         0.156291 |        130.895  |        3 |
| mpc          | line         | none          |        0.40281  |  0.160332 |       0.550124 |         0.178443 |         58.2157 |        3 |
| lqg          | line         | none          |        0.196846 |  0.21015  |       0.555694 |         0.555694 |         29.5566 |        0 |
