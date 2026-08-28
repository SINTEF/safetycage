# Method comparison

| metric | 03-cifar10-cnn / RED | 03-cifar10-cnn / SPARDACUS | 03-cifar10-cnn / RandomBaseline |
| --- | --- | --- | --- |
| threshold | 0.7481 | 0.3578 | 3.4868 |
| auroc | 0.8163 | 0.7851 | 0.5241 |
| test_samples | 400 | 1000 | 1000 |
| total_errors | 115 | 226 | 226 |
| flagged | 160 | 429 | 791 |
| errors_caught | 85 | 177 | 189 |
| errors_missed | 30 | 49 | 37 |
| coverage | 0.6000 | 0.5710 | 0.2090 |
| baseline_accuracy | 0.7125 |  |  |
| baseline_precision_macro | 0.7284 |  |  |
| baseline_recall_macro | 0.7151 |  |  |
| baseline_f1_macro | 0.7169 |  |  |
| baseline_MCC | 0.6810 |  |  |
| cage_kept_accuracy | 0.8750 | 0.9142 | 0.8230 |
| cage_kept_precision_macro | 0.8637 | 0.9097 | 0.8230 |
| cage_kept_recall_macro | 0.8571 | 0.9075 | 0.8305 |
| cage_kept_f1_macro | 0.8571 | 0.9079 | 0.8180 |
| cage_kept_MCC | 0.8613 | 0.9045 | 0.8051 |
| with_review_accuracy | 0.9250 | 0.9510 | 0.9630 |
| with_review_precision_macro | 0.9256 | 0.9514 | 0.9634 |
| with_review_recall_macro | 0.9223 | 0.9511 | 0.9628 |
| with_review_f1_macro | 0.9226 | 0.9510 | 0.9627 |
| with_review_MCC | 0.9168 | 0.9456 | 0.9590 |
| no_cage_accuracy |  | 0.7740 | 0.7740 |
| no_cage_precision_macro |  | 0.7750 | 0.7750 |
| no_cage_recall_macro |  | 0.7731 | 0.7731 |
| no_cage_f1_macro |  | 0.7722 | 0.7722 |
| no_cage_MCC |  | 0.7492 | 0.7492 |
