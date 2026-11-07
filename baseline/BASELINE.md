# Baseline: MedViTV2-tiny + standard cross-entropy (Segment A)

## Locked settings (Segments B and C must use exactly these)

Only the loss function may change between segments. Everything else stays fixed so the comparison is fair.

| Setting | Value |
|---|---|
| Dataset | DermaMNIST (HAM10000), official train / val / test split |
| Test set | 2005 images, 7 classes |
| Model | MedViTV2-tiny (`MedViT_tiny` in config) |
| Epochs | 50 |
| Batch size | 24 |
| Image size | 224 × 224, RGB (`size=224, as_rgb=True`) |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0.05 |
| LR schedule | Cosine annealing to 5e-6, stepped every batch |
| Seeds | 0, 1, 2 |
| Best-epoch rule | Highest validation balanced accuracy |
| Train augmentation | RandomResizedCrop(224), AugMix(alpha=0.4), RandomHorizontalFlip(p=0.4), Normalize(mean 0.5, std 0.5) |
| Val/test transform | Resize(224, 224), Normalize(mean 0.5, std 0.5) |
| Code | `train.py --loss ce --seed {0,1,2}`, repo commit 9d55aa2 |
| GPU | NVIDIA T4 (Google Colab) |
| Time per run | About 2.1 min per epoch, about 105 min per 50-epoch run |

Note: every `config.json` also lists `weighting: inverse` and `gamma: 2.0`. These are only argument defaults: `build_loss()` in `train.py` returns plain `nn.CrossEntropyLoss()` when `--loss ce`, so the baseline is unweighted cross-entropy.

## Test results (mean ± std over 3 seeds)

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± std |
|---|---|---|---|---|
| Accuracy | 0.812 | 0.809 | 0.810 | 0.810 ± 0.002 |
| Balanced accuracy | 0.645 | 0.617 | 0.660 | 0.641 ± 0.022 |
| Macro AUC | 0.950 | 0.953 | 0.955 | 0.953 ± 0.002 |
| Macro F1 | 0.650 | 0.616 | 0.657 | 0.641 ± 0.022 |
| Melanoma recall | 0.399 | 0.408 | 0.408 | 0.405 ± 0.005 |

Best epochs: 48 (seed 0), 47 (seed 1), 44 (seed 2).

## Per-class test results (mean ± std over 3 seeds)

| Class | Test n | Precision | Recall | F1 |
|---|---|---|---|---|
| AKIEC (actinic keratoses) | 66 | 0.570 ± 0.054 | 0.591 ± 0.069 | 0.576 ± 0.015 |
| BCC (basal cell carcinoma) | 103 | 0.573 ± 0.025 | 0.654 ± 0.024 | 0.610 ± 0.008 |
| BKL (benign keratosis) | 220 | 0.646 ± 0.028 | 0.648 ± 0.005 | 0.647 ± 0.012 |
| DF (dermatofibroma) | 23 | 0.462 ± 0.057 | 0.391 ± 0.087 | 0.423 ± 0.075 |
| MEL (melanoma) | 223 | 0.629 ± 0.015 | 0.405 ± 0.005 | 0.493 ± 0.002 |
| NV (melanocytic nevi) | 1341 | 0.892 ± 0.003 | 0.933 ± 0.003 | 0.912 ± 0.002 |
| VASC (vascular lesions) | 29 | 0.797 ± 0.071 | 0.862 ± 0.124 | 0.827 ± 0.089 |

## Key finding (draft text for the GitHub Page)

The baseline reaches 0.81 accuracy and 0.95 macro AUC, but these numbers are carried by the majority class: melanocytic nevi are 67% of the test set and are recognised with 0.93 recall. Balanced accuracy is only 0.64, and melanoma recall is 0.41, so about 6 in 10 melanomas are missed. Across all three seeds, 37–39% of melanomas are predicted as nevi. The high AUC shows the model does separate melanoma from other classes to some degree; the errors come mainly from class imbalance pushing decisions toward nevi. This is the problem that weighted cross-entropy (Segment B) and focal loss (Segment C) are meant to address.

## Files

- `overall_test_metrics.csv`, `per_class_test_metrics.csv`: the two tables above
- `confusion_matrix_seed1.png`: test confusion matrix for seed 1 (counts and row percentages)
- `validation_curves.png`: validation balanced accuracy and melanoma recall per epoch, all seeds
