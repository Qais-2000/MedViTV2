# Imbalance-Aware Training for Skin Lesion Classification (MedViTV2 + DermaMNIST)

**ELEC-6661 Final Project · Concordia University · Fall 2026**
Muhammad Qais Arshad · Mohammad Mohibul Haque · Ifthakhar Parvez Prothoy

📄 **Project page (report):** https://qais-2000.github.io/MedViTV2/

This repository is a fork of [MedViTV2](https://github.com/Omid-Nejati/MedViTV2) (via [IMPACT-L/MedViTV2](https://github.com/IMPACT-L/MedViTV2)). We ask whether **class-weighted cross-entropy** or **focal loss** helps MedViT-tiny detect rare skin lesion classes on DermaMNIST, and what it costs in precision and overall accuracy.

## Main result (test set, mean ± std over 3 seeds)

| Method | Balanced accuracy | Melanoma recall | Accuracy | Macro F1 |
|---|---|---|---|---|
| Cross-entropy (baseline) | 64.1 ± 2.2% | 40.5 ± 0.5% | 81.0 ± 0.2% | 0.641 ± 0.022 |
| Weighted CE (inverse frequency) | 70.6 ± 2.2% | 57.4 ± 2.9% | 70.3 ± 1.4% | 0.556 ± 0.023 |
| Focal loss (γ = 3) | 62.2 ± 2.9% | 36.3 ± 3.1% | 79.9 ± 0.5% | 0.630 ± 0.026 |

Weighted CE improves rare-class recall but lowers precision and overall accuracy; focal loss alone gives no reliable gain. Full analysis on the project page.

## What we added

| File | Purpose |
|---|---|
| `train.py` | Shared training script. Same model/augmentation/optimizer as the original repo; only the loss changes. Best epoch chosen on the **validation** set; test set used once. Resumes after Colab disconnects. |
| `losses.py` | Class-weighted cross-entropy (inverse / square-root weights) and focal loss. |
| `train_runs.ipynb` | Colab notebook to run one experiment (choose loss, setting and seed). |
| `smoke_test_medvit_dermamnist.ipynb` | Checks the setup and reproduces the released checkpoint (78.25% vs 78.1% reported). |
| `final_results.ipynb` | Builds all tables, figures and example images from the saved runs. |
| `docs/` | GitHub Page (report) and figures. |

## How to reproduce

1. Open `smoke_test_medvit_dermamnist.ipynb` in Google Colab (T4 GPU) to verify the environment.
2. Open `train_runs.ipynb`, run Cells 1–3 (sets up Python 3.11, PyTorch 2.5.0 + CUDA 12.4, NATTEN 0.17.5 and this repository), then in Cell 4 choose:
   - `LOSS = "ce"`, `"weighted"` or `"focal"`
   - `WEIGHTING = "inverse"` or `"sqrt"` (weighted only), `GAMMA = 1, 2, 3` (focal only)
   - `SEED = 0, 1, 2`

   One run is 50 epochs, about 1 h 45 min on a T4. Results are saved to Google Drive.
3. Run `final_results.ipynb` (no GPU) to build the tables and figures.

Runs used in the report: CE seeds 0–2; weighted inverse seeds 0–2 and sqrt seed 0; focal γ = 1, 2, 3 seed 0 and γ = 3 seeds 1–2.

### Environment notes

- Colab's default Python 3.13 cannot install PyTorch 2.5 / NATTEN 0.17, so the notebooks create a separate Python 3.11 environment with `uv`.
- We use `MedViT.py` from upstream commit `4a9e517`; the Oct 23, 2026 upstream update requires NATTEN ≥ 0.20 and no longer matches the released checkpoints. The notebooks check out that version automatically.
- `torchattacks` in `requirements.txt` is not used and is skipped (it pins an old `requests` that breaks Colab).

## Data

DermaMNIST from [MedMNIST v2](https://medmnist.com/) (derived from HAM10000), downloaded automatically by the code. DermaMNIST is licensed **CC BY-NC 4.0**; no dataset files or model checkpoints are stored in this repository.

## Credits

- Original model and code: O. N. Manzari et al., *MedViTV2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention*, Applied Soft Computing, 2025 — [github.com/Omid-Nejati/MedViTV2](https://github.com/Omid-Nejati/MedViTV2) (MIT license, kept in `LICENSE`).
- J. Yang et al., *MedMNIST v2*, Scientific Data, 2023 · P. Tschandl et al., *The HAM10000 dataset*, Scientific Data, 2018 · T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV, 2017.

The original MedViTV2 README is available in the [upstream repository](https://github.com/Omid-Nejati/MedViTV2#readme).
