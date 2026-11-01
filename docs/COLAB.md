# Training on Google Colab (free T4 GPU)

**Status:** plan, written 2026-11-01. It becomes a ready-to-run notebook (`notebooks/colab_train.ipynb`)
once the first deep-model training script (`src/train_tcn.py`) exists. Steps marked ⚠ get
verified on the first real run and this file is updated.

## What runs where

| Work | Where | Why |
|---|---|---|
| Data loading, features, EDA, XGBoost, XGBoost Optuna (100 trials), ROCKET, API, Docker, drift scripts, tests | **Local CPU** | Fast enough on CPU. XGBoost and ROCKET barely benefit from a GPU here. Docker must run locally anyway. |
| TCN + Bi-LSTM final training, their Optuna studies (50 trials each), any FD002–FD004 deep-model runs | **Colab T4** | Hundreds of epochs × 50 trials would take hours on CPU. |

Both use the **same code**: Colab clones the GitHub repo and runs `python -m src.<script>`.
Nothing is copy-pasted into notebook cells. The scripts pick the device automatically
(`cuda` if available, else `cpu`).

## The workflow, step by step

1. **Locally:** write and test the script on CPU with a tiny run (e.g. 2 epochs, 2 trials) → commit → `git push`.
2. **Colab:** Runtime → Change runtime type → **T4 GPU**.
3. **Mount Google Drive.** Colab's disk is wiped when the session ends, so every output must go to Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   OUT = '/content/drive/MyDrive/pm-outputs'   # persistent
   ```
4. **Get the code:**
   ```python
   !git clone https://github.com/syed-m07/predictive-maintenance.git
   %cd predictive-maintenance
   ```
   ⚠ If the repo is **private**, store a GitHub fine-grained token (read-only, this repo only) in
   Colab's 🔑 *Secrets* panel as `GH_TOKEN`, and clone with
   `https://$GH_TOKEN@github.com/syed-m07/predictive-maintenance.git`. Never paste the token into a cell.
5. **Install dependencies WITHOUT torch.** Colab ships its own CUDA-enabled torch, and our
   `requirements.txt` pins the CPU build, which would replace it and silently disable the GPU:
   ```python
   !grep -v -E "^(torch|--extra-index-url)" requirements.txt > /tmp/req-colab.txt
   !pip install -q -r /tmp/req-colab.txt
   import torch; print(torch.__version__, torch.cuda.is_available())   # must print True
   ```
   ⚠ Colab's Python version and preinstalled packages change over time. If pip reports
   conflicts with our pins, *Runtime → Restart session* once and re-run the cell. Note the
   Colab torch version in the run's MLflow params so results stay traceable.
6. **Get the data** (same Kaggle source and checksums as local):
   ```python
   import kagglehub, shutil, glob
   src = kagglehub.dataset_download('behrad3d/nasa-cmaps') + '/CMaps'
   for f in glob.glob(src + '/*_FD00?.txt'): shutil.copy(f, 'data/raw/')
   !cd data/raw && sha256sum -c ../sha256sums.txt --ignore-missing
   ```
7. **Point MLflow and model outputs at Drive, then train:**
   ```python
   import os
   os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{OUT}/mlflow_colab.db'
   !python -m src.train_tcn_optuna      # 50 trials
   !python -m src.train_tcn             # final model with best params
   !cp -r trained_models/* {OUT}/trained_models/
   ```
8. **Bring results home:** download `pm-outputs/trained_models/*` (weights + scaler + best-params
   JSON) into the local `trained_models/` / `configs/`, and download `mlflow_colab.db`.
   View Colab runs locally with:
   `mlflow ui --backend-store-uri sqlite:///mlflow_colab.db`
   The registry rows (`model_registry.csv`) are appended on Colab and copied back. The CSV is
   committed, so the comparison table always includes the Colab-trained models.
9. **Locally:** run `scripts/compare_models.py`. It evaluates every model on the same FD001 test
   set on CPU, so latency numbers are comparable across models.

## Things to know

- Free Colab sessions disconnect after ~90 min idle / ~12 h max. Optuna studies are stored in a
  SQLite file on Drive, so an interrupted study **resumes** instead of starting over.
- GPU results are not bit-for-bit reproducible (cuDNN nondeterminism). Seeds keep them close; this is
  noted in the README.
- Latency is always measured **locally on CPU**, never on Colab's GPU, so the comparison is fair.
