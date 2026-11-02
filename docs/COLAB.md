# Training on Google Colab (free T4 GPU)

**Status (2026-11-02):** ready. Run **`notebooks/colab_train.ipynb`** in Colab. It performs every step
below, in order, and stops early if the GPU is missing. Bring the results home with
`scripts/import_colab_outputs.py` (step 8). The local parts (training scripts, tuning, the import
script) are tested. Steps marked ⚠ depend on Colab's environment and get confirmed on the first real run.

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
8. **Bring results home:** the notebook's last cell writes `pm-outputs/colab_outputs_<time>.zip`
   (new model bundles, `configs/tcn*.json` / `lstm*.json`, their `results/` figures, and ONLY the
   new registry rows). Download it, then in the local repo run
   `venv\Scripts\python scripts\import_colab_outputs.py path\to\colab_outputs_<time>.zip`.
   It extracts the files and **appends** the rows. It never overwrites the local registry, and it
   refuses paths outside trained_models/configs/results, existing bundles, and duplicate rows.
   Colab's MLflow runs stay in `pm-outputs/mlflow_colab.db`
   (`mlflow ui --backend-store-uri sqlite:///mlflow_colab.db` shows their params and metrics; their
   artifact links point to Drive paths, so the figures are viewed from `results/` instead).
9. **Locally:** run `scripts/compare_models.py`. It evaluates every model on the same FD001 test
   set on CPU, so latency numbers are comparable across models.

## Things to know

- Free Colab sessions disconnect after ~90 min idle / ~12 h max. Optuna studies are stored in a
  SQLite file on Drive, so an interrupted study **resumes** instead of starting over.
- GPU reproducibility, measured: on 2026-11-02 all four LSTM models were trained twice, on two
  different Colab machines (after a reconnect), and gave **identical** test metrics to 4 decimals
  (EXPERIMENTS E4). cuDNN can in principle be nondeterministic, so treat "identical" as observed,
  not guaranteed.
- **After a disconnect / "No module named 'src'":** the Python session was reset and the working
  folder is no longer the repo. Trained bundles live on the Colab machine's disk (only MLflow and
  Optuna databases are on Drive), so **check before re-running step 3** (step 3 starts with
  `rm -rf predictive-maintenance`):
  `!ls /content/predictive-maintenance/trained_models`
  - Folder exists → do NOT re-run step 3. Run step 2, `%cd /content/predictive-maintenance`, set the
    step-6 environment variables again with `SESSION_START = README.md's mtime + 1` and
    `rows_before = <rows the clone came with>`, then continue with the unfinished cell and step 9.
  - "No such file" → new machine: run everything from the top. Finished studies print
    `50 trials done, running 0 more` and only retrain the tuned model; default runs are retrained
    (seconds). In the Colab MLflow store, tag the duplicate runs `duplicate_of=<original run id>`.
  - Download every `colab_outputs_*.zip` the session produced; one may be from before the
    disconnect. Import only one copy of each model.
- Latency: training scripts time inference on CPU, but on Colab that is *Colab's* CPU. The final
  comparison (`scripts/compare_models.py`, Block 10) **re-measures every bundle's latency on the
  local machine**, so all models are timed on identical hardware (V4).
