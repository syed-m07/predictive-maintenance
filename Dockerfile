# RUL prediction API. Model weights are gitignored: train (or import) the served models first,
# run `python -m scripts.simulate_drift` once (writes the drift reference), then:
#   docker build -t pm-api .
#   docker run --rm -p 8000:8000 pm-api        # -> http://localhost:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# XGBoost's wheel needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements-serve.txt changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Code, serving config and ONLY the shortlisted bundles (configs/serving.json, DV-005).
COPY src/ src/
COPY configs/serving.json configs/serving.json
COPY trained_models/model_registry.csv trained_models/reference_FD001.npz trained_models/
COPY trained_models/lstm_uni_tuned_v1/ trained_models/lstm_uni_tuned_v1/
COPY trained_models/tcn_tuned_v1/ trained_models/tcn_tuned_v1/
COPY trained_models/xgboost_tuned_v1/ trained_models/xgboost_tuned_v1/

# Don't run as root inside the container.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
