"""Predictive maintenance pipeline package (NASA C-MAPSS RUL)."""
import os

# Runs before any src.* module (Python executes a package's __init__ first), so this takes
# effect even in scripts that import mlflow before src.config. Silences an informational banner.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
