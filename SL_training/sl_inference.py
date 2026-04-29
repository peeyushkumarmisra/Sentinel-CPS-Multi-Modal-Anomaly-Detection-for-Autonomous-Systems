"""
File: sl_inference.py
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

class SLInference:
    """
    Loads the RF model and scaler once at construction.
    Repeated calls to .run() pay zero reload cost.
    """

    FEATURE_KEYS = [
        'voc_emissions', 'acoustic_vibration', 'cpu_utilization', 'mechanical_load',
        'elevation_altitude', 'actuator_torque', 'thermal_load', 'compromise_risk_index'
    ]

    def __init__(self, model_dir: str = "MODELS"):
        model_dir = Path(model_dir)
        self.model  = joblib.load(model_dir / "rf_model.pkl")
        self.scaler = joblib.load(model_dir / "sl_scaler.pkl")
        print(f"SL model and scaler loaded from: {model_dir}")

    def run(self, location_data_100_rows: list) -> tuple:
        """
        Args:
            location_data_100_rows: List of 100 dicts from spawner.get_payload()
        Returns:
            (majority_pred, avg_conf, elapsed_sec)
        """
        t0 = time.perf_counter()

        raw_features   = pd.DataFrame(
            [[row[k] for k in self.FEATURE_KEYS] for row in location_data_100_rows],
            columns=self.FEATURE_KEYS
        )
        scaled         = self.scaler.transform(raw_features)
        preds          = self.model.predict(scaled)
        confs          = np.max(self.model.predict_proba(scaled), axis=1)

        majority_pred  = Counter(preds).most_common(1)[0][0]
        avg_conf       = round(float(np.mean(confs)), 3)
        elapsed        = time.perf_counter() - t0

        print(f"[SL] 100 samples | pred: {majority_pred} | conf: {avg_conf:.3f} | elapsed: {elapsed:.4f}s")
        return majority_pred, avg_conf, elapsed