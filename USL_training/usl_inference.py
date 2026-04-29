"""
File: usl_inference.py

"""
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Literal

from USL_training.feature_pipeline import FeaturePipeline
from USL_training.gmm import GMMTrainer
from USL_training.dec import DECTrainer


@dataclass
class InferenceResult:
    predictions:   np.ndarray   # Cluster assignment per image
    confidences:   np.ndarray   # Max soft-probability per image
    elapsed_sec:   float        # Wall-clock time for the full call
    pipeline_name: str


class USLInference:
    """
    Loads all saved artefacts once at construction time so that repeated
    calls to .run() pay zero reload cost.

    Args:
        model_dir:     Directory containing all saved .pt / .pkl artefacts.
        pipeline_name: "gmm" or "dec"
        seed:          Controls any stochastic ops inside the feature pipeline.
    """

    _SUPPORTED = ("gmm", "dec")

    def __init__(self, model_dir, pipeline_name: Literal["gmm", "dec"], seed):
        pipeline_name = pipeline_name.lower()
        if pipeline_name not in self._SUPPORTED:
            raise ValueError(f"pipeline_name must be one of {self._SUPPORTED}, got '{pipeline_name}'")

        self.pipeline_name    = pipeline_name
        self.feature_pipeline = FeaturePipeline(model_path=model_dir, seed=seed, is_train=False)
        self.model            = self.load_model(model_dir)

    def load_model(self, model_dir):
        import os
        if self.pipeline_name == "gmm":
            return GMMTrainer(model_path=os.path.join(model_dir, "gmm_model.pkl"), is_train=False)
        else:
            return DECTrainer(model_path=os.path.join(model_dir, "dec_model.pt"), is_train=False)

    def run(self, img_paths):
        """
        Args:
            img_paths: List of absolute paths to input images.
        Returns:
            InferenceResult with predictions, per-image confidences, and wall-clock time.
        """
        if not img_paths:
            raise ValueError("img_paths must not be empty.")
        t0 = time.perf_counter()
        features = self.feature_pipeline.run_feature_pipeline(image_paths=img_paths)
        predictions, confidences, _ = self.model.predict(features)
        elapsed = time.perf_counter() - t0
        print(f"[{self.pipeline_name.upper()}] {len(img_paths)} images | "
              f"elapsed: {elapsed:.4f}s | avg conf: {confidences.mean():.4%}")
        return InferenceResult(
            predictions   = predictions,
            confidences   = confidences,
            elapsed_sec   = elapsed,
            pipeline_name = self.pipeline_name
        )