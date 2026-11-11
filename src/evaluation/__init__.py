"""Evaluation of de-identification outputs against gold_v2.

Independent of generation and training: this package imports nothing from src.generation.
Protocol and definitions: audit/EVALUATION_PROTOCOL.md. Changing any scoring rule requires a new
EVALUATOR_VERSION. Every run records the version and a hash of this package.
"""
EVALUATOR_VERSION = "1.0.1"   # 1.0.1 changes run provenance only. Scoring is identical to 1.0.0.
HISTORICAL_FINGERPRINTS = {"1.0.0": "8ff823ed86c5525af514b33d212e6fb9e704989038142a8874f2139b8cc6609d"}
