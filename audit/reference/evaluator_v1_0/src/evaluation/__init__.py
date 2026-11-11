"""Evaluation of de-identification outputs against gold_v2.

Independent of generation and training: this package imports nothing from src.generation.
Protocol and definitions: audit/EVALUATION_PROTOCOL.md. Changing any scoring rule requires a new
EVALUATOR_VERSION. Every run records the version and a hash of this package.
"""
EVALUATOR_VERSION = "1.0.0"
