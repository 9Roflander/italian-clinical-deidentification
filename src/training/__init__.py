"""Fold-specific datasets, structured targets, training and inference.

Modules import torch, transformers, peft and bitsandbytes only inside the functions that need them, so datasets, targets,
prediction parsing and their tests run anywhere. Real training requires CUDA. There is no Apple-specific path.
"""
TRAINING_PIPELINE_VERSION = "0.1.0"
