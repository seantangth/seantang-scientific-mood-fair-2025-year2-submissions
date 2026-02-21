# Code to reproduce V124c weights
# V124c is a variant of the V123 model (Fold 1, no bias correction).
# The V123 training pipeline is fully contained within the Jupyter Notebook provided in the non-submission components section.

import os
import sys

if __name__ == "__main__":
    print("For full reproducible training, please refer to: ../../notebooks/v123_mlp_lpft_kfold.ipynb")
    print("This notebook includes data preprocessing, linear probing, fine-tuning, and PyTorch model tracing.")
