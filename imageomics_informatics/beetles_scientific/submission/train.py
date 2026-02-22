# Training reference for V124c MLP LP-FT model
#
# The V124c model was trained using a 5-fold cross-validation pipeline
# with BioCLIP backbone and MLP regressor head (Linear Probing + Fine-Tuning).
#
# To reproduce the training:
#   1. See training/v124c/train_kfold.py for the K-Fold CV training script
#   2. See training/v124c/train.py for the base training loop
#   3. See notebooks/v123_mlp_lpft_kfold.ipynb for the full documented pipeline
#
# Training command (example):
#   cd training/v124c
#   python train_kfold.py --backbone bioclip --unfreeze --k_folds 5 \
#       --epochs 50 --lr 1e-5 --batch_size 32 --exp_version v124c
#
# After training, convert the best fold checkpoint to TorchScript:
#   See notebooks/v123_mlp_lpft_kfold.ipynb for JIT tracing + FP16 conversion

if __name__ == "__main__":
    print("For full reproducible training, see:")
    print("  - training/v124c/train_kfold.py (K-Fold CV script)")
    print("  - training/v124c/train.py (base training loop)")
    print("  - notebooks/v123_mlp_lpft_kfold.ipynb (documented pipeline)")
