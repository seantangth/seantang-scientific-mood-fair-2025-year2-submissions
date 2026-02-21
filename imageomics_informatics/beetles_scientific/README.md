# NSF Beetles Hackathon 2026 - Final Submission (V124c)

This repository contains the code and model weights for our final submission to the 2026 HDR Scientific Mood Challenge (Modeling out of distribution).

## Repository Structure

```
.
└── imageomics_informatics/
    └── beetles_scientific/
        ├── notebooks/
        │   └── v123_mlp_lpft_kfold.ipynb# Full training pipeline (Linear Probing + Fine Tuning)
        ├── submission/
        │   ├── mlp_lpft_fold1_fp16.pt   # Model weights (TorchScript JIT)
        │   ├── model.py                 # Inference script required by Codabench
        │   ├── train.py                 # Reference to notebook
        │   └── requirements.txt         # Dependencies
        ├── training/
        │   └── v124c/
        │       └── train.py             # Reference to notebook
        ├── .gitignore
        └── README.md
```

## Model Architecture
This SOTA submission (V124c) is an MLP-based Linear Probing followed by Fine-Tuning (LP-FT) approach based on the BioCLIP backbone. Note that we consciously opted **not** to use bias correction, as our experiments showed it harmed out-of-domain (OOD) generalization for strong, fine-tuned features.

## Installation & Running

The evaluation on Codabench runs within their standard environment, but you can test the code locally.

1. Create a virtual environment:
   ```bash
   conda create -n beetles-submission python=3.10 -y
   conda activate beetles-submission
   cd imageomics_informatics/beetles_scientific
   pip install -r submission/requirements.txt
   ```

2. Test the predictor (ensure you have sample images):
   ```python
   from submission.model import Model
   
   model = Model()
   # Expected input format is a list containing dicts with `file_path`
   prediction = model.predict([{"file_path": "path_to_sample.jpg"}])
   print(prediction)
   ```

## Performance
* **Overall RMS (CRPS)**: 0.5866
* **In-Domain SPEI_2y**: 0.6229
* **Out-of-Domain SPEI_2y**: 0.5328
