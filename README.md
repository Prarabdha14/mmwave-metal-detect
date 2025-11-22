# mmWave Radar Metal Detection – Assignment Submission

This project implements a complete radar-processing pipeline using simulated Infineon 60GHz mmWave radar data.

## Contents
- **notebooks/**
  - `1_simulation.ipynb` → FMCW radar simulation
  - `2_dataset_and_training.ipynb` → feature extraction + PCA + SVM training
  - `3_detection_pipeline.ipynb` → detection pipeline + inference demo
- **simulate/** → signal-processing & FFT utilities
- **deploy/** → Flask inference API (`app.py`)
- **models/** → trained PCA/SVM
- **results/** → detection JSON + annotated RD map

## Quick Demo
### Start API
