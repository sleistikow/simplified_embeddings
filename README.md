# Improving Comparability of Temporal Evolution in 2D Embeddings of Ensemble Data

This repository contains the source code associated with our paper:

**"Improving Comparability of Temporal Evolution in 2D Embeddings of Ensemble Data"**  
Simon Leistikow, Vladimir Molchanov, and Lars Linsen

![illustration](https://github.com/user-attachments/assets/98427213-efc4-4341-9220-fd5f68c00250)

The code is licensed under the MIT license.
Copyright (c) 2025 Simon Leistikow

## Overview

We present a method to enhance the comparability of time-evolving ensemble data when visualized in 2D embeddings. The repository includes preprocessing scripts, a demonstration application, and required dependencies to reproduce our results.

## Getting Started

To run the demo locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/sleistikow/simplified_embeddings.git
cd simplified_embeddings
```

### 2. Set up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Preprocess the Datasets

```
python data-preprocessing.py
```

If you want to adjust the data (e.g. the temporal resampling) or want to add an own dataset, this is the file to look for.


### 4. Run the Demo Application

```
python application.py
```

## Citation

If you use this code or refer to our work, please cite the following paper:
```bibtex
# will be added as soon as published
```

## Acknowledgements

This work was funded by the Deutsche Forschungsgemeinschaft (DFG):
- Grant LI 1530/28-1 — 468824876
- Grant MO 3050/2-3 — 360330772

