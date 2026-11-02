# Urban Land Cover Mapping via Deep Semantic Segmentation of Aerial Imagery

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-GPU%20Enabled-F9AB00.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An academic, production-grade deep learning research project benchmarking semantic segmentation architectures for automated land cover classification from high-resolution aerial imagery using the **LandCover.ai** benchmark.

---

## 1. Project Overview

Accurate urban land cover mapping from aerial remote sensing imagery is vital for urban planning, environmental monitoring, disaster management, and infrastructure development. This project investigates and benchmarks deep convolutional semantic segmentation architectures:
* **Model A:** UNet + ResNet-34 Encoder (ImageNet Pretrained)
* **Model B:** DeepLabV3+ + ResNet-34 Encoder (ImageNet Pretrained)

### Semantic Classes & Color Mapping
The dataset adheres to the official LandCover.ai five-class taxonomy:
| Class ID | Semantic Class | Hex Color | RGB Values | Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Background** | `#000000` | `(0, 0, 0)` | Unlabeled terrain, fields, bare soil |
| **1** | **Building** | `#FF0000` | `(255, 0, 0)` | Residential, commercial, and industrial structures |
| **2** | **Woodland** | `#00FF00` | `(0, 255, 0)` | Forested and wooded regions (*proxy for vegetation*) |
| **3** | **Water** | `#0000FF` | `(0, 0, 255)` | Rivers, lakes, ponds, and canals |
| **4** | **Road** | `#FFFF00` | `(255, 255, 0)` | Paved and unpaved transportation corridors |

> [!NOTE]
> **Dataset Specification on Vegetation:** LandCover.ai does not annotate low vegetation or agricultural grass as a distinct category. The **Woodland** class represents all tree cover and forested areas and serves as the closest vegetation proxy in this benchmark.

---

## 2. Repository Structure

```
landcover-segmentation/
│
├── notebooks/
│   └── urban_land_cover_segmentation.ipynb   # Main Google Colab execution notebook
│
├── configs/
│   └── config.yaml                           # Master configuration file
│
├── src/
│   ├── __init__.py                           # Module package init
│   ├── utils.py                              # Logging, seeds, palette mapping, overlays
│   ├── dataset.py                            # LandCoverDataset and DataLoader helpers
│   └── prepare_data.py                       # Kaggle download, partitioning, tiling pipeline
│
├── tests/
│   └── test_stage1.py                        # Automated unit test suite
│
├── data/                                     # Generated dataset (ignored by git)
│   ├── train/ (images/ & masks/)
│   ├── val/   (images/ & masks/)
│   └── test/  (images/ & masks/)
│
├── results/                                  # Statistical outputs, plots, metrics
│   ├── class_distribution.json
│   ├── dataset_statistics.json
│   └── source_image_split_assignment.json
│
├── checkpoints/                              # Model weight checkpoints (ignored by git)
├── requirements.txt                          # Python dependencies
├── PROJECT_PROGRESS.md                       # Real-time multi-stage tracker
├── .gitignore                                # Excludes large datasets, models, credentials
└── README.md                                 # Project documentation
```

---

## 3. Data Preparation & Leakage Prevention

### Preventing Spatial Data Leakage
The raw LandCover.ai dataset comprises 41 massive orthophotos (33 at 25 cm/pixel resolution and 8 at 50 cm/pixel resolution) spanning 216.27 km² in Poland. 

In naive preprocessing pipelines, patches are randomly split into train/val/test after slicing. This causes **severe spatial data leakage**: adjoining 512×512 tiles from the exact same aerial flight path share identical atmospheric, lighting, and geographic patterns, falsely inflating test scores.

**Our Implementation:**
1. **Source-Image Grouping:** Slicing assigns all patches from any parent orthophoto strictly to a single subset.
2. **Clean Partitioning:** 41 orthophotos are partitioned into Train (70%), Validation (15%), and Test (15%) subsets with zero overlap:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. **Edge Discarding:** Partial edge tiles smaller than 512 × 512 are discarded to avoid zero-padding border distortions.
4. **Integer Mask Integrity:** Masks are saved as lossless 8-bit PNGs with exact class values in $\{0, 1, 2, 3, 4\}$, never normalized or interpolated with bilinear filtering.

---

## 4. Quick Start & Execution

### Local Installation & Unit Testing
```bash
# Clone the repository
git clone https://github.com/<your-username>/landcover-segmentation.git
cd landcover-segmentation

# Install dependencies
pip install -r requirements.txt

# Run complete automated unit test suite (Stages 1, 2 & 3)
python3 -m unittest discover -s tests -p "test_*.py"

# Optional: Run command-line model training
python3 src/train.py --model both --epochs 40 --batch_size 8
```

### Google Colab Execution
1. Open `notebooks/urban_land_cover_segmentation.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime: **Runtime > Change runtime type > T4 GPU**.
3. Configure Kaggle authentication securely:
   * In Colab's left sidebar, click the **Secrets (Key icon)**.
   * Add `KAGGLE_USERNAME` and `KAGGLE_KEY`.
   * Enable notebook access for both secrets.
4. Run all cells sequentially.

---

## 5. Google Colab Recovery & Disconnection Guide

Google Colab free-tier runtimes reset when idle or when exceeding session limits, clearing ephemeral `/content/` storage. This project includes an automated resilience engine so you **never lose training progress or re-download datasets**:

### 3-Step Recovery Workflow After Any Disconnection:
1. **Reconnect Runtime:** Reconnect to Colab and ensure **T4 GPU** is selected (*Runtime > Change runtime type*).
2. **Mount Google Drive:** Run Cell 4 and grant authorization when prompted.
3. **Execute Cell 2.1 (Master Recovery Cell):**
   * Automatically restores `src/` and `configs/` from persistent Drive storage.
   * Detects the processed dataset in Google Drive (*never re-downloads or re-tiles*).
   * Locates your latest training checkpoints (`unet_latest.pt`, `deeplabv3plus_latest.pt`).
   * Displays the **Recovery Dashboard** indicating your exact epoch and validation mIoU.
4. **Resume Training:** Jump straight to Section 19 (UNet) or Section 20 (DeepLabV3+) to resume training from the exact saved epoch!

---

## 6. Development Roadmap

- [x] **Stage 1: Google Colab Setup and Data Preparation** *(Completed)*
- [x] **Stage 2: Augmentations and Model Architectures** *(Completed)*
- [x] **Stage 3: Training Pipeline & Colab Resilience** *(Completed)*
- [ ] **Stage 4: Model Evaluation and Comparison** *(Pending user command `NEXT`)*
- [ ] **Stage 5: Visualizations**
- [ ] **Stage 6: Final Report and Documentation**

