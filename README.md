# Description
This is the code for Calibrating Dataset Inference for Diffusion Models under Realistic Similar Distributions.


## 📋 Abstract

The rapid adoption of pre-trained diffusion models has heightened concerns that these models may have been trained on unlicensed or proprietary datasets. Existing approaches for copyright detection are largely based on Membership Inference Attacks (MIAs). However, due to the nature of training large-scale diffusion models on massive training data, these models retain only a weak memorization of individual samples, rendering typical sample-level MIAs nearly ineffective. To overcome the fundamental limitations of sample-level MIAs, recent studies have focused on Dataset Inference (DI), which leverages aggregated MIA features to improve performance. Yet, such approaches rely on the strict Independent and Identically Distributed (IID) assumption, which is difficult to satisfy in practical copyright detection scenarios due to distribution shifts across detection datasets. To address the challenges of applying DI in real-world scenarios, in this paper, we theoretically break the strict IID assumption and propose a more realistic similar distribution assumption, quantifying the residual distribution shift in both the semantic and textural spaces. Based on this assumption, we propose a calibration DI framework for real-world detections and theoretically bound the false positive rate (FPR) under residual distribution shift. Extensive experiments demonstrate that our method maintains detection efficiency while suppressing the FPR in real-world scenarios.

<img title="" src="imgs/overview.png" alt="" data-align="center">

---

## 🚀 Getting Started

### ⚙️ Environment

Runtime environment requirements are listed in `src/requirements.txt`. Our experimental setup:

```bash
python -m pip install -r src/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 6000 |
| PyTorch | 2.8.0 |
| Python | 3.12 |
| CUDA | 12.8 |

### 📁 Configuration

All experimental parameters are defined in `src/exp_utils/config_util/attack_config.py`.

---

## 📂 Project Structure

```
LDM-DI/
├── src/
│   ├── run_experiments.sh                      # Full workflow or selected stage
│   ├── main_ttest.py                           # Main DI framework
│   ├── baseline_ttest.py                       # Baseline DI paradigm
│   ├── calculate_threshold_fpr.py              # FPR threshold calibration
│   ├── baseline_calculate_fpr.py               # Baseline FPR evaluation
│   ├── extract_features.py                     # Feature extraction
│   ├── requirements.txt
│   └── exp_utils/
│       ├── config_util/
│       │   ├── attack_config.py
│       │   └── random_provider.py
│       ├── dataset_process_util/
│       │   ├── prepare_experiment_data.py      # Source image manifest
│       │   ├── clip_knn.py                     # CLIP-KNN distribution alignment
│       │   ├── seman_bound.py                  # Semantic similarity bound
│       │   ├── texture_bound.py                # Textural complexity bound
│       │   ├── emp_null_sample.py              # SCNS hold-out generation
│       │   ├── generate_hold_out_datasets.py
│       │   ├── random_select_subset.py
│       │   ├── meta_feature_manager.py
│       │   └── img_preprocess.py
│       ├── mi_util/
│       │   ├── baseline_mi_main.py             # Individual MIA runner
│       │   ├── mia_ablation/                   # Run and export MIA ablation
│       │   ├── piami_src/
│       │   ├── noise_optimmi_src/
│       │   ├── naivelossmi_src/
│       │   ├── structuralmi_src/
│       │   ├── secmi_src/
│       │   ├── rediffusemi_src/
│       │   ├── gradient_maskingmi_src/
│       │   └── multiple_lossmi_src/
│       ├── load_util/
│       │   └── load_data.py
│       ├── metric_util/
│       │   └── evaluate.py
│       └── pipeline_util/
│           ├── txt2img_pipeline_select.py
│           └── img2img_pipeline_select.py
├── dataset/
│   ├── laion-aesthetics_v2_over_4p75/
│   ├── laion2b-multi-translated-hf/
│   ├── coco/
│   ├── commoncatalog/
│   ├── nocaps/
│   └── clean_bad_images.py
└── imgs/
    └── overview.png
```

---

## 🧪 Experimental Workflow

Use the scripts in `dataset/` to obtain the source images and captions, and use `dataset/clean_bad_images.py` to filter corrupted images. Download the model weights listed below and activate the Python environment above. From the repository root, run the full workflow into a fresh output directory or select an individual stage:

```bash
bash src/run_experiments.sh all /path/to/datasets /path/to/models /path/to/new-output
bash src/run_experiments.sh rq1 /path/to/datasets /path/to/models /path/to/existing-output
bash src/run_experiments.sh --help
```

For a single RQ, run `prepare` and `features` first and keep using the same output directory. The script help lists the input layout, additional prerequisites, and commands for similarity measurements, MIA ablation, and plotting.

---

## 🤖 Models & Datasets

### Pre-trained Models

| Model | URL |
|-------|------------|
| Stable Diffusion v1-5 | [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| CommonCanvas-S-C | [CommonCanvas-S-C](https://huggingface.co/common-canvas/CommonCanvas-S-C) |
| Kandinsky 3 | [Kandinsky 3](https://huggingface.co/kandinsky-community/kandinsky-3) |

### Datasets

| Dataset | URL |
|---------|--------|
| LAION Aesthetics v2 4.75 | [LAION Aesthetics v2 4.75](https://huggingface.co/datasets/laion/aesthetics_v2_4.75) |
| LAION-2B Multi Translated | [LAION-2B Multi Translated](https://huggingface.co/datasets/laion/relaion2B-multi-joined-translated-to-en) |
| COCO | [MS COCO](https://cocodataset.org/#download) |
| Common Catalog CC-BY-SA | [Common Catalog CC-BY-SA](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by-sa) |
| NoCaps | [NoCaps](https://huggingface.co/datasets/HuggingFaceM4/NoCaps) |
