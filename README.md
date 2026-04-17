# Calibrating Dataset Inference for Diffusion Models under Realistic Similar Distributions

---

## Abstract
The rapid adoption of pre-trained diffusion models has heightened concerns that these models may have been trained on unlicensed or proprietary datasets. Existing approaches for copyright detection are largely based on Membership Inference Attacks (MIAs). However, due to the nature of training large-scale diffusion models on massive training data, these models retain only a weak memorization of individual samples, rendering typical sample-level MIAs nearly ineffective. To overcome the fundamental limitations of sample-level MIAs, recent studies have focused on Dataset Inference (DI), which leverages aggregated MIA features to improve performance. Yet, such approaches rely on the strict Independent and Identically Distributed (IID) assumption, which is difficult to satisfy in practical copyright detection scenarios due to distribution shifts across detection datasets. To address the challenges of applying DI in real-world scenarios, in this paper, we theoretically break the strict IID assumption and propose a more realistic similar distribution assumption, quantifying the residual distribution shift in both the semantic and textural spaces. Based on this assumption, we propose a calibration DI framework for real-world detections and theoretically bound the false positive rate (FPR) under residual distribution shift. Extensive experiments demonstrate that our method maintains detection efficiency while suppressing the FPR in real-world scenarios.

<img title="" src="imgs/overview.png" alt="" data-align="center">

---

## Getting Started

### Environment

Runtime environment requirements are listed in `src/requirements.txt`. Our experimental setup:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 6000 |
| PyTorch | 2.8.0 |
| Python | 3.12 |
| CUDA | 12.8 |

### Configuration

All experimental parameters are defined in `src/exp_utils/config_util/attack_config.py`.

---

## Experimental Workflow

### 1. Prepare Datasets

Run scripts in `dataset/` to obtain full training datasets and hold-out datasets. Use `dataset/clean_bad_images.py` to filter out corrupted images.

### 2. Mitigate Distribution Shift (CLIP-KNN & SCNS)

After obtaining the datasets, run `src/exp_utils/dataset_process_util/clip_knn.py` to mitigate distribution shift and generate a pair of $D_{\text{sus}}$ and $D_{\text{hold}}$ with similar distributions.

**Evaluate distribution similarity:**

- Semantic similarity: `src/exp_utils/dataset_process_util/seman_bound.py`
- Textural complexity: `src/exp_utils/dataset_process_util/texture_bound.py`

**Generate hold-out subset pairs:**

```bash
python src/exp_utils/dataset_process_util/emp_null_sample.py
```

### 3. Run Dataset Inference

Execute the calibrated dataset inference:

```bash
python src/main_ttest.py
```

Calculate the dynamic calibration threshold and hypothesis testing:

```bash
python src/calculate_threshold_fpr.py
```

**Baseline Comparison:** `src/baseline_ttest.py` implements the standard DI paradigm, and `src/baseline_calculate_fpr.py` evaluates FPR and TPR.

### 4. (Optional) Run Individual MIA

For independent Membership Inference Attacks used to extract memorization features:

```bash
python src/exp_utils/mi_util/baseline_mi_main.py
```

---

## Models & Datasets

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
