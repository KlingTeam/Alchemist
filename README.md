# Alchemist: nlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection
[**Project**](https://peterwang512.github.io/FastGDA/) | [**Paper**](https://www.arxiv.org/abs/2511.10721)

[Kaixin Ding](https://peterwang512.github.io/)<sup>1</sup>, [Yang Zhou](https://www.dgp.toronto.edu/~hertzman/)<sup>2</sup>, [Xi Chen](https://people.eecs.berkeley.edu/~efros/)<sup>1</sup>, [Miao Yang](http://richzhang.github.io/)<sup>3</sup>, [Jiarong Ou](https://cs.cmu.edu/~junyanz)<sup>3</sup>, [Rui Chen](https://cs.cmu.edu/~junyanz)<sup>3</sup>, [Xin Tao](https://cs.cmu.edu/~junyanz)<sup>3</sup>, [Hengshuang Zhao](https://cs.cmu.edu/~junyanz)<sup>1</sup>.
<br> HKU<sup>1</sup>, SCUT<sup>2</sup>, Kuaishou Technology, Kling team<sup>3</sup>


<p align="center">
<img src="assets/teaser.jpg" width="800px"/>
</p>


---

### Abstract

Recent advances in Text-to-Image (T2I) generative models, such as Imagen, Stable Diffusion, and FLUX, have led to remarkable improvements in visual quality. However, their performance is fundamentally limited by the quality of training data. Web-crawled and synthetic image datasets often contain low-quality or redundant samples, which lead to degraded visual fidelity, unstable training, and inefficient computation. Hence, effective data selection is crucial for improving data efficiency. Existing approaches rely on costly manual curation or heuristic scoring based on single-dimensional features in Text-to-Image data filtering. Although meta-learning based method has been explored in LLM, there is no adaptation for image modalities. To this end, we propose Alchemist, a meta-gradient-based framework to select a suitable subset from large-scale text-image data pairs. Our approach automatically learns to assess the influence of each sample by iteratively optimizing the model from a data-centric perspective. Alchemist consists of two key stages: data rating and data pruning. We train a lightweight rater to estimate each sample's influence based on gradient information, enhanced with multi-granularity perception. We then use the Shift-Gsampling strategy to select informative subsets for efficient model training. Alchemist is the first automatic, scalable, meta-gradient-based data selection framework for Text-to-Image model training. Experiments on both synthetic and web-crawled datasets demonstrate that Alchemist consistently improves visual quality and downstream performance. Training on an Alchemist-selected 50% of the data can outperform training on the full dataset.
## Quick Start

### Environment Setup

Create a conda/micromamba environment with all dependencies:

```bash
# Using conda
conda env create -f environment.yaml
conda activate alchemist

```
### Run Interactive Demo

Launch a Gradio demo to explore image attributions:

```bash
python demo.py \
    --checkpoint weights/dino+clip_text.pth \
    --data_dir data/coco \
    --feature_dir data/coco/feats/dino+clip_text
```

The demo will:
1. Load a generated image and its caption
2. Compute calibrated features using the trained model
3. Rank all training images by influence score
4. Display the top-k most influential training images

## Training the Ranker

### Prerequisites

Before training, you need to run the data download script `scripts/download_data.sh`.



### Train on COCO Dataset

Train a model using DINO + CLIP text features from this script:

```bash

bash scripts/train_coco.sh
```

### Training Arguments

**Dataset Settings:**
- `--ftype`: Feature type (e.g., `dino+clip_text`, `dino`, `clip`)
- `--data_dir`: Directory containing feature files
- `--rank_file`: Path to ground truth influence rankings (.pkl)

**Model Architecture:**
- `--hidden_sizes`: Hidden layer sizes (default: `[768, 768, 768]`)
- `--input_norm`: Use layer normalization on input
- `--dropout`: Dropout probability (default: `0.1`)
- `--out_feat_dim`: Output feature dimension (default: `768`)

**Training Hyperparameters:**
- `--epochs`: Number of epochs (default: `10`)
- `--batch_size`: Batch size (default: `4096`)
- `--lr`: Learning rate (default: `0.001`)

**Logging:**
- `--wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name (default: `fastgda`)

### Evaluation

Evaluate a trained model on the test set:

```bash
bash scripts/eval_coco.sh
```

This computes **mAP@k** (mean average precision at k) for different values of k, measuring how well the model ranks truly influential training images.

## (Optional) Preprocessing Pipeline

If you want to generate features and influence rankings from scratch, follow these steps:

### Step 1: Download Raw Data

In case you haven't download the data, run `bash scripts/download_data.sh`.

### Step 2: Extract Features

Extract DINO, CLIP, and text features from images:

```bash
cd feature_extraction

# Extract all features (takes ~60-90 minutes on A100)
bash extract_coco.sh

# This will generate:
# - dino features (768-dim)
# - clip features (512-dim)
# - clip_text features (512-dim)
# - dino+clip_text features (1280-dim)
cd ..
```

The features will be stored in `data/coco/feats_test` by default. You can change the output location by specifying `FEAT_DIR` argument in `feature_extraction/extract_coco.sh`.

### Step 3: Compute Ground Truth Influences

Compute expensive ground truth influence scores using AttributeByUnlearning(AbU). See `abu/coco/README.md` for detailed documentation.

## Acknowledgments

We thank Simon Niklaus for the help on the LAION image retrieval. We thank Ruihan Gao, Maxwell Jones, and Gaurav Parmar for helpful discussions and feedback on drafts. Sheng-Yu Wang is supported by the Google PhD Fellowship. The project was partly supported by Adobe Inc., the Packard Fellowship, the IITP grant funded by the Korean Government (MSIT) (No. RS-2024-00457882, National AI Research Lab Project), NSF IIS-2239076, and NSF ISS-2403303.

## Citation

If you use FastGDA in your research, please cite:

```bibtex
@inproceedings{wang2025fastgda,
  title={Fast Data Attribution for Text-to-Image Models},
  author={Wang, Sheng-Yu and Hertzmann, Aaron and Efros, Alexei A and Zhang, Richard and Zhu, Jun-Yan},
  booktitle={NeurIPS},
  year = {2025},
}
```