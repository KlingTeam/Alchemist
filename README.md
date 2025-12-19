# Alchemist: Unlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection
[**Project**](https://kxding.github.io/project/Alchemist/) | [**Paper**](https://arxiv.org/pdf/2512.16905) | [**Huggingface**](https://huggingface.co/papers/2512.16905) 

[Kaixin Ding](https://kxding.github.io/project/Alchemist/)<sup>1</sup>, [Yang Zhou](https://kxding.github.io/project/Alchemist/)<sup>2</sup>, [Xi Chen](https://xavierchen34.github.io/)<sup>1</sup>, [Miao Yang](https://kxding.github.io/project/Alchemist/)<sup>3</sup>, [Jiarong Ou](https://kxding.github.io/project/Alchemist/)<sup>3</sup>, [Rui Chen](https://kxding.github.io/project/Alchemist/)<sup>3</sup>, [Xin Tao](https://www.xtao.website/)<sup>3</sup>, [Hengshuang Zhao](https://hszhao.github.io/)<sup>1</sup>.
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

## Training the Rater

### Prerequisites

Before training, you need to download the train dataset https://laion.ai/. 
and divid 1M for the val dataset 


### Train on LAION Dataset

Train a model from this script:

```bash
bash train_rater.sh
```

### Training Configurations
All training configurations are stored in `configs/config_rater.json`.

**Dataset Settings:**
- `train_csv_path`: Path to training CSV file
- `val_csv_path`: Path to validation CSV file
- `text_enc_path`: Text encoder model path
- `ckpt_path`: Checkpoint directory path

**Model Architecture:**
- `depth`: Model depth (default: `16`)
- `raterDepth`: Rater depth (default: `8`)
- `patch_size`: Patch size (default: `16`)
- `patch_nums`: Multi-scale patch numbers

**Training Hyperparameters:**
- `bs`: Batch size (default: `64`)
- `ep`: Number of epochs (default: `10`)
- `tlr`: Learning rate (default: `1e-4`)
- `twd`: Weight decay (default: `0.05`)
- `cfg`: Classifier-free guidance scale (default: `4.0`)

**Optimization:**
- `opt`: Optimizer type (default: `adamw`)
- `sche`: Learning rate scheduler (default: `lin0`)
- `wp`: Warmup proportion
- `fp16`: Enable mixed precision training

**Output:**
- `local_out_dir_path`: Output directory for checkpoints and logs
- `exp_name`: Experiment name

### Scoring
```bash
bash infer_rater.sh
```
and rank the data samples by ratings in descending order.


## Acknowledgments

Our code is built upon [STAR-T2I](https://github.com/Davinci-XLab/STAR-T2I) and [SEAL](https://github.com/hanshen95/SEAL).

## Citation
If you use Alchemist in your research, please cite:

```bibtex
@inproceedings{ding2025alchemist,
  title={Alchemist: Unlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection},
  author={Kaixin Ding, Yang Zhou, Xi Chen, Miao Yang, Jiarong Ou, Rui Chen, Xin Tao, Hengshuang Zhao},
  booktitle={arxiv:2512.16905},
  year = {2025},
}
```