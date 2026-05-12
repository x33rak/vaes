# AGENTS.md — VAE Raindrop Removal Project

## Project Goal

Beat the **Attentive Generative Adversarial Network for Raindrop Removal** (Qian et al.) on a single image using **only Variational Autoencoders (VAEs)** — no discriminator, no adversarial training.

| Metric | Target (beat this) | Source |
|--------|-------------------|--------|
| PSNR   | > 31.57 dB        | Qian et al. Attentive GAN baseline |
| SSIM   | > 0.9023          | Qian et al. Attentive GAN baseline |

A secondary goal is **explainability**: VAEs provide an explicit density function `p(x)`, enabling principled uncertainty quantification, latent space inspection, and disentanglement — advantages that GANs fundamentally lack.

---

## Repository Layout

```
.
├── baseline_model_builder.py   # Baseline VAE architecture (reference, do not modify)
├── train.py                    # Training entry point
├── test.py                     # Evaluation entry point (reports PSNR / SSIM)
├── data_setup.py               # Dataset loading and preprocessing utilities
├── utils.py                    # Shared helpers (logging, checkpointing, metrics)
├── datasets/                   # Raw and preprocessed image data
├── images/                     # Sample / visualisation images
├── output/                     # Model output images produced during test
├── logs_512/                   # Train logs — baseline 512-px runs
├── weights_512/                # Saved checkpoints — baseline 512-px runs
├── loss/                       # Loss curve artefacts
├── misc/                       # Scratch files, one-off experiments
├── tutorials/                  # Reference papers, notebooks, and how-to guides
├── explore_data.ipynb          # EDA notebook
├── vae_baseline.ipynb          # Baseline VAE training walkthrough in jupyter notebook
├── test_modular_code.ipynb     # Unit-test notebook for modules in jupyter notebook
└── visualize.ipynb             # Latent-space and reconstruction visualisation in jupyter notebook
```

---

## Architecture Experiments

### Baseline (Do Not Modify)
- **File:** `baseline_model_builder.py`
- Treat as a frozen reference. All new variants must be in separate files (e.g. `model_builder_v2.py`, `model_builder_gmm.py`).
- Always report baseline PSNR / SSIM alongside any new result.

### Naming Convention for New Architectures
```
model_builder_<descriptor>.py
```
Examples: `model_builder_v2.py`, `model_builder_hierarchical.py`, `model_builder_gmm_prior.py`

Each file must expose at minimum:
```python
build_encoder(config) -> nn.Module
build_decoder(config) -> nn.Module
build_vae(config)     -> nn.Module   # wraps encoder + decoder + reparameterisation
```

---

## Experiment Dimensions

Agents should explore improvements along these axes. Open a separate branch or log directory for each meaningful combination.

### 1. Prior Distribution

The prior `p(z)` controls what the latent space "expects." Swapping the standard Gaussian changes both training dynamics and expressiveness.

| Prior | Implementation hint | Expected benefit |
|-------|-------------------|-----------------|
| Isotropic Gaussian `N(0, I)` | Baseline — already implemented | Reference |
| Gaussian Mixture Model (GMM) | Use VampPrior or learnable mixture of `K` Gaussians in latent space | Multi-modal structure; better for images with varied rain density |
| Normalising Flow prior | RealNVP / Glow as the prior | Arbitrarily complex `p(z)` |
| von Mises–Fisher (vMF) | Hyperspherical VAE | Better geometry for directional features |
| Learnable VampPrior | Pseudo-inputs trained end-to-end | Adaptive to the data distribution |

**Key rule:** When changing the prior, you must also update the KL-divergence term in the ELBO. Document the closed-form (or Monte Carlo estimate) used.

### 2. Encoder Architecture

| Variant | Notes |
|---------|-------|
| Baseline CNN encoder | See `baseline_model_builder.py` |
| ResNet encoder | Pre-trained ResNet-50/34 backbone; freeze early layers |
| Attention encoder | CBAM or self-attention blocks to focus on raindrop regions |
| Hierarchical encoder | Multiple stochastic layers `z_1, z_2, …` (NVAE-style) |
| Transformer encoder | Swin-Transformer patch encoder for global context |

### 3. Decoder Architecture

| Variant | Notes |
|---------|-------|
| Transposed-conv decoder | Baseline |
| Sub-pixel shuffle decoder | Avoids checkerboard artefacts |
| U-Net skip-connection decoder | Preserves high-frequency detail; connect encoder feature maps |
| Attention decoder | Cross-attention from latent to spatial feature maps |

### 4. Reconstruction Loss

The choice of `log p(x|z)` drives sharpness vs. smoothness.

| Loss | Formula hint | Notes |
|------|-------------|-------|
| MSE (Gaussian likelihood) | `‖x - x̂‖²` | Baseline; tends to blur |
| MAE (Laplacian likelihood) | `‖x - x̂‖₁` | Sharper than MSE |
| SSIM loss | `1 - SSIM(x, x̂)` | Directly optimises the eval metric |
| Perceptual loss | VGG feature-space L2 | Perceptually sharper |
| MS-SSIM + L1 combo | `α·MS-SSIM + (1-α)·L1` | Strong practical baseline |
| Charbonnier loss | `√(‖x - x̂‖² + ε²)` | Smooth L1 approximation |

**Recommended starting point for beating the GAN baseline:** MS-SSIM + L1 with perceptual loss.

### 5. Latent Space Dimensionality

- Start at `z_dim = 512` (matches existing log directories).
- Try `z_dim ∈ {256, 512, 1024, 2048}`.
- For hierarchical VAEs, try `(z1_dim=64, z2_dim=256)` etc.

### 6. Input / Output Image Distribution

- **Input conditioning:** Model as `p(x_clean | x_rainy)` — a conditional VAE (CVAE). The encoder takes `x_rainy` and produces `z`; the decoder reconstructs `x_clean`.
- **Output distribution:** Try Gaussian, Laplacian, or a normalising-flow-based decoder output distribution.

---

## Training (`train.py`)

### Running a Training Job

```bash
python train.py \
  --model_builder model_builder_v1 \
  --data_dir datasets/ \
  --log_dir logs_512/ \
  --weights_dir weights_512/ \
  --epochs 200 \
  --batch_size 4 \
  --lr 1e-4 \
  --latent_dim 512 \
  --loss_type ms_ssim_l1 \
  --prior gaussian
```

### Key Arguments Agents Should Know

| Argument | Description |
|----------|-------------|
| `--model_builder` | Python module name (without `.py`) for the architecture |
| `--prior` | Prior type: `gaussian`, `gmm`, `vampprior`, `flow` |
| `--latent_dim` | Dimensionality of `z` |
| `--loss_type` | Reconstruction loss: `mse`, `mae`, `ssim`, `perceptual`, `ms_ssim_l1` |
| `--beta` | β-VAE weight on KL term (default `1.0`); try `0.1–4.0` |
| `--lr_schedule` | `cosine`, `step`, `plateau` |
| `--log_dir` | TensorBoard log output directory |
| `--weights_dir` | Checkpoint save directory |

### Log Directories

| Directory | Purpose |
|-----------|---------|
| `logs_512/` | Standard runs at 512-px resolution |

Create new subdirectories for new experiments:
```
logs_<descriptor>_<resolution>/
weights_<descriptor>_<resolution>/
```

---

## Evaluation (`test.py`)

```bash
python test.py \
  --model_builder model_builder_v1 \
  --weights_dir weights_512/ \
  --data_dir datasets/ \
  --output_dir output/ \
  --latent_dim 512
```

`test.py` must report:
- **PSNR** (higher is better; target > 31.57 dB)
- **SSIM** (higher is better; target > 0.9023)
- Per-image results + aggregate mean / std

Agents must **never modify `test.py`** in ways that change the metric calculation; only infrastructure changes (argument parsing, output formatting) are acceptable.

---

## Explainability Requirements

Since a core project motivation is explainability over GANs, every experimental model must support:

1. **Latent traversal:** Ability to interpolate between two latent codes and decode the path.
2. **Reconstruction with uncertainty:** Report pixel-wise variance `Var[x̂]` from multiple samples of `z ~ q(z|x)`.
3. **Prior vs. posterior KL per-dimension:** Log per-dimension KL to identify which latent dimensions are active (KL > threshold).
4. **Disentanglement check:** For each latent dimension, vary it ±3σ and save the decoded grid to `output/latent_traversals/`.

Use `visualize.ipynb` to generate these artefacts after each major experiment.

---

## Evaluation Checklist Before Marking an Experiment Complete

- [ ] `test.py` runs without errors on the held-out test set
- [ ] PSNR and SSIM are logged and compared to the Qian et al. baseline (31.57 / 0.9023) and the internal VAE baseline
- [ ] TensorBoard logs are saved to the correct `logs_*/` directory
- [ ] Checkpoint is saved to the correct `weights_*/` directory
- [ ] Latent traversal visualisations are saved to `output/latent_traversals/`
- [ ] Architecture is documented in a docstring at the top of the `model_builder_*.py` file
- [ ] KL-divergence form (closed-form or MC) is documented in the model file
- [ ] Results are added to the **Results Table** below

---

## Results Table

Update this table after every completed experiment.

| Model File | Prior | Encoder | Decoder | Loss | β | PSNR (dB) | SSIM | Notes |
|------------|-------|---------|---------|------|---|-----------|------|-------|
| `baseline_model_builder.py` | Gaussian | CNN | Transposed-conv | MSE | softmax warmup | 29.425 | 0.8120 | Internal baseline |
| Qian et al. Attentive GAN | N/A | — | — | Adversarial | — | 31.57 | 0.9023 | **Target to beat** |
| `model_builder_v1.py` | Gaussian | CNN + mask | Residual transposed-conv | Baseline composite | softmax warmup | 30.080 | 0.814 | Residual clean-image prediction |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | Baseline composite | softmax warmup | 30.127 | 0.815 | Posterior mean at eval, V1 weights |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | MS-SSIM + L1 | softmax warmup | 30.642 | 0.827 | Best V2 structural-loss checkpoint |
| `model_builder_v3.py` | Gaussian | CNN + mask | Residual + refinement | MS-SSIM + L1 | softmax warmup | 30.578 | 0.827 | Frozen full-resolution refinement branch |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | MS-SSIM[0,1] + L1 | softmax warmup | 30.534 | 0.827 | Corrected MS-SSIM range did not improve test_b |
| `model_builder_v5.py` | Gaussian | U-Net CNN | U-Net residual decoder | MS-SSIM + L1 | 0.1 max | 30.562 | 0.825 | Multi-scale U-Net VAE overfit validation |
| `model_builder_v6.py` | Gaussian | CNN + supervised mask | Residual transposed-conv | MS-SSIM + L1 + mask aux | softmax warmup | 30.441 | 0.824 | Pseudo-mask supervision hurt held-out metrics |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | SSIM + L1 | softmax warmup | 30.493 | 0.829 | Direct single-scale SSIM objective |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | SSIM + L1 + random crops | softmax warmup | 30.599 | 0.829 | Fixed paired augmentation seed and crop fine-tune |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | SSIM + L1 + long crop run | softmax warmup | 30.641 | 0.830 | Longer low-LR crop fine-tune |
| `model_builder_v2.py` | Gaussian | CNN + mask | Residual transposed-conv | SSIM + L1 + drop-weighted L1 | softmax warmup | 30.669 | 0.830 | Best result so far; still below GAN target |
| `model_builder_v11_gmm_prior.py` | GMM | CNN + mask | Residual transposed-conv | SSIM + L1 + drop-weighted L1 | MC KL warmup | 30.690 | 0.830 | GMM prior gave only a tiny PSNR gain |
| `model_builder_v12_vqvae.py` | Discrete VQ | CNN + mask | Residual transposed-conv | SSIM + L1 + drop-weighted L1 + VQ | no Gaussian KL | 30.599 | 0.827 | Global hard quantization destabilized latent/decoder interface |
| `model_builder_v13_vqvae2.py` | Spatial discrete VQ | CNN + mask spatial bottleneck | Spatial residual decoder | SSIM + L1 + drop-weighted L1 + VQ | no Gaussian KL | 29.907 | 0.813 | VQ-VAE-2-style spatial codebook underfit/unstable |
| `model_builder_v11_gmm_prior.py` | GMM | CNN + mask | Residual transposed-conv | Pure SSIM | MC KL warmup | 30.692 | 0.831 | Simplified objective improved SSIM slightly |
| `model_builder_v11_gmm_prior.py` | GMM | CNN + mask | Residual transposed-conv | Pure SSIM + JPEG aug | MC KL warmup | 30.692 | 0.830 | JPEG augmentation did not improve JPEG test_b split |
| `model_builder_v17_gmm_refine.py` | GMM | CNN + mask | Residual + full-res refinement | Pure SSIM | frozen base + MC KL | 30.689 | 0.831 | Refinement improved validation but not test_b |
| `model_builder_v17_gmm_refine.py` | GMM | CNN + mask | Residual + full-res refinement | Pure SSIM + hflip TTA | frozen base + MC KL | 30.720 | 0.833 | Best SSIM so far, still below GAN target |

---

## Code Style & Contribution Rules

1. **One architecture per file.** Do not put multiple VAE variants in the same `model_builder_*.py`.
2. **Config dict, not argparse, inside model files.** All hyperparameters flow through a `config` dict so `train.py` can serialise them alongside checkpoints.
3. **Type hints required** on all public functions.
4. **No in-place modification of `baseline_model_builder.py`.** It is the frozen reference.
5. **GPU-agnostic code.** Use `device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")` everywhere; never hardcode `"cuda"`.
6. **Reproducibility.** Set seeds at the top of `train.py`:
   ```python
   torch.manual_seed(config["seed"])
   torch.cuda.manual_seed_all(config["seed"])
   ```
7. **Log every experiment** — if there are no TensorBoard logs, the experiment did not happen.
8. **Commit message format:** `[experiment] <model_file>: <brief change description>` e.g. `[experiment] model_builder_v2: GMM prior with K=8 components`.

---

## Useful References

- Kingma & Welling, *Auto-Encoding Variational Bayes* (2013) — foundational VAE paper
- Qian et al., *Attentive Generative Adversarial Network for Raindrop Removal From a Single Image* (CVPR 2018) — the baseline to beat
- Rezende & Viola, *Taming VAEs* (2018) — stability and quality improvements
- Child, *Very Deep VAEs* (NVAE) (2020) — hierarchical latent structure
- Tomczak & Welling, *VAE with a VampPrior* (2018) — learnable mixture prior
- Wang et al., *MS-SSIM* — multi-scale SSIM as a perceptual loss
- Zhao et al., *Loss Functions for Image Restoration with Neural Networks* — comprehensive loss function comparison
