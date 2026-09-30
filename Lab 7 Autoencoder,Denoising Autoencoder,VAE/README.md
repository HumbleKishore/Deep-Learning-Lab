# Experiment 7 – End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To develop an end-to-end understanding of autoencoders and their variants
for image representation, reconstruction, denoising and generative
modeling. The experiment starts with a fully connected autoencoder,
progresses to a Convolutional Autoencoder (CAE), introduces image
corruption to build a Denoising Autoencoder, and finally develops a
Variational Autoencoder (VAE). Learned latent representations are
visualized, reconstruction quality is compared using MSE, MAE and SSIM, and
the generative capability of the VAE is explored through sampling and
latent-space interpolation.

## Dataset
- **Name:** MNIST Handwritten Digit Dataset
- **Source:** Loaded directly via `tf.keras.datasets.mnist`
- **Details:** Grayscale images of digits 0–9, each of size 28×28×1.
- **Subset used:** A random subset (random permutation of the full
  train/test indices) of 10,000 training images and 2,000 test images.
  Of the 10,000 training images, 9,000 update the weights and 1,000 are
  held out for validation.
- **Preprocessing:** Pixel values rescaled from [0, 255] to [0, 1]. Labels
  are never used as reconstruction targets (the image is its own target);
  they are used only at the end to colour the VAE latent-space scatter plot.
- **Controlled protocol:** Every model (FC-AE, Conv-AE, Denoising-CAE, VAE)
  uses exactly the same train/validation/test split and preprocessing so
  that comparisons remain fair.

| Split | Shape |
|---|---|
| Training (weights updated) | (9000, 28, 28, 1) |
| Validation | (1000, 28, 28, 1) |
| Test | (2000, 28, 28, 1) |

## Method

### Part 1: Fully Connected Autoencoder
- Images flattened to 784-dimensional vectors.
- Architecture: 784 → 128 → 32 → **16 (latent)** → 32 → 128 → 784, with
  ReLU in hidden layers and sigmoid at the output.
- Adam (lr $10^{-3}$), binary cross-entropy, batch size 128, 20 epochs.
- Original vs. reconstructed images (with per-image MSE) and
  training/validation loss curves were plotted; MSE, MAE and SSIM were
  computed on the test set.

### Part 2: Convolutional Autoencoder
- Encoder: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool, giving a
  7×7×64 latent feature map.
- Decoder: Conv2D(64) → UpSampling → Conv2D(32) → UpSampling →
  Conv2D(1, sigmoid).
- Same optimizer, loss, batch size and epochs as the FC-AE. The first 16
  channels of the latent feature map were visualized, and reconstructions
  were compared against the FC-AE.

### Part 3: Denoising Convolutional Autoencoder
- The CAE architecture was reused with fresh weights, trained on noisy
  inputs against **clean** targets.
- Two corruption types: Gaussian noise ($\sigma \in \{0.1, 0.2, 0.3\}$,
  clipped to [0, 1]; $\sigma = 0.2$ used for training) and salt-and-pepper
  noise ($p \in \{0.05, 0.10, 0.20\}$).
- Clean vs. noisy vs. denoised images and noise level vs. MSE/MAE/SSIM
  were plotted for both noise types.

### Part 4: Variational Autoencoder
- Latent space $z \in \mathbb{R}^2$; encoder outputs $\mu$ and $\log\sigma^2$
  and samples $z = \mu + \sigma \odot \epsilon$ (reparameterization trick).
- Strided-convolution encoder and Conv2DTranspose decoder, implemented as a
  custom `tf.keras.Model` subclass with a custom `train_step`.
- Loss = BCE reconstruction (summed over pixels) + KL divergence to
  $\mathcal{N}(0, I)$. Adam (lr $10^{-3}$), batch size 128, 30 epochs.
- Latent space visualization (coloured by label), random generation from
  $z \sim \mathcal{N}(0, I)$, and 3 → 8 latent interpolation were carried out.

### Part 5: Analysis Plots
- Training/validation curves for the VAE (reconstruction and KL terms),
  per-image reconstruction error distribution (FC-AE vs. Conv-AE), and the
  five highest-error test images under the FC-AE.

### Part 6: Additional Exercises
1. Convolutional AE with a dense bottleneck ($d_z \in \{4, 8, 16, 32\}$)
2. Gaussian- vs. salt-and-pepper-trained denoisers
3. Denoisers trained at different noise levels ($\sigma = 0.1, 0.2, 0.3$)
4. Transposed convolution vs. upsampling decoder
5. VAE with $d_z = 2$ vs. $d_z = 8$
6. Effect of the KL weight $\beta \in \{0.5, 1, 2, 4\}$
7. Larger grid (100 samples) of VAE-generated images
8. Interpolation between mean latent codes of digits '3' and '8'
- Also: an FC-AE latent-dimension study ($d_z \in \{2, 8, 16, 32\}$).

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_7.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- tensorflow
- scikit-image
- jupyter

Install with:
```bash
pip install -r requirements.txt
```

## Execution Instructions
1. Clone this repository:
```bash
   git clone https://github.com/HumbleKishore/Deep-Learning-Lab.git
   cd "Lab 7 Autoencoders"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (MNIST is downloaded
   automatically via `tf.keras.datasets`, no separate dataset download
   needed):
```bash
   jupyter notebook Lab_7.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory (via the `save_plot()` helper in the notebook), and the
   console will print dataset shapes, model parameter counts, per-epoch
   training logs, reconstruction metrics (MSE, MAE, SSIM), VAE loss terms
   and training times.

## Results

### Part 1: Fully Connected Autoencoder
| Parameter | Value |
|---|---|
| Latent dimension | 16 |
| Total parameters | 211,040 |
| Final train / val loss (BCE) | 0.1277 / 0.1299 |
| Training time | 6.8 s |

Digit shape is preserved in every reconstruction, but thin strokes come
back thicker and blurrier and loops in 8s, 9s and 3s are smoothed out.
Training and validation loss stay within about 0.002 of each other, so
there is no meaningful overfitting. Test MSE = 0.021484, MAE = 0.057818,
mean SSIM = 0.754313.

### Part 2: Convolutional Autoencoder
The CAE (74,497 parameters, 56.4 s) reached a final train/val loss of
0.0690 / 0.0696. Its 7×7×64 latent keeps a genuine spatial layout, with
different channels responding to different stroke parts.

| Model | MSE | MAE | SSIM | Parameters |
|---|---|---|---|---|
| Fully Connected AE | 0.021484 | 0.057818 | 0.754313 | 211,040 |
| Convolutional AE | 0.002682 | 0.015649 | 0.972809 | 74,497 |

The CAE reconstructs far more sharply (MSE almost 8× lower) with fewer
parameters, because convolutions exploit spatial locality through shared
filters and its 3136-value latent retains far more information than the
16-value FC latent.

### Part 3: Denoising Autoencoder
Trained at Gaussian $\sigma = 0.2$ (final train/val loss 0.0760 / 0.0767,
55.0 s). Denoised output error stays far below noisy input error at every
noise level tested.

| Gaussian σ | MSE | MAE | SSIM |
|---|---|---|---|
| 0.1 | 0.003636 | 0.018023 | 0.960311 |
| 0.2 | 0.004655 | 0.021622 | 0.946269 |
| 0.3 | 0.007038 | 0.029122 | 0.887798 |

| Salt-and-pepper p | MSE | MAE | SSIM |
|---|---|---|---|
| 0.05 | 0.004933 | 0.021204 | 0.938076 |
| 0.10 | 0.007138 | 0.026795 | 0.900202 |
| 0.20 | 0.013216 | 0.040365 | 0.802060 |

Even though it was trained only on Gaussian noise, the denoiser removes
most salt-and-pepper corruption at p = 0.05/0.10; at p = 0.20 quality
degrades faster.

### Part 4: Variational Autoencoder
The VAE ($d_z = 2$, 134,165 parameters, 44.5 s) reached a final train/val
loss (BCE + KL) of 156.8752 / 158.6146.

| VAE Metric | Value |
|---|---|
| Reconstruction Loss (BCE) | 151.849945 |
| KL Loss | 5.820126 |
| Total Loss | 157.670071 |
| Test MSE | 0.044264 |
| Test MAE | 0.102821 |
| Mean SSIM | 0.502727 |

Digits occupy contiguous regions of the 2D latent space (with overlaps
between similar digits such as 3/8, 4/9, 7/1), the point cloud is centred
near the origin, sampling $z \sim \mathcal{N}(0, I)$ produces recognisable
digits with varied style, and interpolating from a '3' to an '8' produces a
smooth morph with no sudden jumps.

### Part 5: Consolidated Results
| Model | MSE | MAE | SSIM | Parameters | Time (s) |
|---|---|---|---|---|---|
| FC Autoencoder | 0.021484 | 0.057818 | 0.754313 | 211,040 | 6.77 |
| Conv. Autoencoder | 0.002682 | 0.015649 | 0.972809 | 74,497 | 56.45 |
| Denoising CAE (σ = 0.2 input) | 0.004634 | 0.021571 | 0.946698 | 74,497 | 55.01 |
| VAE ($d_z$ = 2) | 0.044264 | 0.102821 | 0.502727 | 134,165 | 44.53 |

The Denoising-CAE is measured on noisy inputs (a harder task), and the VAE
is trained with a different objective (BCE + KL), so its pixel metrics are
not directly comparable to the reconstruction-only models. Per-image error
medians: FC-AE 0.0209 vs. Conv-AE 0.0025. The highest-error images are
unusual handwriting styles that are rare in the training set.

### Part 6: Additional Exercises
- **Latent dimension (FC-AE):** MSE falls from 0.057084 ($d_z$ = 2) to
  0.020623 ($d_z$ = 16) and 0.019906 ($d_z$ = 32); SSIM plateaus near 0.77
  beyond $d_z$ = 16.
- **CAE with dense bottleneck:** MSE drops from 0.039902 ($d_z$ = 4) to
  0.008994 ($d_z$ = 32), still worse than the natural 7×7×64 latent
  (0.002682).
- **Gaussian vs. salt-and-pepper denoiser:** each performs best on the
  corruption it was trained on (SSIM 0.9465 on Gaussian for the
  Gaussian-trained model; 0.9450 on salt-and-pepper for the
  salt-and-pepper-trained model).
- **Training noise level:** the σ = 0.1 model is sharpest on light noise but
  degrades fastest; the σ = 0.3 model is most robust at heavy noise
  (SSIM 0.9208 at σ = 0.3).
- **Decoder type:** Conv2DTranspose (MSE 0.002120, SSIM 0.979607) slightly
  outperformed UpSampling + Conv2D (MSE 0.002682, SSIM 0.972809).
- **VAE $d_z$ = 2 vs. 8:** $d_z$ = 8 reconstructs better (MSE 0.038208,
  SSIM 0.596153) but is less cleanly separable in a 2D projection.
- **KL weight β:** larger β lowers the KL loss (7.81 → 3.50 from β = 0.5 to
  4) but hurts reconstruction; β = 1 gave the best MSE/SSIM.
- **Larger sample grid and class-mean interpolation:** 100 generated samples
  show broad diversity, and interpolating between the mean latent codes of
  '3' and '8' is also smooth.

## Conclusion
This experiment demonstrated the full image-representation pipeline: image
→ encoder → latent representation → decoder → reconstruction, extended to
denoising and to variational generation. The Convolutional Autoencoder
gave the best pure-reconstruction performance (MSE 0.0027, SSIM 0.973)
with fewer parameters than the fully connected model, because it preserves
spatial structure and keeps a much larger latent. The denoising CAE
recovered clean digit structure from both Gaussian and salt-and-pepper
corruption, with error rising gracefully as noise increased. The
2-dimensional VAE gave the weakest pixel-level reconstruction but was the
only model with a smooth, regularised latent space that supports
meaningful random sampling and interpolation, illustrating the central
trade-off between deterministic reconstruction fidelity and a generative
latent representation. Additional studies confirmed that reconstruction
quality improves with latent dimension until it saturates, that denoisers
perform best on the noise they were trained on, and that a larger KL weight
trades reconstruction quality for a more tightly regularised latent space.
