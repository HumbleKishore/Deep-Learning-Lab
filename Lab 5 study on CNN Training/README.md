# Experiment 5 – Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To systematically study the effect of weight initialization,
regularization, optimization algorithms, CNN hyperparameters, transfer
learning, fine tuning and cross-validation on image classification
performance, using a single CNN architecture, MobileNetV2, on the
Oxford-IIIT Pet dataset. This includes freezing the pretrained
convolutional base to train a lightweight classifier head, fine tuning
the upper convolutional layers, and selecting a final model configuration
using 5-fold cross-validation.

## Dataset
- **Name:** Oxford-IIIT Pet Dataset
- **Source:** Loaded directly via `tensorflow_datasets.load("oxford_iiit_pet")`
- **Details:** 3,680 training images and 3,669 testing images, RGB, 37
  pet breed classes (cats and dogs).
- **Preprocessing:** Images resized to 224×224×3 and normalized using
  MobileNetV2's `preprocess_input`; 20% of the training split (736 images)
  held out for validation, leaving 2,944 images for training.

## Method

### Part 1: Bottleneck Feature Extraction
- A pretrained MobileNetV2 model (ImageNet weights, `include_top=False`,
  global average pooling) was loaded and used purely as a frozen feature
  extractor to cache a 1280-dimensional feature vector per image
  (`train_feat`, `val_feat`, `test_feat`), so that all exploratory
  classifier-head experiments could be run cheaply on a CPU without
  repeatedly passing images through the full convolutional base.

### Part 2: Weight Initialization
- Four initialization strategies for the classifier head were compared:
  Zero, Random, Xavier/Glorot and He, each trained for 10 epochs on the
  cached bottleneck features.

### Part 3: Regularization and Batch Normalization
- Four head configurations were compared: No Regularization, L2
  Regularization ($\lambda=10^{-3}$), Dropout ($p=0.5$) and Batch
  Normalization, to study overfitting via training/validation accuracy
  and loss curves.
- A numerical Batch Normalization example was worked through directly on
  `x = [2, 4, 6, 8]`, and With-BN vs. Without-BN validation accuracy was
  compared.

### Part 4: Optimization Algorithms
- SGD, Momentum, RMSProp and Adam were compared (learning rate $10^{-3}$,
  He initialization, Batch Normalization enabled) over 10 epochs each.

### Part 5: CNN Hyperparameter Tuning
- Learning rate (0.001, 0.0001), batch size (16, 32, 64) and dropout rate
  (0, 0.25, 0.5) were each varied independently (one at a time) on the
  classifier head, and convolution output-size behaviour was verified
  using the standard $O = \lfloor (N+2P-K)/S \rfloor + 1$ formula.

### Part 6: Transfer Learning and Fine Tuning
- **Case A (Feature Extraction):** the MobileNetV2 base was kept frozen
  and only the classifier head (dropout 0.3) was trained for 5 epochs.
- **Case B (Fine Tuning):** layers from index 100 onward in the base were
  unfrozen and the full model was trained end-to-end for 5 further epochs
  at a lower learning rate ($10^{-4}$), with an additional comparison
  against fine tuning at $10^{-5}$.

### Part 7: 5-Fold Cross-Validation and Final Evaluation
- Four candidate head configurations (C1–C4, varying Batch Normalization,
  Dropout and L2) were evaluated with 5-fold cross-validation on the
  training data. The best configuration was retrained on the full training
  set and evaluated once on the untouched test set using accuracy,
  precision, recall, F1-score and a confusion matrix.

### Part 8: Additional Exercise
- Two further configurations were evaluated and compared against the
  selected model: a frozen-base head with a different learning
  rate/dropout/batch-size combination (5-fold CV), and a partial-unfreezing
  fine-tuned model (single train/validation/test split).

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_5.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- tensorflow-datasets
- scikit-learn
- jupyter

Install with:
```bash
pip install -r requirements.txt
```

## Execution Instructions
1. Clone this repository:
```bash
   git clone https://github.com/HumbleKishore/Deep-Learning-Lab.git
   cd "Lab 5 CNN Comprehensive Study"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (the Oxford-IIIT Pet
   dataset is downloaded automatically via `tensorflow_datasets`, no
   separate dataset download needed):
```bash
   jupyter notebook Lab_5.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory (via the `save_plot()` helper in the notebook), and the
   console will print dataset shapes, MobileNetV2 layer statistics,
   per-epoch training logs, cross-validation fold accuracies, fine-tuning
   logs, and final evaluation metrics.

## Results

### Part 1: Weight Initialization
Random, Xavier/Glorot and He initialization all converged to a similarly
high validation accuracy on the classifier head, since only a shallow
head sits on top of fixed, already-good ImageNet features. Zero
initialization converged far more slowly and plateaued at a visibly
lower accuracy throughout, consistent with the classic symmetry problem.

### Part 2: Regularization and Batch Normalization
All four configurations showed a training/validation gap, but Dropout
and Batch Normalization kept validation accuracy and loss tracking
training performance more closely than the unregularized baseline or L2
alone. The Batch Normalization worked example on `x = [2, 4, 6, 8]` gave
mean = 5, variance = 5, and normalized output
`[-1.342, -0.447, 0.447, 1.342]`; enabling BN on the head produced higher
and smoother validation accuracy than without it.

### Part 3: Optimization Algorithms
| Optimizer | Final Loss | Best Val. Accuracy | Epoch to Converge | Time (s) |
|---|---|---|---|---|
| SGD | 0.8100 | 0.7921 | 10 | 2.007 |
| Momentum | 0.1266 | 0.9511 | 9 | 2.133 |
| RMSProp | 0.0009 | 0.9755 | 9 | 2.208 |
| Adam | 0.0103 | 0.9728 | 10 | 2.607 |

RMSProp and Adam converged fastest and reached the highest validation
accuracy (97.55% and 97.28% respectively), Momentum performed well
(95.11%), while plain SGD lagged well behind (79.21%).

### Part 4: Transfer Learning and Fine Tuning
With the MobileNetV2 base frozen, the classifier head reached only
**76.63%** validation accuracy after 5 epochs. After unfreezing layers
from index 100 onward and fine tuning for 5 further epochs at a lower
learning rate ($10^{-4}$), validation accuracy rose sharply to
**98.78%** — a substantial gain confirming that adapting the upper
convolutional layers to the fine-grained pet-breed task improves
performance considerably over frozen features alone.

### Part 5: 5-Fold Cross-Validation and Final Evaluation
| Configuration | Mean CV Accuracy | SD |
|---|---|---|
| C1_Baseline | 0.9113 | 0.0117 |
| C2_BestInit_BN | 0.9005 | 0.0085 |
| C3_BN_Dropout | 0.9076 | 0.0137 |
| C4_BN_Dropout_L2 | 0.9096 | 0.0057 |

**C1_Baseline** was selected as the final configuration.

| Metric | Value |
|--------|-------|
| Mean CV Accuracy | 0.9113 |
| CV Standard Deviation | 0.0117 |
| Test Accuracy | 0.8986 |
| Precision (macro) | 0.9032 |
| Recall (macro) | 0.8982 |
| F1-score (macro) | 0.8977 |
| Training Time | 3.239 s |
| Total Parameters | 2,305,381 |

The confusion matrix shows a strong diagonal across the 37 breeds, with
most confusion occurring between visually similar cat or dog breeds that
share coat colour, pattern or body shape.

### Part 6: Hyperparameter Study
Increasing the learning rate improved validation accuracy within the
fixed epoch budget; validation accuracy varied only mildly across batch
sizes 16/32/64; and a moderate dropout rate (0.25) offered the best
trade-off between under- and over-regularizing the head.

### Part 7: Additional Exercise
| Configuration | CV Mean | CV SD | Test Accuracy | Training Time (s) |
|---|---|---|---|---|
| Extra1 (Frozen, LR 5e-4, Dropout 0.4, BS 16) | 0.8998 | 0.0117 | 0.8896 | 10.991 |
| Extra2 (Partial Unfreeze, LR 1e-4, Dropout 0.2, BS 64) | 0.9497 | – | 0.8498 | 351.150 |
| C1_Baseline (selected) | 0.9113 | 0.0117 | 0.8986 | 3.239 |

Extra1 performed slightly worse than C1_Baseline at a higher
computational cost. Extra2 reached a higher single-split validation
accuracy through partial unfreezing, but its test accuracy was actually
the lowest of the three and its training time over 100× higher, so
C1_Baseline remained the more reliable and efficient practical choice.

## Conclusion
This experiment confirmed that weight initialization, regularization,
optimizer choice and CNN hyperparameters each have a measurable effect
on classifier-head performance, with He/Xavier initialization, Batch
Normalization/Dropout, and adaptive optimizers such as RMSProp and Adam
consistently outperforming their simpler counterparts. Using a frozen
MobileNetV2 base as a feature extractor made it computationally feasible
to explore all of these design choices on a CPU, but fine tuning the
upper convolutional layers was essential to close the larger performance
gap on the fine-grained 37-breed task, raising validation accuracy from
76.63% (frozen) to 98.78% (fine-tuned). The final model, selected via
5-fold cross-validation for its balance of accuracy, low variability and
minimal training cost, achieved a test accuracy of 89.86%. Overall, the
experiment illustrates how transfer learning, careful regularization and
principled hyperparameter selection combine to produce a reliable image
classifier, and why the choice between a cheap frozen-feature model and a
more expensive fully fine-tuned model should be guided by both accuracy
and computational budget.
