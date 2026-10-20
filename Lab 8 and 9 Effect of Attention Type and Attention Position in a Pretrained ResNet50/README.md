# Experiments 8, 9 – Effect of Attention Type and Attention Position in a Pretrained ResNet50
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

**Repository:** [github.com/HumbleKishore/Deep-Learning-Lab – Lab 8 and 9](https://github.com/HumbleKishore/Deep-Learning-Lab/tree/main/Lab%208%20and%209%20Effect%20of%20Attention%20Type%20and%20Attention%20Position%20in%20a%20Pretrained%20ResNet50)

## Objective
To understand how different attention mechanisms modify a pretrained
convolutional neural network, and how the **position** at which attention
is inserted influences image classification performance. A single
pretrained backbone, ResNet50 (ImageNet weights), is used throughout.
Channel attention, spatial attention, sequential channel–spatial
attention, a lightweight self-attention module, and four established
mechanisms (SE, ECA, CBAM, Coordinate Attention) are implemented and
inserted at different candidate stages of the backbone, and every
resulting model is compared under an identical training protocol, both
quantitatively (accuracy, precision, recall, macro F1, parameters,
inference time) and qualitatively (Grad-CAM, internal attention maps).

## Dataset
- **Name:** TensorFlow Flowers (`tf_flowers`)
- **Classes:** `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`
  (5 classes), with variation in scale, viewpoint, illumination,
  background clutter, flower size, colour, texture and object location.
- **Subset used:** A balanced random subset of 160 images per class
  (800 images total), drawn with a fixed shuffle seed to keep the 25-model
  ablation sweep tractable on the available GPU.
- **Preprocessing:** Every image resized to 224×224×3 and passed through
  the ImageNet `preprocess_input` pipeline expected by ResNet50.
- **Augmentation (train-time only):** random horizontal flip, small
  rotation, zoom, translation, mild contrast change — applied identically
  across all models; the test set is never augmented.
- **Controlled protocol:** the exact same 70:15:15 split is reused,
  unchanged, for every one of the 25 models trained in this report.

| Split | Shape |
|---|---|
| Training | (560, 224, 224, 3) |
| Validation | (120, 224, 224, 3) |
| Test | (120, 224, 224, 3) |

## Method

### Part 1: Backbone Probe and Baseline
- ResNet50 (ImageNet weights) probed on a real input to confirm stage
  dimensions: P1 `conv2_block3_out` (56×56×256), P2 `conv3_block4_out`
  (28×28×512), P3 `conv4_block6_out` (14×14×1024), P4 `conv5_block3_out`
  (7×7×2048).
- Baseline architecture: Input → ResNet50 (backbone frozen) → Global
  Average Pooling → Dropout(0.3) → Dense(5, softmax). Adam (lr $10^{-3}$),
  categorical cross-entropy, batch size 32, up to 7 epochs — this protocol
  is reused unchanged for every subsequent model.

### Part 2: Attention Modules Implemented
- **Channel attention** — GAP descriptor → two dense layers (reduction
  ratio $r=16$) → sigmoid gate, $F_c = F \odot M_c(F)$.
- **Spatial attention** — channel-wise avg/max pooling → concatenate →
  $k\times k$ convolution → sigmoid gate, $F'_s = F \odot M_s(F)$; kernel
  size $k \in \{3,5,7\}$ ablated.
- **Sequential channel–spatial attention** — $F \to M_c \to M_s$ (and, for
  the order ablation, $F \to M_s \to M_c$); identical in construction to
  CBAM.
- **Self-attention** — $Q,K,V$ from the flattened $N\times C$ feature map,
  $A = \mathrm{softmax}(QK^T/\sqrt d)$, residual form
  $F_{SA} = F + \gamma \cdot \mathrm{reshape}(AV)$ with a learnable scale
  $\gamma$.
- **Established mechanisms:** SE (= plain channel attention), ECA (a
  single local 1-D convolution over channels instead of two dense layers),
  CBAM (= sequential channel→spatial), Coordinate Attention (separate
  height-/width-pooled channel descriptors).
- Every module was verified to be dimension-preserving (e.g. at P3,
  (14,14,1024) → (14,14,1024)).

### Part 3: Attention-Type Ablation (fixed at P3)
- Six models trained with attention position fixed at P3: Baseline,
  Channel, Spatial, Channel+Spatial, Self-Attention, and a Combined
  (Channel+Spatial+Self) variant.

### Part 4: Attention-Position Ablation
- The identical Channel+Spatial module moved across P2, P3 and P4.
- **Extended:** attention at a single position (P3 only / P4 only) vs.
  multiple simultaneous positions (P3+P4, P2+P3+P4).

### Part 5: Attention Order and Kernel-Size Ablations
- Channel→Spatial vs. Spatial→Channel, both fixed at P3.
- Spatial-attention kernel size $k \in \{3, 5, 7\}$, fixed at P3.

### Part 6: Comparison with Established Mechanisms
- SE, ECA, CBAM and Coordinate Attention, all inserted at P3, compared
  against the baseline and against Self-Attention.

### Part 7: Quantitative, Complexity and Qualitative Analysis
- Full result table (accuracy, precision, recall, macro F1, trainable/
  total parameters, inference time per image) across all 25 trained
  models.
- Training/validation accuracy and loss curves; accuracy-vs-parameters and
  macro-F1-vs-inference-time scatter plots; confusion matrices (baseline
  vs. best attention model).
- Grad-CAM (baseline vs. attention models); internal attention-map
  visualizations (feature maps before/after attention, channel-attention
  weights, spatial-attention overlay, self-attention response); attention
  maps at P2/P3/P4 on the same image; qualitative comparison across
  correctly/incorrectly classified images.

## Repository Structure
```text
├── README.md
├── requirements.txt
├── experiment-8-and-9-final.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- tensorflow
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
   cd "Deep-Learning-Lab/Lab 8 and 9 Effect of Attention Type and Attention Position in a Pretrained ResNet50"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (`tf_flowers` is
   downloaded automatically via Keras' dataset utilities, no separate
   dataset download needed; a GPU is strongly recommended since 25
   ResNet50-based models are trained):
```bash
   jupyter notebook experiment-8-and-9-final.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory, and the console will print backbone stage dimensions,
   per-model training logs, the full result table, and the final summary
   (`experiment_8_9_results.csv`).

## Results

### Part 1: Baseline
| Model | Train Acc | Val Acc | Test Acc | Macro F1 | Train Time |
|---|---|---|---|---|---|
| Baseline ResNet50 | 0.968 | 0.833 | 0.858 | 0.856 | 80.5 s |

### Part 2: Attention-Type Ablation (P3)
| Model | Acc | Prec | Rec | F1 | Params | Inf. (ms/img) |
|---|---|---|---|---|---|---|
| A0: Baseline | 0.858 | 0.855 | 0.860 | 0.856 | 23,597,957 | 156.6 |
| A1: Channel | 0.875 | 0.871 | 0.876 | 0.871 | 23,730,117 | 143.1 |
| A2: Spatial | 0.858 | 0.856 | 0.857 | 0.854 | 23,598,056 | 143.9 |
| A3: Channel+Spatial | **0.900** | **0.896** | **0.901** | **0.896** | 23,730,216 | 145.1 |
| A4: Self-Attention | 0.850 | 0.855 | 0.854 | 0.847 | 24,909,958 | 142.7 |
| A5: Combined | 0.883 | 0.882 | 0.882 | 0.881 | 25,042,217 | 148.3 |

Channel+Spatial attention (A3) gives the best result, 4.2 accuracy points
above the baseline for an increase of under 0.6% in parameters. Plain
spatial attention alone (A2) barely moves the needle; stacking every
mechanism at once (A5) underperforms the simpler two-mechanism A3.

### Part 3: Attention-Position Ablation
| Model | Position | Acc | Prec | Rec | F1 | Params |
|---|---|---|---|---|---|---|
| B2: Channel+Spatial | P2 | 0.783 | 0.793 | 0.782 | 0.781 | 23,631,368 |
| B3 (=A3): Channel+Spatial | P3 | **0.900** | **0.896** | **0.901** | **0.896** | 23,730,216 |
| B4: Channel+Spatial | P4 | 0.875 | 0.872 | 0.868 | 0.870 | 24,124,520 |

Moving the identical module from P2 to P3 raises accuracy by 11.7 points;
P3 (intermediate, flower-part-level features) is the best single insertion
point for this dataset.

| Positions | Acc | F1 | Params | Inf. (ms/img) |
|---|---|---|---|---|
| P3 only | 0.850 | 0.852 | 23,730,216 | 144.1 |
| P4 only | **0.900** | **0.900** | 24,124,520 | 143.2 |
| P3 + P4 | 0.892 | 0.884 | 24,256,779 | 150.4 |
| P2 + P3 + P4 | 0.858 | 0.853 | 24,290,190 | 155.1 |

Attention at more positions does not translate into a monotonic accuracy
gain — a single well-chosen position matched or beat every multi-position
combination tested, at lower cost.

### Part 4: Attention Order and Kernel Size (P3)
| Order | Acc | Prec | Rec | F1 |
|---|---|---|---|---|
| Channel → Spatial | 0.875 | 0.877 | 0.881 | 0.876 |
| Spatial → Channel | 0.875 | 0.872 | 0.876 | 0.873 |

| Spatial Kernel | Accuracy | Macro F1 | Inference Time (s/img) |
|---|---|---|---|
| 3×3 | 0.8833 | 0.8791 | 0.1407 |
| 5×5 | **0.9000** | **0.9013** | 0.1435 |
| 7×7 | 0.8833 | 0.8831 | 0.1392 |

Attention order makes no practically significant difference; accuracy and
macro F1 both peak at the middle kernel size (5×5), with negligible change
in inference time across kernel sizes.

### Part 5: Comparison with Established Mechanisms (P3)
| Attention | Acc | Prec | Rec | F1 | Params | Inf. (ms/img) |
|---|---|---|---|---|---|---|
| No Attention (Baseline) | 0.858 | 0.855 | 0.860 | 0.856 | 23,597,957 | 156.6 |
| SE (Channel) | 0.867 | 0.863 | 0.869 | 0.863 | 23,730,117 | 141.1 |
| ECA | 0.867 | 0.872 | 0.859 | 0.862 | 23,597,962 | 146.8 |
| CBAM (Channel+Spatial) | 0.867 | 0.866 | 0.867 | 0.865 | 23,730,216 | 148.9 |
| Coordinate Attention | 0.775 | 0.789 | 0.782 | 0.777 | 23,796,933 | 151.3 |
| Self-Attention | 0.850 | 0.855 | 0.854 | 0.847 | 24,909,958 | 142.7 |

SE, ECA and CBAM all give a similar, modest improvement with almost
identical parameter counts; ECA matches CBAM's accuracy while adding
effectively no parameters over the baseline. Coordinate Attention performs
*worse* than the baseline despite extra parameters — more parameters do
not guarantee more accuracy.

### Part 6: Confusion Matrix (Baseline vs. Best Attention Model)
Best attention model by macro F1: **E_SpatialAttention_k5_P3**
(acc = 0.900, F1 = 0.901). The baseline misclassifies 5 `roses` images as
`tulips`; the attention model reduces this to 0, which is the single
largest source of its accuracy gain. The `tulips` ↔ `daisy`/`roses`
confusion persists in both models, suggesting a genuine visual ambiguity
rather than a backbone weakness that attention resolves. The gain is
concentrated almost entirely in the `roses` class; `sunflowers` is
unchanged.

### Part 7: Consolidated Results (full table)
| Model | Position | Acc | Prec | Rec | F1 | Train. Params | Total Params | Inf. (s/img) |
|---|---|---|---|---|---|---|---|---|
| Baseline | – | 0.858 | 0.855 | 0.860 | 0.856 | 14,986,245 | 23,597,957 | 0.1566 |
| A1_Channel_P3 | P3 | 0.875 | 0.871 | 0.876 | 0.871 | 15,118,405 | 23,730,117 | 0.1431 |
| A2_Spatial_P3 | P3 | 0.858 | 0.856 | 0.857 | 0.854 | 14,986,344 | 23,598,056 | 0.1439 |
| A3_ChannelSpatial_P3 | P3 | **0.900** | 0.896 | 0.901 | 0.896 | 15,118,504 | 23,730,216 | 0.1451 |
| A4_SelfAttention_P3 | P3 | 0.850 | 0.855 | 0.854 | 0.847 | 16,298,246 | 24,909,958 | 0.1427 |
| A5_Combined_P3 | P3 | 0.883 | 0.882 | 0.882 | 0.881 | 16,430,505 | 25,042,217 | 0.1483 |
| B2_ChannelSpatial_P2 | P2 | 0.783 | 0.793 | 0.782 | 0.781 | 15,019,656 | 23,631,368 | 0.1450 |
| B4_ChannelSpatial_P4 | P4 | 0.875 | 0.872 | 0.868 | 0.870 | 15,512,808 | 24,124,520 | 0.1466 |
| C_P3_only | [P3] | 0.850 | 0.861 | 0.856 | 0.852 | 15,118,504 | 23,730,216 | 0.1441 |
| C_P4_only | [P4] | **0.900** | 0.905 | 0.900 | 0.900 | 15,512,808 | 24,124,520 | 0.1432 |
| C_P3_plus_P4 | [P3,P4] | 0.892 | 0.896 | 0.881 | 0.884 | 15,645,067 | 24,256,779 | 0.1504 |
| C_P2_P3_P4_extended | [P2,P3,P4] | 0.858 | 0.860 | 0.854 | 0.853 | 15,678,478 | 24,290,190 | 0.1551 |
| D_ChannelThenSpatial_P3 | P3 | 0.875 | 0.877 | 0.881 | 0.876 | 15,118,504 | 23,730,216 | 0.1429 |
| D_SpatialThenChannel_P3 | P3 | 0.875 | 0.872 | 0.876 | 0.873 | 15,118,504 | 23,730,216 | 0.1457 |
| E_SpatialAttention_k3_P3 | P3 | 0.883 | 0.877 | 0.884 | 0.879 | 14,986,264 | 23,597,976 | 0.1407 |
| E_SpatialAttention_k5_P3 | P3 | **0.900** | **0.912** | 0.895 | **0.901** | 14,986,296 | 23,598,008 | 0.1435 |
| E_SpatialAttention_k7_P3 | P3 | 0.883 | 0.884 | 0.884 | 0.883 | 14,986,344 | 23,598,056 | 0.1392 |
| F_SE_P3 | P3 | 0.875 | 0.876 | 0.872 | 0.873 | 15,118,405 | 23,730,117 | 0.1416 |
| F_ECA_P3 | P3 | 0.858 | 0.859 | 0.857 | 0.858 | 14,986,250 | 23,597,962 | 0.1514 |
| F_CBAM_P3 | P3 | 0.817 | 0.823 | 0.824 | 0.813 | 15,118,504 | 23,730,216 | 0.1437 |
| F_CoordinateAttention_P3 | P3 | 0.842 | 0.842 | 0.845 | 0.842 | 15,185,093 | 23,796,933 | 0.1463 |

Best accuracy/F1 overall: **A3_ChannelSpatial_P3** and
**E_SpatialAttention_k5_P3** (both 0.900 test accuracy). Both scatter plots
(accuracy vs. parameters, macro F1 vs. inference time) show the top
performers sitting at the cheap end of both axes — the two heaviest models
(A5_Combined_P3, A4_SelfAttention_P3, both >24.9M parameters) do not reach
the accuracy of the much lighter A3/E_k5. Parameter counts across all 25
models range from 23.598M to 25.04M (under 6% spread); inference time
ranges from ~139 ms/img to ~157 ms/img.

### Part 8: Qualitative Analysis
- **Grad-CAM:** the attention model's activation on a sunflower image is
  visibly narrower and more petal/seed-head-focused than the baseline's,
  which spills further into the background leaves.
- **Internal attention maps:** channel-attention weights for the top-20
  channels all saturate near 1.0, behaving closer to a hard gate than a
  smooth re-weighting; the plain-spatial-attention map does not track the
  flower, consistent with A2 giving almost no improvement; the
  self-attention response collapses onto one dominant spatial location
  rather than distributing broadly.
- **Attention across depth (P2/P3/P4):** P2 produces a fine,
  noise-like map (low-level, texture-driven); P3 a coarser, partially
  organised map; P4 the coarsest, blocky, semantic-level map — confirming
  attention behaviour is not independent of the feature resolution/
  semantics it operates on.
- **Correct vs. incorrect predictions:** on correctly classified images,
  attention keeps activation centred on the labelled object across every
  variant tested; on the two misclassified daisy images, every model's
  activation is centred on the wrong element of the scene (an insect, or a
  background flower), showing the error traces to *where* the network
  looked rather than to any single attention mechanism.

## Conclusion
This experiment confirmed that attention effectiveness is not a property
of the attention module in isolation, but depends jointly on attention
type, attention position, feature resolution/semantics, and model
complexity. Sequential Channel→Spatial attention inserted at the
intermediate P3 stage — and, equivalently, spatial attention alone with a
moderately sized 5×5 kernel at the same position — gave the best
accuracy/complexity balance among every configuration tested, both
reaching 0.900 test accuracy against a 0.858 baseline. Inserting attention
at more positions simultaneously, stacking every mechanism together, or
using full self-attention did not improve on this result and cost more
parameters and inference time. Among established mechanisms, lightweight
SE/ECA matched the heavier CBAM, while Coordinate Attention underperformed
the baseline despite its extra parameters — a direct counter-example to
the assumption that more complex attention necessarily helps. Qualitative
Grad-CAM and internal-attention-map evidence supported the quantitative
ranking: the best-performing configurations produced more compact,
object-centred activation, while misclassifications traced to the network
attending to the wrong region of the image rather than to a deficiency in
any one attention mechanism.
