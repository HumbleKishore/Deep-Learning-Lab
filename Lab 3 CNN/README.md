# Experiment 3 – Implementation of Convolutional Neural Networks (CNNs) for Image Classification
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To understand the working principle of Convolutional Neural Networks (CNNs)
by implementing the convolution operation, studying the effect of kernel
size, stride and padding, visualizing feature maps, comparing pooling
strategies, and building a CNN for multi-class image classification on the
CIFAR-10 dataset using TensorFlow/Keras.

## Dataset
- **Name:** CIFAR-10
- **Source:** Loaded directly via `keras.datasets.cifar10.load_data()`
- **Details:** 50,000 training images and 10,000 testing images, each
  32×32 RGB pixels, 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog,
  Frog, Horse, Ship, Truck).
- **Class distribution:** Perfectly balanced, 5,000 training samples per
  class.
- **Preprocessing:** Pixel values normalized to the [0,1] range; labels
  one-hot encoded.

## Method

### Part 1: Dataset Exploration
- Loaded CIFAR-10, displayed ten sample images with their class labels, and
  plotted the class distribution of the training set to confirm balance.

### Part 2: Convolution Fundamentals
- Applied a single Conv2D layer (8 filters, stride 1, `valid` padding) on a
  normalized sample image and compared output feature map sizes for
  3×3, 5×5 and 7×7 kernels.
- Studied the effect of stride (1, 2) and padding (`same`, `valid`) on a
  3×3 convolution layer, and verified the resulting output shapes manually
  using the formula `(N − F + 2P) / S + 1`, confirming an exact match with
  the Keras outputs.

### Part 3: Feature Maps and Pooling
- Built a single-layer model `Conv2D(16, (3,3), activation='relu',
  padding='same')` and visualized the resulting feature maps
  (shape `(1, 32, 32, 16)`).
- Compared Max Pooling and Average Pooling (2×2) on the same feature maps,
  both reducing spatial size from 32×32 to 16×16.

### Part 4: CNN Construction and Training
- Built the following CNN:
  `Input → Conv(32) → ReLU → MaxPool → Conv(64) → ReLU → MaxPool → Flatten → Dense(128) → Softmax`
  (545,098 trainable parameters).
- Trained with the Adam optimizer, categorical cross-entropy loss, 20
  epochs, batch size 32, on normalized pixel values with one-hot encoded
  labels.

### Part 5: Model Evaluation
- Evaluated Accuracy, Precision, Recall, F1-score, Confusion Matrix and
  Classification Report on the CIFAR-10 test set.

### Part 6 (Additional Exercises)
- Manually calculated output feature map size and convolution layer
  parameter counts for two numerical examples.
- Compared ReLU and Sigmoid activation functions analytically and visually.
- Rebuilt the CNN with `AveragePooling2D` in place of `MaxPooling2D` and
  compared training/validation accuracy.
- Trained a smaller two-block CNN with 16 vs 64 filters per block to study
  the accuracy–computation time trade-off.

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_3_CNN.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
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
   cd "Lab 3 CNN"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (CIFAR-10 is
   downloaded automatically via Keras, no separate dataset download needed):
```bash
   jupyter notebook Lab_3_CNN.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory (via the `save_plot()` helper in the notebook), and the console
   will print dataset shapes, feature map sizes, per-epoch training logs,
   and final evaluation metrics.

## Results

### Part 2: Kernel Size, Stride and Padding
| Kernel Size | Output Feature Map Size (stride 1, valid) |
|---|---|
| 3×3 | 30×30 |
| 5×5 | 28×28 |
| 7×7 | 26×26 |

| Stride | Padding | Output Shape |
|---|---|---|
| 1 | same | 32×32 |
| 1 | valid | 30×30 |
| 2 | same | 16×16 |
| 2 | valid | 15×15 |

The manually computed values using `(N − F + 2P) / S + 1` matched the
Keras outputs exactly in every case.

### Part 4 & 5: Main CNN on CIFAR-10
Training accuracy rose steadily from about 0.516 (epoch 1) to about 0.964
(epoch 20), while validation accuracy plateaued around 0.69–0.70 after the
first few epochs and validation loss began rising from around epoch 6
onward — indicating overfitting in the later epochs despite continued
improvement on the training set.

| Metric | Value |
|---|---|
| Training Accuracy | 0.9636 |
| Testing (Validation) Accuracy | 0.6841 |
| Precision (macro) | 0.6890 |
| Recall (macro) | 0.6841 |
| F1-score (macro) | 0.6839 |
| Number of Parameters | 545,098 |

The confusion matrix shows strong performance on Ship (819) and Frog (750),
with the most confusion occurring between Cat and Dog — the largest
single off-diagonal errors in the matrix (260 Cats predicted as Dog and
148 Dogs predicted as Cat) — reflecting their visual similarity at low
resolution.

### Part 6: Additional Exercises
- **Max Pooling vs Average Pooling:** Max pooling reached a higher final
  training accuracy (0.9636) but a similar final validation accuracy
  (0.6841) compared to average pooling (0.9460 training, 0.6834
  validation), indicating max pooling overfits slightly more while both
  generalize comparably.
- **16 vs 64 filters:** Increasing filters per block from 16 to 64 improved
  validation accuracy from 0.6723 to 0.7185, but roughly tripled training
  time (53.19 s → 162.05 s), illustrating the typical accuracy–computation
  trade-off in CNN design.
- **ReLU vs Sigmoid:** ReLU is unbounded, cheap to compute, and avoids
  vanishing gradients for positive inputs, making it preferable in hidden
  layers, while Sigmoid saturates for large \|x\| and is generally reserved
  for binary-classification output layers.

## Conclusion
This experiment demonstrated the core building blocks of CNNs from first
principles — the convolution operation, the effect of kernel size, stride
and padding on output dimensions, and the role of pooling in reducing
spatial size while retaining important features. The full CNN trained on
CIFAR-10 achieved a reasonable test accuracy of 68.41%, but the growing
gap between training and validation performance across epochs highlighted
overfitting, motivating techniques such as regularization, data
augmentation, or transfer learning (explored in later experiments) to
improve generalization. The additional exercises further reinforced key
design trade-offs in CNNs: max vs average pooling affects the degree of
overfitting, increasing the number of filters improves accuracy at a
significant computational cost, and activation function choice (ReLU vs
Sigmoid) has a direct impact on training stability and speed in deep
networks.
