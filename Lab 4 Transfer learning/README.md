# Experiment 4 – Comparative Study of Deep CNN Architectures Using Transfer Learning
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To study the evolution of deep Convolutional Neural Network (CNN)
architectures — LeNet-5, AlexNet, VGG16, GoogleNet and ResNet — and to
implement transfer learning using a pretrained VGG16 model for multi-class
image classification on the CIFAR-10 dataset. This includes loading
pretrained ImageNet weights, freezing the convolutional base, training a
custom classifier head, fine tuning selected convolution layers, and
evaluating classification performance using standard metrics.

## Dataset
- **Name:** CIFAR-10
- **Source:** Loaded directly via `keras.datasets.cifar10.load_data()`
- **Details:** 50,000 training images and 10,000 testing images, each
  32×32 RGB pixels, 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog,
  Frog, Horse, Ship, Truck).
- **Preprocessing:** Pixel values normalized to the [0,1] range; labels
  one-hot encoded.

## Method

### Part 1: Transfer Learning with VGG16
- A pretrained VGG16 model (ImageNet weights, `include_top=False`) was
  loaded with its convolutional base frozen, taking the raw 32×32×3 CIFAR-10
  images as input.
- A custom classifier head was added: Global Average Pooling → Dense(256,
  ReLU) → Dense(10, Softmax), giving 14,848,586 total parameters (133,898
  trainable with the base frozen).
- The model was compiled with the Adam optimizer (learning rate 0.001),
  categorical cross-entropy loss, and trained for 15 epochs with batch
  size 32.

### Part 2: Fine Tuning
- The last convolutional block of VGG16 (`block5`) was unfrozen and the
  model recompiled with a lower learning rate (0.0001).
- Training continued for 8 additional epochs, after which the classification
  accuracy before and after fine tuning was compared.

### Part 3: Model Evaluation
- The fine-tuned model was evaluated on the CIFAR-10 test set using
  accuracy, precision, recall, F1-score, a confusion matrix and a full
  classification report.

### Part 4: Hyperparameter Study
- Four configurations were trained (5 epochs each, frozen VGG16 base),
  varying learning rate (0.001, 0.0001), batch size (32, 64), optimizer
  (Adam, SGD) and dense layer width (256, 128), to study their effect on
  validation accuracy and training time.

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_4_CNN_Transfer_Learning.ipynb
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
   cd "Lab 4 CNN Transfer Learning"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (CIFAR-10 is
   downloaded automatically via Keras, no separate dataset download needed):
```bash
   jupyter notebook Lab_4_CNN_Transfer_Learning.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory (via the `save_plot()` helper in the notebook), and the console
   will print dataset shapes, model summaries, per-epoch training logs,
   fine-tuning logs, and final evaluation metrics.

## Results

### Part 1 & 2: Transfer Learning and Fine Tuning
With the VGG16 base frozen, training accuracy rose from about 53% to 75%
over 15 epochs, while validation accuracy plateaued around 61–62%,
indicating mild overfitting on the frozen features. After unfreezing
`block5` and fine tuning for 8 further epochs at a lower learning rate,
training accuracy reached 92.69% and test accuracy improved to **74.44%**
— a substantial gain over the frozen-base result, confirming that adapting
the later convolutional layers to CIFAR-10 features improves performance.

### Part 3: Evaluation Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 92.69% (final fine-tuning epoch) |
| Testing Accuracy | 74.44% |
| Precision (weighted avg) | 0.7453 |
| Recall (weighted avg) | 0.7444 |
| F1-score (weighted avg) | 0.7427 |
| Training Time | 2684.63 s (~44.7 min): Phase 1 = 1206.25 s, Phase 2 (fine tuning) = 1478.38 s |
| Total Parameters | 14,848,586 |

The confusion matrix shows the strongest performance on Ship (0.85) and
Automobile (0.83), with the most confusion occurring between visually
similar classes — particularly Cat and Dog, and to a lesser extent Bird
and Deer.

### Part 4: Hyperparameter Study
| Learning Rate | Batch Size | Optimizer | Dense Units | Val. Accuracy | Time (s) |
|---|---|---|---|---|---|
| 0.001 | 32 | Adam | 256 | 0.6095 | 423.63 |
| 0.0001 | 32 | Adam | 256 | 0.5732 | 436.80 |
| 0.001 | 64 | Adam | 128 | 0.5984 | 280.25 |
| 0.001 | 32 | SGD | 256 | 0.4494 | 351.72 |

Within a short 5-epoch window, learning rate and optimizer choice had the
largest impact: the higher learning rate (0.001) and Adam optimizer both
clearly outperformed the lower learning rate (0.0001) and SGD respectively,
while reducing dense units and increasing batch size traded a small amount
of accuracy for noticeably faster training.

## Conclusion
Transfer learning with a frozen pretrained VGG16 base provided a strong
starting point on CIFAR-10, but fine tuning the last convolutional block
was essential to close the gap between training and validation performance,
raising test accuracy from roughly 62% to 74.44%. This confirms that while
generic ImageNet features transfer reasonably well to a new dataset, they
are not fully optimal on their own — allowing the later, more
task-specific layers to adapt further yields a meaningfully better model.
The hyperparameter study further reinforced that optimizer choice and
learning rate are the most influential factors for this task, with Adam
and a moderately high learning rate (0.001) giving the best results within
a limited training budget. Overall, the experiment illustrates why transfer
learning followed by targeted fine tuning is the standard practical
approach for image classification tasks with limited data or compute,
compared to training a deep CNN such as VGG16 entirely from scratch.
