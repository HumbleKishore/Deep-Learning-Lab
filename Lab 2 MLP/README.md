# Experiment 2 – Multi-Layer Perceptron for Multi-Class Image Classification
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for
multi-class image classification on the Fashion-MNIST dataset. This includes
image preprocessing, model construction, training, evaluation, and automated
hyperparameter optimization using RandomizedSearchCV.

Additionally, as a supplementary task, a single-layer perceptron is implemented
from scratch and trained on the XOR logic gate to demonstrate why a single
linear unit cannot solve non-linearly separable problems, motivating the need
for a Multi-Layer Perceptron.

## Dataset
- **Name:** Fashion-MNIST
- **Source:** Loaded directly via `keras.datasets.fashion_mnist.load_data()`
- **Details:** 60,000 training images and 10,000 testing images, each
  28×28 grayscale pixels, 10 classes (T-shirt/top, Trouser, Pullover, Dress,
  Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), no missing values.
- **Class distribution:** Perfectly balanced, 6000 training samples per class.

## Method

### Part 1: MLP on Fashion-MNIST
- Images were flattened from 28×28 to 784-length vectors and pixel values
  normalized to the [0,1] range. Labels were one-hot encoded.
- A baseline MLP (Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax),
  109,386 trainable parameters) was compiled with the Adam optimizer and
  categorical cross-entropy loss, and trained for 20 epochs with batch size 32.
- Automated hyperparameter optimization was performed using
  `RandomizedSearchCV` (15 candidates, 5-fold cross-validation) together with
  the SciKeras wrapper, over a search space covering hidden layers, hidden
  neurons, learning rate, optimizer, activation function, dropout rate, batch
  size and epochs. The search was run on a 12,000-sample subset of the training
  data to keep it computationally feasible.
- The optimized model was retrained on the full training set using the best
  hyperparameters found, and compared against the baseline.

### Part 2 (Additional Task): Single-Layer Perceptron and the XOR Problem
- A step activation function was implemented, outputting 1 if the weighted
  sum is ≥ 0, else 0.
- A single perceptron was implemented from scratch using the perceptron
  learning rule, with weights and bias initialized to zero and a learning
  rate of 0.1.
- The XOR truth table was used as training data: (0,0)→0, (0,1)→1, (1,0)→1,
  (1,1)→0.
- The perceptron was trained for a maximum of 20 epochs. On every
  misclassified sample, the weights and bias were updated, and the resulting
  decision boundary was plotted.
- After each full epoch, total error was checked; if zero, convergence was
  declared and training stopped, otherwise training continued.
- Inference was performed on all four XOR input combinations using the final
  weights and bias, and the geometric reason XOR cannot be separated by a
  single straight line was analysed.

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_2_MLP.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- scikeras
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
   cd "Lab 2 MLP"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (Fashion-MNIST is
   downloaded automatically via Keras, no separate dataset download needed):
```bash
   jupyter notebook Lab_2_MLP.ipynb
```
4. All plots are saved automatically as `.eps` files in the working directory
   (via the `save_plot()` helper in the notebook), and the console will print
   dataset shapes, model summaries, per-epoch training logs, weight/bias
   update logs for the perceptron task, and final evaluation metrics.

## Results

### Part 1: MLP on Fashion-MNIST
The baseline model achieved a test accuracy of 0.8899 (precision 0.8929,
recall 0.8899, F1-score 0.8903) after 20 epochs, taking 52.58 seconds to train.
The weakest performing classes were Shirt, Pullover and Coat, which were
frequently confused with each other due to their visual similarity.

RandomizedSearchCV found the best hyperparameters to be 2 hidden layers, 256
hidden neurons, learning rate 0.001, Adam optimizer, sigmoid activation, no
dropout, 30 epochs and batch size 32, giving a cross-validation accuracy of
0.8659. Retrained on the full training set, this optimized model achieved a
test accuracy of 0.8920 (precision 0.8931, recall 0.8920, F1-score 0.8917),
taking 117.35 seconds to train.

### Part 2 (Additional Task): Single-Layer Perceptron and XOR
Starting from zero weights and bias with a learning rate of 0.1, the
perceptron underwent 78 weight updates across 20 epochs and never reached
zero total error at the end of any epoch, i.e. it did **not** converge.

From epoch 3 onward, the same four weight updates repeated identically every
epoch — the weight vector cycled through the states
`[−0.1, 0] → [−0.1, 0.1] → [0, 0.1] → [−0.1, 0]` indefinitely, showing the
perceptron was stuck in a perpetual oscillation rather than slowly
converging.

The final weights were `[−0.1, 0.0]` with bias `0.0`. Inference on the four
XOR inputs gave:

| Input | Target | Predicted |
|-------|--------|-----------|
| (0,0) | 0 | 1 (incorrect) |
| (0,1) | 1 | 1 (correct) |
| (1,0) | 1 | 0 (incorrect) |
| (1,1) | 0 | 0 (correct) |

Only 2 of the 4 patterns were classified correctly — the best any single
linear boundary can do on XOR, since the class-0 points (0,0) and (1,1) lie
on one diagonal of the input square while the class-1 points (0,1) and (1,0)
lie on the other, and no single straight line can separate one diagonal from
the other. This is the classical result of Minsky and Papert (1969), which
historically limited early interest in perceptrons until the development of
multi-layer networks.

## Conclusion
Hyperparameter optimization gave a modest improvement over the baseline MLP
(~0.2 percentage points across all metrics), but more than doubled training
time due to the larger hidden layer size and additional epochs. This suggests
the baseline configuration was already reasonably close to optimal for this
architecture. The experiment also highlighted the inherent limitation of MLPs
on image data, since flattening images into vectors discards spatial
relationships between pixels, motivating the use of Convolutional Neural
Networks for image classification tasks.

The additional single-layer perceptron task on XOR reinforced this same
theme from first principles: a single perceptron can only represent one
linear decision boundary and is therefore mathematically incapable of
solving non-linearly separable problems like XOR, regardless of training
duration. This directly motivates the use of a Multi-Layer Perceptron, which
combines multiple hidden units with non-linear activation functions to
construct non-linear decision boundaries — capable of solving XOR, and, by
extension, the far more complex, high-dimensional patterns present in image
data such as Fashion-MNIST.
