# Experiment 1 – Single Layer Perceptron for Binary Classification

CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To understand the concept of an artificial neuron and implement a Single Layer
Perceptron from scratch for binary classification. This includes understanding
the perceptron learning algorithm, the role of the step activation function,
visualizing the learning process, and evaluating the classifier on a real-world
dataset.

## Dataset
- **Name:** Banknote Authentication Dataset
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/267/banknote+authentication
- **File:** `data_banknote_authentication.txt` (included in this repository, same
  folder as the notebook)
- **Details:** 1372 samples, 4 numerical features (variance, skewness, curtosis,
  entropy), binary class label (0 = authentic, 1 = forged), no missing values.
  The raw file has no header row; column names are assigned manually in the
  notebook while loading the data.
- **Class distribution:** 762 authentic, 610 forged.

## Method
- Features were standardized using `StandardScaler`.
- Data was split into 1097 training samples and 275 testing samples (80/20 split).
- A perceptron was implemented from scratch with a step activation function and
  trained using the perceptron learning rule (weight update `w = w + lr*(y - y_pred)*x`)
  for 50 epochs with a learning rate of 0.01.
- The effect of learning rate (0.001, 0.01, 0.1) on convergence was also studied,
  along with a comparison against Scikit-learn's built-in `Perceptron`.

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab1_perceptron.ipynb
└── data_banknote_authentication.txt
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
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
   cd "Lab 1 Perceptron"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (the dataset file is already
   in the same folder, so no separate download step is needed):
```bash
   jupyter notebook Lab1_perceptron.ipynb
```
4. All plots are saved automatically as `.eps` files in the working directory
   (via the `save_plot()` helper in the notebook) and the console will print
   per-epoch training logs, final weights/bias, and test set metrics.

## Results
Training misclassifications dropped sharply from 47 in epoch 1, then oscillated
between roughly 13 and 21 for the remaining epochs without reaching zero,
indicating the two classes are not perfectly linearly separable.

- Final weights: `[-0.1966, -0.2194, -0.2107, -0.0185]`
- Final bias: `-0.10`

Test set performance:
- Accuracy: 0.9782
- Precision: 0.9840
- Recall: 0.9685
- F1-score: 0.9762

Learning rate comparison (0.001, 0.01, 0.1) produced identical error curves per
epoch, since the step activation only depends on the sign of the weighted sum,
not its magnitude — learning rate only affects the final magnitude of the
learned weights/bias, not convergence behaviour.

Scikit-learn's `Perceptron` trained on the same split gave an accuracy of 0.9891,
close to the from-scratch implementation's 0.9782, confirming the custom
implementation behaves correctly.

## Conclusion
The perceptron learned a reasonably good linear decision boundary for this
dataset, achieving over 97% test accuracy despite not fully converging during
training. This shows the two classes are mostly, but not perfectly, linearly
separable. The experiment also confirmed that learning rate has no effect on
this perceptron's convergence pattern, and that the custom implementation
matches a standard library implementation in performance, motivating the need
for multilayer networks to handle non-linearly separable problems like XOR.
