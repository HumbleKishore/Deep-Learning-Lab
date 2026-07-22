# Deep Learning Lab – CS3807

This repository contains all lab experiments for the Deep Learning Laboratory
(CS3807), Shiv Nadar University Chennai, AY 2026–27.

Each experiment is maintained in its own subfolder, with a dedicated README,
source code, dataset information, dependency list and execution instructions.

## Experiments

| # | Experiment | Description | Link |
|---|------------|-------------|------|
| 1 | Single Layer Perceptron | Binary classification on the Banknote Authentication dataset using a perceptron implemented from scratch | [Lab-1-perceptron](./Lab%201%20Perceptron) |
| 2 | Multi-Layer Perceptron | Multi-class image classification on the Fashion-MNIST dataset using an MLP with automated hyperparameter optimization | [Lab-2-MLP](./Lab%202%20MLP) |

More experiments will be added here as the semester progresses.

## Repository Structure
```
deep-learning-lab/
├── README.md
├── experiment-1-perceptron/
│   ├── README.md
│   ├── requirements.txt
│   ├── Lab1_perceptron.ipynb
│   └── data_banknote_authentication.txt
├── experiment-2-mlp/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab_2_MLP.ipynb
├── experiment-3-.../
│   └── ...
```

## General Notes
- Each experiment subfolder is self-contained: it can be cloned, its dependencies
  installed, and its notebook run independently of the others.
- Refer to the README inside each experiment's folder for objective, methodology,
  results and execution instructions specific to that experiment.
