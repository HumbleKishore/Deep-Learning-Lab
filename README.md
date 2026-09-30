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
| 3 | CNN | understand the working principle of Convolutional Neural Networks by implementing convolution, pooling, feature map visualization, and image classification using TensorFlow/Keras. | [Lab-3-MLP](./Lab%203%20CNN) |
| 4 | Transfer Learning | The objective of this experiment are Study the evolution of deep CNN architectures, Compare LeNet-5, AlexNet, VGG16, GoogleNet and ResNet,Understand transfer learning,Fine tune pretrained CNN models,Compare classification performance of different architectures | [Lab-4-Transfer-Learning](Lab%204%20Transfer%20learning) |
| 5 | study-on-CNN-Training |This experiment provides a comprehensive study of Convolutional Neural Network training and optimization using the Oxford-IIIT Pet dataset with images resized to 224×224×3. Students investigate weight initialization, regularization, Batch Normalization, Dropout, optimization algorithms, CNN hyperparameters, transfer learning, and fine-tuning using the MobileNetV2 architecture. Finally, 5-fold cross-validation is used to select the best-performing configuration, followed by evaluation on an independent test set. | [Lab-5-study-on-CNN-Training](Lab%205%20study%20on%20CNN%20Training) |
| 6 | RNN,LSTM-and-GRU | This experiment provides an end-to-end study of sequence learning using RNN, LSTM, and GRU architectures. Students preprocess and represent sequential data, implement the three recurrent models, analyze their training behavior, and evaluate their performance using appropriate classification metrics and visualizations. The experiment also introduces the use of CNN-extracted features with recurrent networks for video understanding and demonstrates sequence-to-sequence learning. Students compare the models based on predictive performance, model complexity, and computational cost, and draw conclusions from the experimental results. | [Lab-6-RNN,LSTM-and-GRU](Lab%206%20RNN%2CLSTM%20and%20GRU) |
| 7 | Autoencoder,Denoising-Autoencoder,VAE | This experiment provides an end-to-end study of autoencoders for image representation, reconstruction, denoising, and generation. Students implement and evaluate fully connected autoencoders, convolutional autoencoders, denoising autoencoders, and variational autoencoders using image data. The experiment focuses on latent representations, reconstruction quality, denoising performance, and the generative capability of VAEs through quantitative evaluation and visual analysis. | [Lab-7-Autoencoder,Denoising-Autoencoder,VAE](Lab%207%20Autoencoder%2CDenoising%20Autoencoder%2CVAE) |

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
├── experiment-3-CNN/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab3.ipynb
├── experiment-4-Transfer-Learning/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab_4.ipynb
├── experiment-5-study-on-CNN-Training/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab_5.ipynb
├── experiment-6-RNN,LSTM-and-GRU/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab_6.ipynb
├── experiment-7-Autoencoder,Denoising-Autoencoder,VAE/
│   ├── README.md
│   ├── requirements.txt
│   └── Lab_7.ipynb
```

## General Notes
- Each experiment subfolder is self-contained: it can be cloned, its dependencies
  installed, and its notebook run independently of the others.
- Refer to the README inside each experiment's folder for objective, methodology,
  results and execution instructions specific to that experiment.
