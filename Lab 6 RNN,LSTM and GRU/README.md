# Experiment 6 – End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding
CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai

## Objective
To develop an end-to-end understanding of recurrent sequence learning by
implementing and comparing Vanilla RNN, LSTM and GRU models. The experiment
also introduces Backpropagation Through Time (BPTT), the limitations of
conventional RNNs, and the use of CNN-extracted features with recurrent
networks for video understanding. The UCI Human Activity Recognition dataset
is the primary sequence dataset, the pipeline is extended to video
understanding using MobileNetV2 features followed by an LSTM, and a small
synthetic sequence-reversal task demonstrates the encoder–decoder framework.

## Datasets

### Primary: UCI Human Activity Recognition Using Smartphones
- **Source:** Official UCI HAR release (downloaded automatically by the notebook).
- **Details:** Raw inertial signal files (not the pre-computed 561-feature
  vectors). Each window has 128 time steps across 9 channels (body
  acceleration, gyroscope and total acceleration, each in 3 axes), so
  $X \in \mathbb{R}^{N \times 128 \times 9}$. Six activities: WALKING,
  WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
- **Raw shapes:** Train signals (7352, 128, 9); test signals (2947, 128, 9).
- **Subset used:** 330 windows per class drawn (stratified) from the training
  partition, giving a pool of 1,980 windows split 70/15 into training and
  validation sets. The official test partition (2,947 windows) was kept
  untouched as the independent test set.
- **Preprocessing:** Normalization using statistics computed only from the
  training data.

| Split | Shape |
|---|---|
| Training | (1631, 128, 9) |
| Validation | (349, 128, 9) |
| Test | (2947, 128, 9) |

### Video: UCF101 subset
- **Source:** `sayakpaul/ucf101-subset` (~171 MB) downloaded via
  `huggingface_hub`, used in place of the full ~6.5 GB UCF101 release.
- **Classes (5):** Basketball, BaseballPitch, BabyCrawling, BandMarching,
  BenchPress (chosen by video count), giving 228 videos.
- **Preprocessing:** 10 frames sampled uniformly per video, resized to
  224×224×3 and passed through MobileNetV2's `preprocess_input`.

### Synthetic: Sequence reversal
- 5,000 random digit sequences (length 4, values 0–9) whose target is the
  reversed input, e.g. `[1, 4, 7, 2] → [2, 7, 4, 1]`.

## Method

### Part 1: Temporal Data Visualization and Numerical Exercise
- Three channels (body acc x, body gyro x, total acc x) were plotted for one
  window each from WALKING, SITTING and LAYING.
- A worked Vanilla RNN example ($x = [0.5, 0.7, 0.2]$, $W_x = 0.5$,
  $W_h = 0.8$, $b = 0.1$) was computed in code and verified against the
  hand calculation: $h_1 = 0.3364$, $h_2 = 0.6164$, $h_3 = 0.6000$.

### Part 2: RNN, LSTM and GRU Classifiers
- Identical classifier for all three models: recurrent layer (32 units) →
  Dropout (0.2) → Dense(16, ReLU) → Dense(6, softmax). Only the recurrent
  layer (`SimpleRNN`, `LSTM`, `GRU`) differs.
- Adam (lr $10^{-3}$), sparse categorical cross-entropy, batch size 32,
  30 epochs.
- Evaluated on the independent test set using accuracy, macro precision,
  recall and F1, confusion matrices, parameter count and training time.

### Part 3: Effect of Sequence Length
- Models retrained (20 epochs) after truncating every window to the first
  $T \in \{32, 64, 128\}$ time steps.

### Part 4: Video Understanding (CNN + LSTM)
- A frozen MobileNetV2 (ImageNet weights, global average pooling) extracts a
  1280-dimensional feature vector per frame, giving inputs of shape
  (228, 10, 1280). Features are extracted once; the CNN is never trained.
- Classifier: LSTM(32) → Dense(16, ReLU) → Dense(5, softmax), trained for
  30 epochs with batch size 8. Sampled frames, loss/accuracy curves, a
  confusion matrix and an example prediction were produced.

### Part 5: Sequence-to-Sequence Learning
- Encoder LSTM (latent dimension 32) compresses the input into its final
  hidden and cell states; decoder LSTM (latent dimension 32) generates the
  output one token at a time, trained with teacher forcing (40 epochs, batch
  size 64) and evaluated with step-by-step decoding (no teacher forcing).

### Part 6: Additional Exercises
1. LSTM recurrent-unit variation (16 / 32 / 64)
2. GRU vs. LSTM with equal units (32)
3. Stacked two-layer LSTM
4. Bidirectional vs. unidirectional LSTM
5. Effect of sequence length on computational cost
6. Video LSTM vs. GRU on identical CNN features
7. Variable-length seq2seq (length-4 input → length-2 output)

## Repository Structure
```text
├── README.md
├── requirements.txt
├── Lab_6.ipynb
```

## Dependencies
Listed in `requirements.txt`:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- scikit-learn
- opencv-python
- huggingface_hub
- jupyter

Install with:
```bash
pip install -r requirements.txt
```

## Execution Instructions
1. Clone this repository:
```bash
   git clone https://github.com/HumbleKishore/Deep-Learning-Lab.git
   cd "Lab 6 RNN, LSTM and GRU"
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch Jupyter and run the notebook top to bottom (the UCI HAR dataset
   and the UCF101 subset are downloaded automatically, so an internet
   connection is required on the first run):
```bash
   jupyter notebook Lab_6.ipynb
```
4. All plots are saved automatically as `.eps` files in the working
   directory (via the `save_plot()` helper in the notebook), and the console
   will print dataset shapes, class distributions, per-epoch training logs,
   test metrics, and training times.

## Results

### Part 1: Temporal Patterns and Numerical Exercise
WALKING shows a clear repeating oscillation from footsteps, while LAYING and
SITTING are almost flat. Static postures (SITTING, STANDING, LAYING) are hard
to tell apart from signal shape alone, and shuffling the time steps would
destroy the information that defines each activity.

### Part 2: RNN vs. LSTM vs. GRU (UCI HAR test set, N = 2947)
| Metric | RNN | LSTM | GRU |
|---|---|---|---|
| Accuracy (%) | 73.53 | 88.46 | 84.36 |
| Macro Precision (%) | 73.76 | 88.90 | 84.58 |
| Macro Recall (%) | 73.19 | 88.55 | 84.37 |
| Macro F1 (%) | 73.13 | 88.43 | 84.13 |
| Parameters | 1,974 | 6,006 | 4,758 |
| Training Time (s) | 13.5 | 26.8 | 41.2 |

LSTM converged most smoothly and reached the best validation and test
performance, GRU behaved similarly but settled slightly lower, and the
vanilla RNN plateaued earlier and less smoothly. All three showed a mild
train/validation gap, smallest for LSTM. In the confusion matrices LAYING is
recognised almost perfectly, SITTING/STANDING is the most consistent
confusion across all models, and WALKING_UPSTAIRS/WALKING_DOWNSTAIRS are also
mutually confused.

### Part 3: Effect of Sequence Length (Macro F1, %)
| Sequence Length T | RNN | LSTM | GRU |
|---|---|---|---|
| 32 | 58.79 | 84.93 | 82.57 |
| 64 | 69.48 | 85.17 | 84.44 |
| 128 | 66.73 | 86.88 | 84.21 |

LSTM benefits most consistently from longer context, while the plain RNN's
F1 drops from T = 64 to T = 128, consistent with vanishing gradients.

### Part 4: CNN–LSTM Video Model
| Metric | Value |
|---|---|
| Accuracy | 95.65% |
| Macro Precision | 96.36% |
| Macro Recall | 95.56% |
| Macro F1 | 95.65% |
| Parameters | 168,677 |

The confusion matrix is dominated by a strong diagonal, and the training and
validation curves rise together with very little gap. The CNN captures
per-frame spatial appearance while the LSTM models how those features evolve
across the 10-frame sequence.

### Part 5: Sequence-to-Sequence Reversal
| Metric | Value |
|---|---|
| Token Accuracy | 0.4930 |
| Sequence Accuracy | 0.0140 |
| Final Training Loss | 0.0077 |
| Final Validation Loss | 0.0079 |
| Trainable Parameters | 11,627 |

Teacher-forced loss converges to a very small value, but free-running
inference is much harder: a single early mistake cascades, and sequence
accuracy is far below token accuracy because every token in a sequence must
be correct.

### Part 6: Additional Exercises
- **LSTM units (accuracy):** 16 → 0.8273, 32 → 0.8371, 64 → 0.8802 (parameters
  grow from 2,038 to 20,086).
- **GRU vs. LSTM (32 units):** LSTM 0.8846 (6,006 params) vs. GRU 0.8436
  (4,758 params).
- **Stacked LSTM:** F1 0.8561 (14,326 params, 52.7 s), lower than the
  single-layer 0.8843.
- **Bidirectional LSTM:** F1 0.8235 (11,894 params), lower than the
  unidirectional 0.8843.
- **Video LSTM vs. GRU:** both reached F1 = 0.9565 on identical CNN features.
- **Variable-length seq2seq (4 → 2):** token and sequence accuracy both
  reached 1.0000.

## Conclusion
This experiment demonstrated the complete sequence-learning pipeline
(sequential data → preprocessing → RNN/LSTM/GRU → evaluation), the
video-understanding pipeline (video → frames → CNN features → LSTM/GRU →
action prediction), and the encoder–decoder formulation. LSTM gave the best
accuracy and F1 on the HAR task (88.43% macro F1) at a modest parameter cost
over GRU, while the vanilla RNN lagged behind and benefited least from longer
sequences. The frozen-CNN + LSTM pipeline generalised well to a small
five-class video task (95.65% accuracy), and the encoder–decoder model
learned the synthetic reversal task, illustrating the sharp difference
between token-level and sequence-level accuracy as well as the flexibility of
encoder–decoder models to handle input and output sequences of different
lengths.
