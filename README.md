# Speech Emotion Recognition Using Wav2Vec2 and Linear SVM

A Speech Emotion Recognition (SER) project using pretrained **wav2vec 2.0** representations with a **Linear Support Vector Machine (LinearSVC)** classifier.

The project studies how pretrained speech representations can be used to recognize emotions from speech while keeping the downstream classification pipeline lightweight and interpretable.

## Project Overview

The system classifies speech into six emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad

The project uses two emotional speech datasets:

* **RAVDESS**
* **CREMA-D**

Both datasets are mapped to the same six-class emotion space.

## Methodology

The overall pipeline is:

```text
Input .wav Audio
       ↓
Mono Conversion
       ↓
16 kHz Resampling
       ↓
Peak Normalization
       ↓
Pretrained wav2vec 2.0
       ↓
Temporal Pooling
       ↓
StandardScaler
       ↓
LinearSVC
       ↓
Predicted Emotion
```

The wav2vec 2.0 encoder is kept frozen and used as a feature extractor. The resulting frame-level representations have 768 dimensions.

Four temporal pooling strategies are evaluated:

1. Mean
2. Max
3. Mean + Standard Deviation
4. Mean + Standard Deviation + Max

## Experiments

### 1. RAVDESS In-Domain

The model is trained using speakers 1–20 and evaluated on speakers 21–24.

**Accuracy:** 59.66%
**Macro-F1:** 0.5633

### 2. CREMA-D In-Domain

The dataset is divided using speaker-independent training and testing splits.

**Accuracy:** 58.31%
**Macro-F1:** 0.5710

### 3. Cross-Domain Evaluation

The model is trained on CREMA-D and tested on RAVDESS.

**Accuracy:** 33.81%
**Macro-F1:** 0.3342

This experiment demonstrates the difficulty of generalizing speech emotion recognition models across different datasets and recording conditions.

## Pooling Study

The four pooling strategies were compared on RAVDESS.

| Pooling Method   | Accuracy | Macro-F1 |
| ---------------- | -------: | -------: |
| Mean             |   66.48% |   0.6166 |
| Max              |   50.00% |   0.4835 |
| Mean + Std       |   59.66% |   0.5633 |
| Mean + Std + Max |   63.64% |   0.6194 |

The experiments show that increasing the feature size does not necessarily improve classification accuracy.

## Final Model

A final LinearSVC model is trained using mean-pooled wav2vec 2.0 representations from RAVDESS and a sampled subset of CREMA-D.

A prediction function is also implemented that accepts a `.wav` file and returns a predicted emotion label.

```python
emotion = predict_emotion("audio.wav")
print(emotion)
```

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* wav2vec 2.0
* scikit-learn
* librosa
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

## Datasets

The project uses:

* RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
* CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset

The datasets are not included in this repository.

## Project Limitations

* The wav2vec 2.0 encoder is frozen.
* The datasets contain acted emotional speech.
* The classifier is linear.
* Cross-domain evaluation is limited to CREMA-D → RAVDESS.
* No data augmentation, attention-based pooling, or domain adaptation is used.

## Future Work

Possible extensions include:

* Fine-tuning wav2vec 2.0 for emotion recognition
* Attention-based temporal pooling
* Domain adaptation
* Multi-corpus training
* More advanced nonlinear classifiers
* Deployment as a real-time speech emotion recognition system

## Authors

**Jaskaran Singh**
Indian Institute of Technology Ropar

**Anand Patel**
Indian Institute of Technology Ropar

