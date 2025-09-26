# 🤖 Conversational AI: A Deep Dive into Multi-Label Emotion Recognition

![Project Banner](placeholder_for_banner_image.png)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/) [![Hugging Face Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)](https://huggingface.co/transformers/) [![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 1. Title & Introduction

**Project Introduction & Motivation:**

In today's rapidly evolving tech landscape, understanding human emotions through text is more critical than ever. This project moves beyond traditional sentiment analysis by implementing a nuanced multi-label emotion classification system. By leveraging advanced deep learning techniques and state-of-the-art pre-trained models, the project addresses the challenges of ambiguous emotional expressions and overlapping sentiments, making it ideal for modern conversational AI applications.

**Core Objectives & Achievements:**

- Classifying text into 9 distinct emotion categories.
- A comparative analysis of BERT and RoBERTa models.
- Conducting rigorous experiments with different configurations.
- Engineering an advanced prediction threshold optimization to significantly boost the F1-score.

## 2. Technical Blueprint

### Tech Stack

| Technology                    | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| Python 3.8+                   | Core programming language                             |
| PyTorch                       | Deep learning framework                               |
| Hugging Face Transformers     | Pre-trained models and NLP utilities                  |
| Pandas                        | Data manipulation and analysis                        |
| NumPy                         | Numerical computation                                 |
| Scikit-learn                  | Machine learning utilities and evaluation             |
| Jupyter Notebooks             | Interactive data exploration and visualization        |

### Repository Structure

```
.
├── data/
│   ├── raw/          # Contains the original GoEmotions dataset files.
│   └── processed/    # Contains data after preprocessing and label mapping.
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb
│   └── 2_RoBERTa_Finetuning_Optimized.ipynb
├── experiments/
│   ├── BERT_lr_3e-5.ipynb
│   ├── BERT_lr_5e-5.ipynb
│   └── RoBERTa_lr_3e-5.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The above structure illustrates a clear separation of raw and processed data, interactive experiments, and notebooks used for step-by-step analysis.

## 3. The Journey & Results

**Project Pipeline:**  
Data Ingestion → Preprocessing → Model Experimentation → Threshold Optimization → Final Evaluation

**Data Sourcing & Preprocessing:**

Utilizing the GoEmotions dataset, we mapped the original fine-grained labels to 9 broader emotion categories to focus on capturing nuanced sentiments effectively.

**Modeling & Experimentation:**

We started with `bert-base-uncased` as our baseline and experimented with `roberta-base` as an optimized alternative. The experiments provided valuable insights into learning rates and the effect of different prediction thresholds.

**Experimentation Log:**

| **Run ID** | **Model**         | **Learning Rate** | **Prediction Threshold** | **Key Metric (Macro F1)** | **Notes / Notebook Link**                |
| :--------- | :---------------- | :--------------:  | :----------------------: | :-----------------------: | :---------------------------------------- |
| `exp-01`   | BERT-base         | `3e-5`          | `0.5` (Default)          | `0.453`                 | Baseline BERT performance.                |
| `exp-02`   | BERT-base         | `5e-5`          | `0.5` (Default)          | `0.445`                 | Higher LR was not beneficial with default threshold. |
| `exp-03`   | RoBERTa-base      | `3e-5`          | `0.5` (Default)          | `0.454`                 | RoBERTa shows a slight edge over BERT baseline. |
| `exp-04`   | RoBERTa-base      | `5e-5`          | `0.5` (Default)          | `0.454`                 | Higher LR with default threshold was still not optimal. |
| `exp-05`   | **RoBERTa-base**  | **`5e-5`**      | **`0.26` (Optimized)**   | **🏆 0.498**            | **Best performing model after optimization.** |

**The Differentiator - Prediction Threshold Optimization:**

While a default threshold of 0.5 is commonly used, it often falls short for multi-label tasks where class probabilities overlap. Through systematic optimization, we identified 0.26 as the optimal threshold, dramatically enhancing our model's macro F1-score.

**Final Model Performance:**

```
               precision    recall  f1-score   support

      Joyful       0.49      0.62      0.54      1430
Affectionate       0.56      0.77      0.65      4070
Positive_Outlook       0.39      0.29      0.33       539
Anger_Frustration       0.46      0.65      0.54      2812
Sadness_Disappointment       0.47      0.52      0.50      1177
     Fear_Anxiety       0.38      0.42      0.40       401
Surprise_Confusion       0.48      0.62      0.54      2083
     Desire       0.40      0.42      0.41       224
    Neutral       0.49      0.68      0.57      5600

     micro avg       0.50      0.65      0.56     18336
     macro avg       0.46      0.56      0.50     18336
  weighted avg       0.49      0.65      0.56     18336
   samples avg       0.53      0.65      0.57     18336
```

## 4. How to Reproduce

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebooks or Scripts:**  
   Use Jupyter Notebook or your preferred IDE to run the analysis notebooks and scripts.

## 5. Future Scope

- Experimenting with larger models (e.g., `roberta-large`).
- Using automated hyperparameter tuning tools like Optuna.
- Implementing ensemble methods to further enhance prediction accuracy.

## 6. Closing

Thank you for exploring this project. We welcome contributions and collaboration to continue advancing the state-of-the-art in emotion recognition technology. If you have any questions or suggestions, please open an issue or submit a pull request.
