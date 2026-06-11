# 📰 Fake News Detection using BERT

A Natural Language Processing (NLP) project that fine-tunes Google's BERT model to classify news articles as **Fake** or **Real**.

This project demonstrates the complete workflow of applying a pretrained Transformer model to a text classification task, including data preprocessing, tokenization, fine-tuning, evaluation, and inference.

---

## 📌 Project Overview

Fake news has become a major challenge in the digital era. Traditional machine learning approaches often struggle to capture the contextual meaning of text.

In this project, we leverage **BERT (Bidirectional Encoder Representations from Transformers)**, a pretrained Transformer-based language model, to understand the semantic and contextual relationships within news articles and classify them as either:

* **Fake News (0)**
* **Real News (1)**

---

## 🎯 Objectives

* Understand how BERT processes textual data.
* Learn tokenization, embeddings, and fine-tuning.
* Build a real-world NLP classification system.
* Evaluate model performance using standard classification metrics.
* Gain hands-on experience with the Hugging Face Transformers library.

---

## 🛠️ Technologies Used

* Python
* Google Colab
* PyTorch
* Hugging Face Transformers
* Pandas
* Scikit-learn
* NumPy

---

## 📂 Dataset

Dataset used:

**Fake and Real News Dataset**

The dataset contains:

* `Fake.csv` → Fake news articles
* `True.csv` → Real news articles

Each article includes:

* Title
* Text
* Subject
* Date

### Target Labels

| Label | Meaning   |
| ----- | --------- |
| 0     | Fake News |
| 1     | Real News |

---

## 🧠 Model Architecture

The project uses:

### BERT Base Uncased

Specifications:

* 12 Transformer Encoder Layers
* 12 Attention Heads
* Hidden Size: 768
* Parameters: ~110 Million

Pipeline:

```text
News Article
      ↓
BERT Tokenizer
      ↓
Input IDs + Attention Masks
      ↓
BERT Encoder
      ↓
[CLS] Representation
      ↓
Classification Layer
      ↓
Fake / Real
```

---

## 🔄 Workflow

### 1. Data Loading

* Load Fake.csv and True.csv
* Assign labels
* Merge datasets

### 2. Data Preprocessing

* Combine title and article text
* Shuffle data
* Train-test split

### 3. Tokenization

Convert raw text into BERT-compatible tokens using:

```python
BertTokenizer.from_pretrained("bert-base-uncased")
```

---

### 4. Encoding

Generate:

* Input IDs
* Attention Masks

Example:

```python
{
  'input_ids': [...],
  'attention_mask': [...]
}
```

---

### 5. Fine-Tuning BERT

Load pretrained BERT:

```python
BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

Train on the Fake News dataset while updating:

* BERT weights
* Classification head

---

### 6. Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score

---

### 7. Inference

Predict whether a custom news article is fake or real.

Example:

```python
predict_news("Scientists discover a new renewable energy source.")
```

Output:

```text
REAL
```

---

## 📊 Key NLP Concepts Learned

Through this project, the following concepts were explored:

* Transformer Architecture
* Encoder-Only Models
* BERT
* Tokenization
* WordPiece Tokenizer
* Input IDs
* Attention Masks
* Embeddings
* Contextual Representations
* Self-Attention
* Fine-Tuning
* Text Classification

---


## 🚀 Future Improvements

Potential enhancements include:

* Deploying the model using Gradio or Streamlit
* Comparing BERT with DistilBERT
* Hyperparameter tuning
* Cross-validation
* Explainable AI using SHAP/LIME
* Multi-class misinformation classification
* Real-time news verification pipeline

---

## 📚 Learning Outcomes

This project helped in understanding:

* How pretrained language models work
* Why contextual embeddings outperform traditional text representations
* The difference between pretraining and fine-tuning
* Practical implementation of Transformer-based NLP systems

---

## 📁 Repository Structure

```text
Fake-News-Detection-BERT/
│
├── Fake_News_Detection_BERT.ipynb
├── README.md
├── requirements.txt
├── Fake.csv
├── True.csv
└── results/
```
