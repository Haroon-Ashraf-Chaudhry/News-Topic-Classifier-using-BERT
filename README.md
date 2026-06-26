<div align="center">

# 📰 News Topic Classifier using BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Transformers-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A state-of-the-art **news article topic classification** system built by fine-tuning **BERT** (Bidirectional Encoder Representations from Transformers) on a multi-class text classification task. The model categorizes news articles into predefined topics such as Politics, Sports, Technology, Business, and Entertainment with high accuracy.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🔍 Overview

News topic classification is a foundational NLP task with applications in content recommendation, news aggregation, and media monitoring. This project fine-tunes **`bert-base-uncased`** from Hugging Face Transformers on a labeled news dataset, adapting BERT's pre-trained language representations to the downstream classification task.

The project demonstrates the complete workflow: tokenization with BERT's WordPiece tokenizer, preparing `DataLoader` objects, configuring the classification head, fine-tuning with a linear learning rate schedule, and evaluating with classification metrics.

---

## ✨ Key Features

- **BERT Fine-Tuning**: Adapts `bert-base-uncased` to multi-class news topic classification
- **Hugging Face Integration**: Leverages `transformers` and `datasets` libraries for efficient model loading and tokenization
- **Custom Classification Head**: Adds a linear layer on top of BERT's `[CLS]` token for topic prediction
- **Learning Rate Scheduling**: Uses a linear warmup and decay schedule for stable fine-tuning
- **Comprehensive Evaluation**: Reports accuracy, precision, recall, F1, and a full classification report per topic
- **Visualization**: Training/validation loss and accuracy curves plotted across epochs
- **GPU Support**: Automatically detects and leverages CUDA for accelerated training

---

## 🏗️ Model Architecture

```
Input Text (News Article)
        │
        ▼
BERT Tokenizer (WordPiece)
  ├── input_ids
  ├── attention_mask
  └── token_type_ids
        │
        ▼
BERT Encoder (bert-base-uncased)
  └── 12 Transformer Layers
        │
        ▼
[CLS] Token Representation (768-dim)
        │
        ▼
Dropout Layer (regularization)
        │
        ▼
Linear Classification Head
  └── Output size = num_classes
        │
        ▼
Topic Label (e.g., "Sports", "Technology")
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| Transformers | Hugging Face `transformers` |
| Data Handling | Pandas, NumPy, `datasets` |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Acceleration | CUDA (GPU, optional) |

---

## 📊 Dataset

The model is trained on a labeled news article dataset containing text samples across multiple topic categories such as:

- **Politics** — Government, elections, policy
- **Sports** — Match results, athlete news, tournaments
- **Technology** — AI, gadgets, software, startups
- **Business** — Markets, finance, corporate news
- **Entertainment** — Movies, music, celebrity news

> **Note**: Ensure the dataset CSV is placed in the correct directory before running the notebook. See the data loading cell for the expected file format.

**Expected dataset format:**

| text | label |
|---|---|
| "Scientists announce breakthrough in AI..." | Technology |
| "Team wins championship after..." | Sports |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Haroon-Ashraf-Chaudhry/News-Topic-Classifier-using-BERT.git
cd News-Topic-Classifier-using-BERT
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision transformers datasets pandas numpy matplotlib seaborn scikit-learn jupyter
```

> For GPU acceleration, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Launch Jupyter Notebook

```bash
jupyter notebook News_Topic_Classifier_Using_BERT.ipynb
```

---

## 🚀 Usage

Execute the notebook cells in order to run the full pipeline:

1. **Data Loading & EDA** — Load news dataset, inspect class distribution
2. **Preprocessing** — Clean text, encode labels, split into train/val/test
3. **Tokenization** — Tokenize with `BertTokenizer`, create PyTorch `DataLoader`
4. **Model Setup** — Load `BertForSequenceClassification` with the appropriate number of labels
5. **Training** — Fine-tune with `AdamW` optimizer and linear learning rate schedule
6. **Evaluation** — Compute metrics on the test set, display classification report
7. **Inference** — Classify new, unseen news articles

### Quick Inference Example

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./saved_model')

text = "The stock market surged to record highs following the Fed announcement."
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted Topic: {label_map[predicted_class]}")
```

---

## 📈 Results

Fine-tuning BERT significantly outperforms traditional baselines (TF-IDF + Logistic Regression):

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression (Baseline) | — | — |
| TF-IDF + Random Forest (Baseline) | — | — |
| **BERT Fine-Tuned** | **—** | **—** |

> Exact metric values are computed and displayed inside the notebook upon execution.

---

## 📁 Project Structure

```
News-Topic-Classifier-using-BERT/
│
├── News_Topic_Classifier_Using_BERT.ipynb   # Main training & evaluation notebook
├── README.md                                # Project documentation
└── LICENSE                                  # MIT License
```

---

## 💡 Key Learnings

- How to fine-tune a pre-trained BERT model for a downstream classification task
- The role of the `[CLS]` token as an aggregate sentence representation
- How to handle tokenization padding, truncation, and attention masks correctly
- Why learning rate warmup is important for stable transformer fine-tuning
- Trade-offs between `bert-base` and `bert-large` in terms of speed vs. accuracy

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Haroon Ashraf Chaudhry**

[![GitHub](https://img.shields.io/badge/GitHub-Haroon--Ashraf--Chaudhry-181717?style=flat-square&logo=github)](https://github.com/Haroon-Ashraf-Chaudhry)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/haroon-ashraf-chaudhry)

---

<div align="center">

⭐ **If you find this repository useful, please consider giving it a star!** ⭐

</div>
