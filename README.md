# DocSearch Information Retrieval System

An interactive Information Retrieval (IR) platform built with **Python** and **Streamlit**.  
The application demonstrates core IR concepts including preprocessing pipelines, inverted indexing, phrase queries, tolerant retrieval, ranking models, and retrieval analytics through an intuitive UI.

---

## Features

- Text preprocessing pipeline
  - Lowercasing
  - Stop-word removal
  - Punctuation removal
  - Hyphen normalization
  - Stemming / Lemmatization

- Inverted index generation

- Boolean and phrase query processing

- Ranked retrieval models
  - TF-IDF
  - BM25

- Tolerant retrieval
  - K-Gram indexing
  - Spelling correction
  - Edit distance suggestions

- Retrieval analytics dashboard
  - Vocabulary statistics
  - Query performance insights
  - Retrieval quality metrics

- Interactive Streamlit UI

---

## Project Structure

```text
DocSearch-IR-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── sample_dataset.csv        # 8 documents
│   ├── sample_technology.csv     # 15 documents
│   └── sample_default.csv        # 10 documents
│
├── assets/
│   └── screenshots/
│
└── modules/
    ├── preprocessing.py
    ├── indexing.py
    ├── retrieval.py
    ├── tolerant_retrieval.py
    └── analytics.py
```

---

## Sample Datasets

The project currently includes three built-in datasets:

| Dataset | Documents |
|---|---|
| `sample_dataset.csv` | 8 |
| `sample_technology.csv` | 15 |
| `sample_default.csv` | 10 |

You can also upload your own CSV dataset directly from the application UI.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/2025aa05012devvissuvamsi/bits-wilp-sem2-ir-grp-32-assignment-1.git
cd bits-wilp-sem2-ir-grp-32-assignment-1
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

After running, Streamlit will open the application in your browser.

---

## Deployed Application

Live Demo:  
`https://bits-wilp-sem2-ir-grp-32-assignment-1.streamlit.app`

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- NLTK
- Scikit-learn

---

## Educational Purpose

This project is designed for learning and demonstrating fundamental concepts in:

- Information Retrieval
- Search Engines
- Text Processing
- Ranking Algorithms
- Retrieval Evaluation

---
