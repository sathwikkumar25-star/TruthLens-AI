# 🔎 TRUTHLENS AI
### Evidence-Based AI Evaluation & Verification Platform

**Every claim deserves evidence.**

TRUTHLENS AI is a Python-powered research project that uses Natural Language Inference (NLI) to evaluate whether a claim is supported by, contradicted by, or not established by a reference passage.

The goal is to explore how AI-generated statements can be checked against evidence instead of being accepted blindly.

## ✨ Features

- **Evidence-based claim verification** — compares a claim with a supplied reference passage.
- **Three-way classification** — predicts entailment, contradiction, or neutral.
- **Prediction score visualization** — displays the model's scores for each category.
- **Interactive web interface** — built with Streamlit.
- **Automated evaluation** — includes an initial test suite for measuring classification performance.
- **Local model inference** — runs a pretrained model through Hugging Face Transformers and PyTorch.

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| PyTorch | Model inference |
| Hugging Face Transformers | Tokenization and model loading |
| RoBERTa Large MNLI | Natural Language Inference |
| Pandas | Score presentation |
| Scikit-learn | Evaluation metrics |

## 🧠 How It Works

1. The user enters a reference passage.
2. The user enters a claim to evaluate.
3. The tokenizer prepares the evidence and claim for the model.
4. The pretrained NLI model predicts the relationship between the two texts.
5. TRUTHLENS AI displays the predicted category and classification scores.

### Possible Results

- **Entailment:** The model predicts that the evidence supports the claim.
- **Contradiction:** The model predicts that the evidence conflicts with the claim.
- **Neutral:** The model predicts that the evidence does not establish the claim.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd TruthLens-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the web application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The pretrained model must be downloaded on the first run, so an internet connection is required.

## 🧪 Evaluation

The initial evaluation script tests six example evidence–claim pairs across the three NLI categories.

The first run achieved 6/6 correct predictions on those examples. This is only a preliminary result on a very small dataset; it is not a reliable estimate of real-world accuracy.

The next development goal is to expand the dataset and evaluate generalization, precision, recall, F1 score, and common failure cases.

## ⚠️ Limitations

- The user supplies the reference evidence.
- The application does not independently verify source credibility.
- A claim can be true even when the supplied passage does not establish it.
- Model predictions and classification scores are not guarantees of factual correctness.
- The current evaluation dataset is too small to establish real-world reliability.

TRUTHLENS AI is an experimental research and learning project, not a substitute for independent fact-checking.

## 🛣️ Roadmap

- [x] Build a Python claim-verification prototype.
- [x] Integrate a pretrained NLI model.
- [x] Create an interactive Streamlit interface.
- [x] Add initial automated evaluation.
- [ ] Expand and improve the evaluation dataset.
- [ ] Add evidence retrieval from multiple documents.
- [ ] Improve source attribution and evidence traceability.
- [ ] Deploy a public demo.
- [ ] Add further robustness and error analysis.

## 👨‍💻 Author

Developed as a learning project in Python, AI engineering, and natural language processing.

**Project:** TRUTHLENS AI  
**Tagline:** Every claim deserves evidence.