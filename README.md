# 📰 News Classification by Theme

This project classifies news articles into predefined themes using Machine Learning models and provides a web interface through FastAPI and a simple frontend.

---

## 📁 Project Structure
projectNLP/
├── data.py             # Data preprocessing
├── artifacts/          # Trained ML models
│   ├── model.pkl
│   ├── model_lstm.pth
│   ├── tokenizer.pkl
│   ├── vectorizer.pkl
│   └── labels.pkl
├── backend/            # FastAPI + training scripts
│   ├── server.py
│   ├── train.py
│   └── utils.py
├── frontend/           # Web interface
│   ├── index.html
│   ├── script.js
│   └── style.css
├── README.md
└── .gitignore
---

## 🔧 Tech Stack

- Python
- FastAPI
- Scikit-learn
- PyTorch
- HTML / CSS / JavaScript

---

## 🚀 How to Run

### Backend (API)

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload

•	The API will run at: http://127.0.0.1:8000/
•	Test endpoints via: http://127.0.0.1:8000/docs

Frontend (Web Interface)
•	Open the file frontend/index.html in your browser.
•	The frontend interacts with the backend API for predictions
