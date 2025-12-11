🏡 California House Price Prediction — ML + Flask + SHAP

A complete end-to-end Machine Learning project that predicts California house prices using a trained LightGBM regression model, served through a Flask backend, with a clean HTML/CSS/JavaScript frontend.
The project also includes optional SHAP explainability to interpret model predictions.

🚀 Features
🔧 Machine Learning
Optuna use

LightGBM regression model

Fully preprocessed using Scikit-Learn Pipeline

Accurate prediction of California housing median values

📊 Model Explainability (Optional)

SHAP KernelExplainer (lightweight for low-memory environments)

Bar plot showing Top 10 influential features

SHAP visualization sent to frontend as Base64 image

🌐 Backend (Flask API)

/predict endpoint for model inference

JSON input/output

CORS enabled

Error-handled, stable API

🖥️ Frontend

Modern UI (HTML + CSS + JS)

Async API calls

Displays prediction instantly

Shows SHAP plot (if enabled)

🏗️ Project Structure
📁 project-root
│── app.py
│── model.pkl
│── pipeline.pkl
│── requirements.txt
│── README.md
│
├── templates/
│     └── index.html
│
└── static/
      ├── script.js
      └── style.css

🔥 How It Works

User fills in housing details

JavaScript sends them to Flask API

Backend:

Preprocesses with the saved pipeline

Predicts using LightGBM model

(Optional) Generates SHAP explanations

Frontend displays:

Predicted house value

SHAP feature importance plot

🛠️ Local Setup
1. Clone this repository
git clone https://github.com/subal43/california-house-price-prediction.git
cd california-house-price-prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the Flask server
python app.py

4. Visit the application

➡️ http://127.0.0.1:5000/

⚠️ SHAP Note

SHAP explainability can require significant memory depending on the model.
To avoid timeouts on low-memory environments:

SHAP generation can be disabled if needed

Even with SHAP off — predictions will work perfectly.

🧠 Technologies Used

Python

Flask

Pandas, NumPy

Scikit-Learn

Optuna

LightGBM

SHAP

HTML, CSS, JavaScript

📌 Future Improvements

Add model comparison dashboard

Add SHAP summary plot

Add map-based visualization for predictions

👤 Author

Subal Kundu
MCA Student | Data Science & Web Development Enthusiast

GitHub: https://github.com/subal43

X (Twitter): https://x.com/subal64780

LinkedIn: https://www.linkedin.com/in/subal-kundu-b26905261