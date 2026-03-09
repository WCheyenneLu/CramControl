
# CramControl 

CramControl is an AI-powered pipeline that parses course syllabi and generates a personalized, week-by-week workload forecast. It uses the Gemini API to extract unstructured text from PDFs and an XGBoost machine learning model to predict estimated completion hours for every assignment.

## Prerequisites & Installation

Before running the application, ensure you have Python installed and environment is activated.

1. **Clone the repository and navigate to the folder:**
```bash
cd CramControl

```

2. **Install the required packages:**
```bash
pip install -r requirements.txt

```


---

## Generating Data & Training the Model

Because this project relies on a machine learning model to estimate workload hours, you need to generate synthetic student data and train the model locally before the Streamlit app can run.

Run these three scripts in order from your terminal:

1. **Generate synthetic students and tasks:**
```bash
python src/synth_students.py

```


2. **Compile the training dataset:**
```bash
python src/make_training_data.py

```


3. **Train the XGBoost model:**
*(This will create the `data/model_hours_xgb.joblib` file required by the app).*
```bash
python src/train_model.py

```



---

## 🚀 Running the App

Once the model is trained and your `.env` file is ready, you can start the Streamlit dashboard:

```bash
streamlit run app.py

```

---

## 🛑 Troubleshooting & Common Issues

### 1. `ModuleNotFoundError`

If you try to run any of the scripts or the app and see an error like `ModuleNotFoundError: No module named 'xgboost'` or `'sklearn'`:

* **Cause:** You are missing a required Python package in your current environment.
* **Fix:** Look at the exact name of the missing module in the error message and install it via your terminal (e.g., `pip install xgboost` or `pip install scikit-learn`).

### 2. Finals/Late-term assignments showing up in Week 1

If the app successfully parses the PDF, but late-quarter events (like final exams or capstone projects) are incorrectly stacking up in Week 1 of your workload graph:

* **Cause:** The "Quarter start date" in the app's settings doesn't match the actual academic calendar, causing the week-number math to break.
* **Fix:** Open the sidebar in the Streamlit app and change the **Quarter start date** to the correct first Monday of the term (e.g., set it to **Jan 5th** for the Winter quarter).

---

## Acknowledgements

Thank you to my project team members Kiara, Emilio, Noah, and Sanskriti for their contributions. This project represents the final version of an internal project developed through the Data Science Union at UCLA.
