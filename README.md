---
license: mit
title: ML PLAYGROUND
sdk: streamlit
emoji: 📊
short_description: Train ML Modes with your dataset!
sdk_version: 1.45.1
colorFrom: red
---

# 📊 ML Playground

A **Streamlit-based Machine Learning Playground** to:

- Upload your dataset (CSV)
- Preprocess data (label, one-hot, custom mapping)
- Select features and target
- Train machine learning models (classification or regression)
- Visualize data relationships
- Make predictions on new data (manual or CSV input)

---

## ✨ Features

### 📂 CSV Upload
- Upload any CSV dataset quickly through the UI.

### 👁️ Data Preview
- Instantly preview the top rows of your uploaded data.

### 🧹 Categorical Encoding Options
- **Label Encoding**: Convert categories to integers.
- **One-Hot Encoding**: Create binary columns per category.
- **Custom Numeric Mapping**: Manually assign values to categories (e.g., `{'Yes': 1, 'No': 0}`).

### 🧠 Feature and Target Selection
- Select the **target variable** and multiple **input features**.

### 🤖 Smart Model Suggestion
- **Classification** (for categorical targets):
  - Random Forest Classifier
  - MLP Classifier (Neural Network)
- **Regression** (for numerical targets):
  - Linear Regression
  - Random Forest Regressor

### 🧰 MLP Neural Network Customization
- Customize:
  - Hidden layers (e.g., `(100,)`)
  - Activation (`relu`, `tanh`, `logistic`)
  - Solver (`adam`, `sgd`, etc.)
  - Max iterations

### ⚙️ Training Options
- Set train/test split ratio (e.g., 80/20).
- For binary classification, adjust threshold for prediction probabilities.

### 📊 Model Evaluation Metrics
- **Classification:**
  - Accuracy Score
  - Confusion Matrix Heatmap
  - Classification Report (Precision, Recall, F1)
- **Regression:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Feature Importances** (from Random Forest)

### 📈 Feature-Target Visualization
- Select any feature and visualize its relationship with the target.
- Auto-selected plot types:
  - Scatter
  - Box
  - Violin
  - Count Heatmap

### 🔮 Prediction on New Data
- **Manual Input**: Input feature values via form to get a single prediction.
- **Batch CSV Upload**: Upload new CSV for predictions on multiple rows (auto-preprocessing applied).

---

## 🚀 How to Run the App

### 1. Save the App
Save the app code in a file named `app.py`.

### 2. Install Required Libraries

Create a file named `requirements.txt` with the following content
  streamlit
  pandas
  scikit-learn
  matplotlib
  seaborn
  numpy
  joblib

Then run:


pip install -r requirements.txt
3. Launch the Streamlit App
In your terminal, navigate to the folder where app.py is located and run:

streamlit run app.py
The app will open automatically in your default web browser.

## 👨‍💻 How to Use

### 1. Upload CSV  
Click **"Upload your CSV file"** to upload your dataset.

### 2. Preview Data  
View the first few rows to verify your data was uploaded correctly.

### 3. Encode Categorical Columns  
Choose an encoding method for each categorical column:
- **None**  
- **Label Encoding**  
- **One-Hot Encoding**  
- **Custom Mapping** (e.g., `{Yes: 1, No: 0}`)

### 4. Select Features and Target  
Choose:
- The **target column** (what you want to predict)
- The **feature columns** (variables used for prediction)

### 5. Configure Model  
Select a suggested model:
- For classification or regression based on target type  
- If using **MLP Classifier**, customize hidden layers, activation, solver, and iterations

### 6. Train Model  
Click **"Train Model"** to train and evaluate the model.

### 7. Visualize  
Choose a feature to visualize its relationship with the target:
- Scatter plots, box plots, violin plots, or count heatmaps are auto-selected

### 8. Make Predictions  
You can:
- **Manually enter feature values** in a form  
- **Upload a new CSV file** with the same structure to predict multiple rows

---

## 📄 License

**MIT License**


🧑‍🏫 Credits
Developed by Yogesh Goutam(github/YOGESHGOUTAM)
Feel free to fork, enhance, or contribute!