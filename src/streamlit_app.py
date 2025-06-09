import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🤖 ML Playground")
st.write("Upload a CSV file, select target and features, and train a model!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Dataset:", df.head())

    non_floats_cols=df.select_dtypes(exclude=['float64']).columns.tolist()
    st.write("Non-Float Columns:",non_float_cols)

    
    all_columns = df.columns.tolist()
    
    # Column selection
    target = st.selectbox("🎯 Select target column", all_columns)
    features = st.multiselect("🧠 Select feature columns", [col for col in all_columns if col != target])

    if features and target:
        if st.button("Train Model"):

            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score

            df_encoded=df.copy()
            for col in [target]+features:
                if df_encoded[col].dtype=='object':
                    df_encoded[col]=LabelEncoder().fit_transform(df_encoded[col])
            # Split and train
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")
