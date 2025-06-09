import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np # Used for checking numeric types

# Set wide layout for better display
st.set_page_config(layout="wide") 

st.title("🤖 ML Playground")
st.write("Upload a CSV file, select target and features, and train a model!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None # Initialize df outside the if block

if uploaded_file:
    # Read the uploaded CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Dataset:", df.head())

    # Identify categorical columns (object or category dtype) for encoding options
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Sidebar for data preprocessing and encoding options
    st.sidebar.header("⚙️ Data Preprocessing & Encoding")
    encoder_choices = {} # Dictionary to store user's encoder choices for each column
    if categorical_cols:
        st.sidebar.write("Select encoding method for categorical columns:")
        for col in categorical_cols:
            # Create a selectbox in the sidebar for each categorical column
            encoder_choices[col] = st.sidebar.selectbox(
                f"Encoder for '{col}'",
                ("None", "Label Encoding", "One-Hot Encoding"),
                key=f"encoder_choice_{col}" # Unique key for each selectbox
            )
    else:
        st.sidebar.info("No categorical columns detected for encoding.")
        
    # Get all column names from the original DataFrame
    all_columns = df.columns.tolist()

    # Main section for feature and target selection
    st.header("🎯 Feature and Target Selection")
    target_column = st.selectbox("🎯 Select target column", all_columns)
    feature_columns = st.multiselect("🧠 Select feature columns", [col for col in all_columns if col != target_column])

    # Only enable the train button if features and target are selected
    if feature_columns and target_column:
        if st.button("Train Model", help="Click to apply selected preprocessing and train the model."):
            st.info("Applying selected encoders and training model...")

            df_processed = df.copy() # Create a copy of the DataFrame to apply transformations

            # Keep track of original feature columns selected by the user.
            # This is crucial because One-Hot Encoding will replace original columns with new ones.
            original_selected_features = list(feature_columns)
            
            # Apply encoding based on user choices from the sidebar
            for col in categorical_cols:
                # Check if the column was chosen for encoding and still exists in the DataFrame
                if col in encoder_choices and encoder_choices[col] != "None" and col in df_processed.columns:
                    if encoder_choices[col] == "Label Encoding":
                        try:
                            # Apply Label Encoding (converts categories to numerical labels)
                            le = LabelEncoder()
                            df_processed[col] = le.fit_transform(df_processed[col])
                            st.write(f"Applied Label Encoding to '{col}'.")
                        except Exception as e:
                            st.error(f"Error applying Label Encoding to '{col}': {e}")
                            st.stop() # Stop execution on error
                    elif encoder_choices[col] == "One-Hot Encoding":
                        try:
                            # Convert column to string type to handle potential mixed types before OHE
                            df_processed[col] = df_processed[col].astype(str)

                            # Apply One-Hot Encoding (creates new binary columns for each category)
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            # Fit and transform the column, ensuring it's a 2D array for OHE
                            ohe_array = ohe.fit_transform(df_processed[[col]])
                            # Create a new DataFrame from the one-hot encoded array with appropriate column names
                            ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out([col]), index=df_processed.index)

                            # Drop the original categorical column from the processed DataFrame
                            df_processed = df_processed.drop(columns=[col])
                            
                            # Concatenate the new one-hot encoded columns to the processed DataFrame
                            df_processed = pd.concat([df_processed, ohe_df], axis=1)

                            st.write(f"Applied One-Hot Encoding to '{col}'. New columns created: {', '.join(ohe_df.columns.tolist())}")
                        except Exception as e:
                            st.error(f"Error applying One-Hot Encoding to '{col}': {e}")
                            st.stop() # Stop execution on error

            # Now, identify the actual feature columns to use for the model from the processed DataFrame.
            # This accounts for original features that might have been replaced by One-Hot Encoded columns.
            final_features_for_model = []
            for feat in original_selected_features:
                if feat in df_processed.columns:
                    # If the original feature still exists (e.g., it was numeric or label encoded)
                    final_features_for_model.append(feat)
                else:
                    # If the original feature was one-hot encoded, find the new OHE columns.
                    # This assumes OHE column names start with the original column name as a prefix.
                    ohe_cols = [c for c in df_processed.columns if c.startswith(f"{feat}_")]
                    final_features_for_model.extend(ohe_cols)
            
            # Ensure all identified final features actually exist in the processed DataFrame
            final_features_for_model = [f for f in final_features_for_model if f in df_processed.columns]

            if not final_features_for_model:
                st.error("No valid feature columns found after preprocessing. Please check your selections and encoding choices.")
                st.stop() # Stop if no features are left

            X = df_processed[final_features_for_model] # Features DataFrame
            y = df_processed[target_column] # Target Series (name remains the same)

            # Check for non-numeric data in X (features) before training
            non_numeric_X = X.select_dtypes(exclude=np.number).columns.tolist()
            if non_numeric_X:
                st.error(f"Error: Feature columns '{non_numeric_X}' contain non-numeric data even after encoding. Please ensure all features are numeric.")
                st.stop()
            
            # Check if the target column is non-numeric. For classification, it needs to be numeric.
            if not pd.api.types.is_numeric_dtype(y):
                # If target is non-numeric, try to auto-apply Label Encoding
                try:
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                    st.write(f"Auto-applied Label Encoding to target column '{target_column}' as it was non-numeric for classification.")
                except Exception as e:
                    st.error(f"Error: Target column '{target_column}' is non-numeric and could not be auto-encoded. Please ensure it's suitable for classification: {e}")
                    st.stop()

            # Ensure X and y have consistent indices, which is crucial after concatenating/dropping columns.
            X, y = X.align(y, join='inner', axis=0)

            # Check if there are still missing values after processing.
            # For this example, rows with NaNs are dropped. In a production app, you might offer imputation.
            if X.isnull().sum().sum() > 0 or pd.Series(y).isnull().sum() > 0:
                st.warning("Missing values detected in features or target after encoding. Dropping rows with NaNs.")
                # Combine X and y temporarily to drop rows where either has NaN
                combined_df = pd.concat([X, pd.Series(y, name='target')], axis=1).dropna()
                X = combined_df[X.columns]
                y = combined_df['target']
                st.info(f"Dropped rows with missing values. Remaining samples: {len(X)}")

            # Check if any valid data remains after processing and handling missing values
            if len(X) == 0:
                st.error("No valid data remaining after preprocessing and handling missing values. Cannot train model.")
                st.stop()
            
            # Split and train the model
            try:
                # Ensure enough samples for splitting into train and test sets
                if len(X) < 2:
                    st.error("Not enough samples in your dataset to split into train and test sets. Need at least 2 samples.")
                    st.stop()
                
                # Adjust test_size for very small datasets to ensure at least one sample in test set
                test_size_val = 0.2
                if len(X) * test_size_val < 1: 
                    if len(X) > 1:
                        test_size_val = 1 / len(X) 
                    else:
                        test_size_val = 0.0 # Will be caught by len(X) < 2 check above

                # Perform train-test split. Stratify if the target has multiple unique classes.
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_val, random_state=42, 
                    stratify=y if pd.Series(y).nunique() > 1 and len(y.value_counts()) > 1 else None
                )
                
                # Check if the training target set has only one class, which causes issues for classifiers
                if len(np.unique(y_train)) < 2:
                    st.error("Target variable in training set has only one class after split. Cannot train a classifier. Please check your data or target selection.")
                    st.stop()

                # Initialize and train the RandomForestClassifier model
                model = RandomForestClassifier(random_state=42) # random_state for reproducibility
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate and display accuracy score
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")

                # Display model evaluation details
                st.subheader("Model Evaluation")
                st.write(f"**Selected Features (After Encoding):** {', '.join(final_features_for_model)}")
                st.write(f"**Target Column:** {target_column}")
                st.write(f"**Test Set Accuracy:** {acc:.2f}")
                st.write("---")
                st.write("First 10 Actual vs. Predicted values on Test Set:")
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
                st.dataframe(results_df)

            except Exception as e:
                st.error(f"An error occurred during model training or splitting: {e}")
                st.write("Please ensure your selected features and target are numeric after encoding and suitable for the chosen model. Also check for sufficient data for splitting.")
    elif uploaded_file:
        st.warning("Please select both target and feature columns to enable model training.")
elif uploaded_file is None:
    st.info("Upload a CSV file to begin.")
