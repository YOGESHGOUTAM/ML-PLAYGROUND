import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout for better display
st.set_page_config(layout="wide")

st.title("🤖 ML Playground")
st.write("Upload a CSV file, select target and features, train a model, and see results!")

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
    manual_numeric_mappings = {} # Dictionary to store manual numeric mappings for 'Custom Numeric Mapping'

    if categorical_cols:
        st.sidebar.write("Select encoding method for categorical columns:")
        for col in categorical_cols:
            # Create a selectbox in the sidebar for each categorical column
            encoder_choices[col] = st.sidebar.selectbox(
                f"Encoder for '{col}'",
                ("None", "Label Encoding", "One-Hot Encoding", "Custom Numeric Mapping"),
                key=f"encoder_choice_{col}" # Unique key for each selectbox
            )

            if encoder_choices[col] == "Custom Numeric Mapping":
                st.sidebar.markdown(f"**Define numeric mapping for '{col}':**")
                unique_categories = df[col].astype(str).unique().tolist()
                manual_numeric_mappings[col] = {}
                for category in unique_categories:
                    value = st.sidebar.number_input(
                        f"Value for '{category}'",
                        key=f"manual_value_{col}_{category}",
                        value=0.0,
                        help=f"Enter the numeric value for the category '{category}'"
                    )
                    manual_numeric_mappings[col][category] = value
                st.sidebar.markdown("---")
    else:
        st.sidebar.info("No categorical columns detected for encoding.")

    # Get all column names from the original DataFrame
    all_columns = df.columns.tolist()

    # Main section for feature and target selection
    st.header("🎯 Feature and Target Selection")
    target_column = st.selectbox("🎯 Select target column", all_columns)
    feature_columns = st.multiselect("🧠 Select feature columns", [col for col in all_columns if col != target_column])

    # Model Selection
    st.header("✨ Model Selection")
    model_choice = st.selectbox(
        "Choose your Machine Learning Model",
        ("Random Forest Classifier", "Neural Network (MLP Classifier)")
    )

    # Model-specific parameters
    model_params = {}
    if model_choice == "Neural Network (MLP Classifier)":
        st.subheader("Neural Network (MLP) Parameters")
        hidden_layers_str = st.text_input(
            "Hidden Layer Sizes (e.g., '100,50' for 2 layers with 100 and 50 neurons)",
            value="100",
            help="Enter comma-separated integers for the number of neurons in each hidden layer."
        )
        try:
            model_params['hidden_layer_sizes'] = tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())
            if not model_params['hidden_layer_sizes']:
                st.warning("No valid hidden layer sizes entered. Using default (100,).")
                model_params['hidden_layer_sizes'] = (100,)
        except ValueError:
            st.error("Invalid format for hidden layer sizes. Please enter comma-separated integers.")
            model_params['hidden_layer_sizes'] = (100,)

        model_params['activation'] = st.selectbox(
            "Activation Function",
            ("relu", "tanh", "logistic", "identity"),
            help="The activation function for the hidden layer."
        )
        model_params['solver'] = st.selectbox(
            "Solver for Weight Optimization",
            ("adam", "sgd", "lbfgs"),
            help="The solver for weight optimization."
        )
        model_params['max_iter'] = st.number_input(
            "Maximum Iterations (Epochs)",
            min_value=50,
            max_value=2000,
            value=200,
            step=50,
            help="Maximum number of iterations for the solver to converge."
        )

    # --- Training Options ---
    st.header("⚙️ Training Options")
    test_size_percent = st.slider(
        "Test Set Size Percentage",
        min_value=10,
        max_value=90,
        value=20,
        step=5,
        help="Percentage of data to be used for the test set."
    )
    test_size_val = test_size_percent / 100.0

    # Initialize threshold_classification to None, will only be set if applicable
    threshold_classification = None

    if feature_columns and target_column:
        # Check if the target column is already numeric
        is_target_numeric_before_processing = pd.api.types.is_numeric_dtype(df[target_column])
        
        # Temporarily get unique values for the target to determine if it's binary after potential auto-encoding
        temp_y = df[target_column].copy()
        if not is_target_numeric_before_processing:
            try:
                le_temp = LabelEncoder()
                temp_y = le_temp.fit_transform(temp_y.astype(str))
            except:
                pass # If auto-encoding fails, it will be caught later

        # Only show threshold option for binary classification after potential auto-encoding
        if len(np.unique(temp_y)) == 2:
            st.write("---") # Separator for visual clarity
            st.subheader("Binary Classification Threshold")
            st.info("Adjust this threshold to classify probabilities into classes.")
            threshold_classification = st.slider(
                "Prediction Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5, # Default threshold
                step=0.01,
                help="Probability threshold for positive class classification (0.0 to 1.0)."
            )

    if feature_columns and target_column:
        if st.button("Train Model", help="Click to apply selected preprocessing and train the model."):
            st.info("Applying selected encoders and training model...")

            df_processed = df.copy()
            original_selected_features = list(feature_columns)

            for col in categorical_cols:
                if col in encoder_choices and encoder_choices[col] != "None" and col in df_processed.columns:
                    df_processed[col] = df_processed[col].astype(str)

                    if encoder_choices[col] == "Label Encoding":
                        try:
                            le = LabelEncoder()
                            df_processed[col] = le.fit_transform(df_processed[col])
                            st.write(f"Applied Label Encoding to '{col}'.")
                        except Exception as e:
                            st.error(f"Error applying Label Encoding to '{col}': {e}")
                            st.stop()
                    elif encoder_choices[col] == "One-Hot Encoding":
                        try:
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            ohe_array = ohe.fit_transform(df_processed[[col]])
                            ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out([col]), index=df_processed.index)
                            df_processed = df_processed.drop(columns=[col])
                            df_processed = pd.concat([df_processed, ohe_df], axis=1)
                            st.write(f"Applied One-Hot Encoding to '{col}'. New columns created: {', '.join(ohe_df.columns.tolist())}")
                        except Exception as e:
                            st.error(f"Error applying One-Hot Encoding to '{col}': {e}")
                            st.stop()
                    elif encoder_choices[col] == "Custom Numeric Mapping":
                        try:
                            mapping = manual_numeric_mappings.get(col, {})
                            actual_categories = df_processed[col].unique().tolist()
                            missing_mappings = [cat for cat in actual_categories if cat not in mapping]
                            if missing_mappings:
                                st.error(f"Error: For '{col}', the following categories do not have a custom numeric mapping defined: {', '.join(missing_mappings)}. Please assign a value for all categories.")
                                st.stop()
                            df_processed[col] = df_processed[col].map(mapping)
                            st.write(f"Applied Custom Numeric Mapping to '{col}': {mapping}")
                        except Exception as e:
                            st.error(f"Error applying Custom Numeric Mapping to '{col}': {e}. Please check your manual value inputs.")
                            st.stop()

            final_features_for_model = []
            for feat in original_selected_features:
                if feat in df_processed.columns:
                    final_features_for_model.append(feat)
                else:
                    ohe_cols = [c for c in df_processed.columns if c.startswith(f"{feat}_")]
                    final_features_for_model.extend(ohe_cols)

            final_features_for_model = [f for f in final_features_for_model if f in df_processed.columns]

            if not final_features_for_model:
                st.error("No valid feature columns found after preprocessing. Please check your selections and encoding choices.")
                st.stop()

            X = df_processed[final_features_for_model]
            y = df_processed[target_column]

            st.info("Converting all feature columns to numeric. Non-convertible values will become NaN.")
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            if not pd.api.types.is_numeric_dtype(y):
                try:
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y.astype(str))
                    target_classes = le_target.classes_
                    st.write(f"Auto-applied Label Encoding to target column '{target_column}' as it was non-numeric for classification.")
                except Exception as e:
                    st.error(f"Error: Target column '{target_column}' is non-numeric and could not be auto-encoded. Please ensure it's suitable for classification: {e}")
                    st.stop()
            else:
                target_classes = np.unique(y)

            X, y = X.align(y, join='inner', axis=0)

            if X.isnull().sum().sum() > 0 or pd.Series(y).isnull().sum() > 0:
                initial_rows = len(X)
                st.warning("Missing values detected in features or target after encoding/conversion. Dropping rows with NaNs.")
                combined_df = pd.concat([X, pd.Series(y, name='target')], axis=1).dropna()
                X = combined_df[X.columns]
                y = combined_df['target']
                st.info(f"Dropped {initial_rows - len(X)} rows with missing values. Remaining samples: {len(X)}")

            if len(X) == 0:
                st.error("No valid data remaining after preprocessing and handling missing values. Cannot train model.")
                st.stop()

            try:
                if len(X) < 2:
                    st.error("Not enough samples in your dataset to split into train and test sets. Need at least 2 samples.")
                    st.stop()
                
                # Check for minimum 2 samples per class for stratification
                # Only attempt to stratify if all classes have at least 2 samples
                stratify_needed = False
                if pd.Series(y).nunique() > 1: # Only stratify if multiple classes exist
                    class_counts = pd.Series(y).value_counts()
                    if all(count >= 2 for count in class_counts):
                        stratify_needed = True
                    else:
                        st.warning("Cannot perform stratified split because some classes have less than 2 members. Proceeding with non-stratified split.")

                if stratify_needed:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_val, random_state=42,
                        stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_val, random_state=42
                    )

                if len(np.unique(y_train)) < 2:
                    st.error("Target variable in training set has only one class after split. Cannot train a classifier. Please check your data or target selection.")
                    st.stop()

                model = None
                if model_choice == "Random Forest Classifier":
                    model = RandomForestClassifier(random_state=42)
                    st.subheader(f"Training Random Forest Classifier")
                elif model_choice == "Neural Network (MLP Classifier)":
                    if 'hidden_layer_sizes' in model_params and not isinstance(model_params['hidden_layer_sizes'], tuple):
                         model_params['hidden_layer_sizes'] = (model_params['hidden_layer_sizes'],)
                    model = MLPClassifier(random_state=42, **model_params)
                    st.subheader(f"Training Neural Network (MLP Classifier)")

                model.fit(X_train, y_train)

                # --- Apply threshold for binary classification if applicable ---
                y_pred_final = None
                if len(target_classes) == 2 and threshold_classification is not None:
                    # Get probabilities for the positive class (assuming class 1 is positive)
                    # Need to know which column corresponds to which class
                    # For binary classification, predict_proba returns columns in sorted order of class labels.
                    # We assume the LabelEncoder maps to 0 and 1, so 1 is the second column.
                    y_proba = model.predict_proba(X_test)
                    # Check if the model outputs probabilities for more than one class.
                    # It's usually safe to assume the second column is the positive class probability if
                    # the target was encoded to 0 and 1.
                    if y_proba.shape[1] == 2:
                        # y_proba[:, 1] is the probability of the positive class (class 1)
                        y_pred_final = (y_proba[:, 1] >= threshold_classification).astype(int)
                        st.write(f"Predictions made using a threshold of **{threshold_classification}**.")
                    else:
                        st.warning("Model's predict_proba output is not suitable for binary thresholding. Using direct predictions.")
                        y_pred_final = model.predict(X_test) # Fallback to direct predict
                else:
                    y_pred_final = model.predict(X_test) # Use direct predict for non-binary or no threshold set


                # --- Display Model Evaluation ---
                st.success(f"✅ Model Trained! ({model_choice})")
                st.subheader("📊 Model Evaluation")

                # Use y_pred_final for all metrics
                acc = accuracy_score(y_test, y_pred_final)
                st.metric(label="Accuracy", value=f"{acc:.2f}")

                st.write("### Classification Report")
                report = classification_report(y_test, y_pred_final, target_names=[str(c) for c in target_classes], output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).T)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_final)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=[str(c) for c in target_classes],
                    yticklabels=[str(c) for c in target_classes],
                    ax=ax
                )
                ax.set_ylabel('Actual Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                plt.close(fig)

                if model_choice == "Random Forest Classifier":
                    st.write("### Feature Importances (Random Forest)")
                    importances = model.feature_importances_
                    features_df = pd.DataFrame({
                        'Feature': final_features_for_model,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)

                    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=features_df, ax=ax_importance)
                    ax_importance.set_title('Feature Importances')
                    ax_importance.set_xlabel('Importance')
                    ax_importance.set_ylabel('Feature')
                    st.pyplot(fig_importance)
                    plt.close(fig_importance)

                st.write("---")
                st.write("First 10 Actual vs. Predicted values on Test Set:")
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_final}).head(10)
                st.dataframe(results_df)

            except Exception as e:
                st.error(f"An error occurred during model training or splitting: {e}")
                st.write("Please ensure your selected features and target are numeric after encoding and suitable for the chosen model. Also check for sufficient data for splitting. For MLP, consider adjusting parameters like `max_iter` or `hidden_layer_sizes` if convergence warnings occur.")
    elif uploaded_file:
        st.warning("Please select both target and feature columns to enable model training.")
elif uploaded_file is None:
    st.info("Upload a CSV file to begin.")
