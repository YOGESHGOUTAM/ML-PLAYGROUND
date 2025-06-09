import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving/loading scikit-learn objects (encoders, model)

# Set wide layout for better display
st.set_page_config(layout="wide")

st.title("🤖 ML Playground")
st.write("Upload a CSV file, select target and features, train a model, and see results!")

# Initialize session state variables if not already present
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le_target' not in st.session_state:
    st.session_state.le_target = None
if 'encoder_choices' not in st.session_state:
    st.session_state.encoder_choices = {}
if 'manual_numeric_mappings' not in st.session_state:
    st.session_state.manual_numeric_mappings = {}
if 'ohe_encoders' not in st.session_state: # To store fitted OneHotEncoder objects
    st.session_state.ohe_encoders = {}
if 'final_features_for_model' not in st.session_state:
    st.session_state.final_features_for_model = []
if 'target_classes' not in st.session_state:
    st.session_state.target_classes = []
if 'is_target_continuous' not in st.session_state:
    st.session_state.is_target_continuous = False
if 'original_feature_dtypes' not in st.session_state: # To store original dtypes for manual input hints
    st.session_state.original_feature_dtypes = {}
if 'original_feature_cols_map' not in st.session_state: # To map OHE features back to original for input
    st.session_state.original_feature_cols_map = {}


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
    encoder_choices_current_run = {} # Use a temp variable for current run's choices
    manual_numeric_mappings_current_run = {}

    if categorical_cols:
        st.sidebar.write("Select encoding method for categorical columns:")
        for col in categorical_cols:
            encoder_choices_current_run[col] = st.sidebar.selectbox(
                f"Encoder for '{col}'",
                ("None", "Label Encoding", "One-Hot Encoding", "Custom Numeric Mapping"),
                key=f"encoder_choice_{col}"
            )

            if encoder_choices_current_run[col] == "Custom Numeric Mapping":
                st.sidebar.markdown(f"**Define numeric mapping for '{col}':**")
                unique_categories = df[col].astype(str).unique().tolist()
                manual_numeric_mappings_current_run[col] = {}
                for category in unique_categories:
                    value = st.sidebar.number_input(
                        f"Value for '{category}'",
                        key=f"manual_value_{col}_{category}",
                        value=0.0,
                        help=f"Enter the numeric value for the category '{category}'"
                    )
                    manual_numeric_mappings_current_run[col][category] = value
                st.sidebar.markdown("---")
    else:
        st.sidebar.info("No categorical columns detected for encoding.")

    all_columns = df.columns.tolist()

    st.header("🎯 Feature and Target Selection")
    target_column = st.selectbox("🎯 Select target column", all_columns)
    feature_columns = st.multiselect("🧠 Select feature columns", [col for col in all_columns if col != target_column])

    # Determine if the target column is likely continuous or discrete
    is_target_continuous_current_run = False
    target_unique_values_count = 0
    if target_column:
        temp_target_series = df[target_column].copy()
        temp_target_series = pd.to_numeric(temp_target_series, errors='coerce').dropna()

        if len(temp_target_series) == 0:
            st.warning(f"Target column '{target_column}' contains no valid numeric data after cleaning. Cannot determine if it's continuous or discrete.")
        else:
            if pd.api.types.is_numeric_dtype(temp_target_series):
                target_unique_values_count = temp_target_series.nunique()
                if target_unique_values_count / len(temp_target_series) > 0.05 or target_unique_values_count > 20:
                    is_target_continuous_current_run = True


    # --- Feature-Target Relationship Plots ---
    st.header("📈 Feature-Target Relationship Plots")
    if target_column and all_columns:
        plot_feature_options = [col for col in all_columns if col != target_column]
        if plot_feature_options:
            selected_plot_feature = st.selectbox(
                "Select a feature to plot against the target column",
                plot_feature_options,
                key="plot_feature_selector"
            )

            if selected_plot_feature:
                st.write(f"Displaying relationship between **'{selected_plot_feature}'** and **'{target_column}'**")

                plot_df = df[[selected_plot_feature, target_column]].copy()

                plot_df[selected_plot_feature + '_numeric'] = pd.to_numeric(plot_df[selected_plot_feature], errors='coerce')
                plot_df[target_column + '_numeric'] = pd.to_numeric(plot_df[target_column], errors='coerce')

                is_selected_feature_continuous = False
                if not plot_df[selected_plot_feature + '_numeric'].dropna().empty:
                    if pd.api.types.is_numeric_dtype(plot_df[selected_plot_feature + '_numeric'].dropna()):
                        if plot_df[selected_plot_feature + '_numeric'].nunique() / len(plot_df[selected_plot_feature + '_numeric'].dropna()) > 0.05 or plot_df[selected_plot_feature + '_numeric'].nunique() > 10:
                            is_selected_feature_continuous = True

                plot_df.dropna(subset=[selected_plot_feature + '_numeric', target_column + '_numeric'], inplace=True)

                if plot_df.empty:
                    st.warning("No valid data points available for plotting after handling missing/non-numeric values for the selected feature and target.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if is_target_continuous_current_run and is_selected_feature_continuous:
                        sns.scatterplot(x=selected_plot_feature + '_numeric', y=target_column + '_numeric', data=plot_df, ax=ax)
                        ax.set_title(f'Scatter Plot of {selected_plot_feature} vs. {target_column}')
                        ax.set_xlabel(selected_plot_feature)
                        ax.set_ylabel(target_column)

                    elif is_target_continuous_current_run and not is_selected_feature_continuous:
                        sns.boxplot(x=selected_plot_feature, y=target_column + '_numeric', data=plot_df, ax=ax)
                        ax.set_title(f'Box Plot of {target_column} by {selected_plot_feature}')
                        ax.set_xlabel(selected_plot_feature)
                        ax.set_ylabel(target_column)
                        plt.xticks(rotation=45, ha='right')

                    elif not is_target_continuous_current_run and is_selected_feature_continuous:
                        sns.violinplot(x=target_column, y=selected_plot_feature + '_numeric', data=plot_df, ax=ax)
                        ax.set_title(f'Violin Plot of {selected_plot_feature} by {target_column}')
                        ax.set_xlabel(target_column)
                        ax.set_ylabel(selected_plot_feature)
                        plt.xticks(rotation=45, ha='right')

                    else:
                        plot_df[selected_plot_feature] = plot_df[selected_plot_feature].astype(str)
                        plot_df[target_column] = plot_df[target_column].astype(str)

                        crosstab_df = pd.crosstab(plot_df[selected_plot_feature], plot_df[target_column])
                        if not crosstab_df.empty:
                            sns.heatmap(crosstab_df, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_title(f'Count Heatmap of {selected_plot_feature} vs. {target_column}')
                            ax.set_xlabel(target_column)
                            ax.set_ylabel(selected_plot_feature)
                            plt.xticks(rotation=45, ha='right')
                            plt.yticks(rotation=0)
                        else:
                            st.warning("Cannot plot heatmap: Crosstab result is empty for the selected categorical features.")
                            plt.close(fig)
                            st.stop()

                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Select a feature to see its relationship with the target.")
        else:
            st.info("No features available for plotting against the target column.")
    else:
        st.info("Please select a target column to enable relationship plots.")

    st.header("✨ Model Selection")

    model_options = []
    if is_target_continuous_current_run:
        model_options = ["Linear Regression", "Random Forest Regressor"]
        st.info("Target column appears to be **continuous**. Only regression models are available.")
    else:
        model_options = ["Random Forest Classifier", "Neural Network (MLP Classifier)"]
        st.info("Target column appears to be **categorical**. Only classification models are available.")

    model_choice = st.selectbox(
        "Choose your Machine Learning Model",
        model_options
    )

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

    threshold_classification = None

    if target_column and not is_target_continuous_current_run and model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"]:
        temp_y_for_threshold_check = df[target_column].copy()
        if not pd.api.types.is_numeric_dtype(temp_y_for_threshold_check):
            try:
                le_temp = LabelEncoder()
                temp_y_for_threshold_check = le_temp.fit_transform(temp_y_for_threshold_check.astype(str))
            except:
                pass
        if len(np.unique(temp_y_for_threshold_check)) == 2:
            st.write("---")
            st.subheader("Binary Classification Threshold")
            st.info("Adjust this threshold to classify probabilities into classes.")
            threshold_classification = st.slider(
                "Prediction Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Probability threshold for positive class classification (0.0 to 1.0)."
            )

    if feature_columns and target_column:
        if st.button("Train Model", help="Click to apply selected preprocessing and train the model."):
            st.info("Applying selected encoders and training model...")

            df_processed = df.copy()
            st.session_state.original_feature_dtypes = {col: df[col].dtype for col in feature_columns} # Store original dtypes
            st.session_state.original_feature_cols_map = {feat: feat for feat in feature_columns} # Initialize map

            # Clear previous encoder state before training
            st.session_state.ohe_encoders = {}
            st.session_state.manual_numeric_mappings = {}
            st.session_state.encoder_choices = {} # Clear this too


            for col in categorical_cols:
                # Store the encoder choice for this column for prediction application
                st.session_state.encoder_choices[col] = encoder_choices_current_run.get(col, "None") 

                if encoder_choices_current_run[col] != "None" and col in df_processed.columns:
                    df_processed[col] = df_processed[col].astype(str)

                    if encoder_choices_current_run[col] == "Label Encoding":
                        try:
                            le = LabelEncoder()
                            df_processed[col] = le.fit_transform(df_processed[col])
                            st.write(f"Applied Label Encoding to '{col}'.")
                            # No need to store LabelEncoder for prediction for now as they are only used during training.
                        except Exception as e:
                            st.error(f"Error applying Label Encoding to '{col}': {e}")
                            st.stop()
                    elif encoder_choices_current_run[col] == "One-Hot Encoding":
                        try:
                            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            ohe_array = ohe.fit_transform(df_processed[[col]])
                            ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out([col]), index=df_processed.index)
                            
                            df_processed = df_processed.drop(columns=[col]) 
                            df_processed = pd.concat([df_processed, ohe_df], axis=1)
                            st.write(f"Applied One-Hot Encoding to '{col}'. New columns created: {', '.join(ohe_df.columns.tolist())}")
                            
                            st.session_state.ohe_encoders[col] = ohe
                            st.session_state.original_feature_cols_map[col] = ohe.get_feature_names_out([col]).tolist()
                        except Exception as e:
                            st.error(f"Error applying One-Hot Encoding to '{col}': {e}")
                            st.stop()
                    elif encoder_choices_current_run[col] == "Custom Numeric Mapping":
                        try:
                            mapping = manual_numeric_mappings_current_run.get(col, {})
                            actual_categories = df_processed[col].unique().tolist()
                            missing_mappings = [cat for cat in actual_categories if cat not in mapping]
                            if missing_mappings:
                                st.error(f"Error: For '{col}', the following categories do not have a custom numeric mapping defined: {', '.join(missing_mappings)}. Please assign a value for all categories.")
                                st.stop()
                            df_processed[col] = df_processed[col].map(mapping)
                            st.write(f"Applied Custom Numeric Mapping to '{col}': {mapping}")
                            
                            st.session_state.manual_numeric_mappings[col] = mapping
                        except Exception as e:
                            st.error(f"Error applying Custom Numeric Mapping to '{col}': {e}. Please check your manual value inputs.")
                            st.stop()
            
            # For any categorical columns NOT selected for encoding, their choice defaults to "None"
            # And they will be handled by the numeric conversion and NaN dropping later
            for col in [c for c in categorical_cols if c not in st.session_state.encoder_choices]:
                 st.session_state.encoder_choices[col] = "None"


            final_features_for_model = []
            for feat in feature_columns:
                if feat in df_processed.columns:
                    final_features_for_model.append(feat)
                else:
                    ohe_cols_for_feat = st.session_state.original_feature_cols_map.get(feat, [])
                    final_features_for_model.extend(ohe_cols_for_feat)

            final_features_for_model = [f for f in final_features_for_model if f in df_processed.columns]
            st.session_state.final_features_for_model = final_features_for_model

            if not final_features_for_model:
                st.error("No valid feature columns found after preprocessing. Please check your selections and encoding choices.")
                st.stop()

            X = df_processed[final_features_for_model]
            y = df_processed[target_column]

            st.info("Converting all feature columns to numeric. Non-convertible values will become NaN.")
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            target_classes_current_run = []
            if model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"]:
                if not pd.api.types.is_numeric_dtype(y):
                    try:
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                        target_classes_current_run = le_target.classes_
                        st.session_state.le_target = le_target
                        st.write(f"Auto-applied Label Encoding to target column '{target_column}' as it was non-numeric for classification.")
                    except Exception as e:
                        st.error(f"Error: Target column '{target_column}' is non-numeric and could not be auto-encoded. Please ensure it's suitable for classification: {e}")
                        st.stop()
                else:
                    target_classes_current_run = np.unique(y)
            elif model_choice in ["Linear Regression", "Random Forest Regressor"]:
                y = pd.to_numeric(y, errors='coerce')
                if y.isnull().any():
                    st.warning(f"Missing values introduced in target column '{target_column}' during numeric conversion for regression. These rows will be dropped.")
            
            st.session_state.target_classes = target_classes_current_run
            st.session_state.is_target_continuous = is_target_continuous_current_run


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

                stratify_y = None
                if model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"] and pd.Series(y).nunique() > 1:
                    class_counts = pd.Series(y).value_counts()
                    if all(count >= 2 for count in class_counts):
                        stratify_y = y
                    else:
                        st.warning("Cannot perform stratified split because some classes have less than 2 members. Proceeding with non-stratified split.")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_val, random_state=42,
                    stratify=stratify_y
                )

                if model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"] and len(np.unique(y_train)) < 2:
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
                elif model_choice == "Linear Regression":
                    model = LinearRegression()
                    st.subheader(f"Training Linear Regression Model")
                elif model_choice == "Random Forest Regressor":
                    model = RandomForestRegressor(random_state=42)
                    st.subheader(f"Training Random Forest Regressor")

                model.fit(X_train, y_train)
                st.session_state.model = model # Store the trained model

                y_pred = model.predict(X_test)

                # --- Display Model Evaluation ---
                st.success(f"✅ Model Trained! ({model_choice})")
                st.subheader("📊 Model Evaluation")

                if model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"]:
                    y_pred_final_classification = y_pred
                    if len(st.session_state.target_classes) == 2 and threshold_classification is not None:
                        if hasattr(model, 'predict_proba') and model.predict_proba(X_test).shape[1] == 2:
                            y_proba = model.predict_proba(X_test)
                            y_pred_final_classification = (y_proba[:, 1] >= threshold_classification).astype(int)
                            st.write(f"Predictions made using a threshold of **{threshold_classification}**.")
                        else:
                            st.warning("Model does not support `predict_proba` or output is not binary. Using direct predictions for classification metrics.")

                    acc = accuracy_score(y_test, y_pred_final_classification)
                    st.metric(label="Accuracy", value=f"{acc:.2f}")

                    st.write("### Classification Report")
                    report = classification_report(y_test, y_pred_final_classification, target_names=[str(c) for c in st.session_state.target_classes], output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T)

                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred_final_classification)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=[str(c) for c in st.session_state.target_classes],
                        yticklabels=[str(c) for c in st.session_state.target_classes],
                        ax=ax
                    )
                    ax.set_ylabel('Actual Label')
                    ax.set_xlabel('Predicted Label')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    plt.close(fig)

                elif model_choice in ["Linear Regression", "Random Forest Regressor"]:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
                    st.metric(label="R-squared (R²)", value=f"{r2:.2f}")

                    st.write("### Actual vs. Predicted Values Plot")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, alpha=0.7)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs. Predicted Values')
                    st.pyplot(fig)
                    plt.close(fig)

                if model_choice in ["Random Forest Classifier", "Random Forest Regressor"]:
                    st.write("### Feature Importances")
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
                if model_choice in ["Random Forest Classifier", "Neural Network (MLP Classifier)"]:
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_final_classification}).head(10)
                else:
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
                st.dataframe(results_df)

            except Exception as e:
                st.error(f"An error occurred during model training or splitting: {e}")
                st.write("Please ensure your selected features and target are numeric after encoding and suitable for the chosen model. Also check for sufficient data for splitting. For MLP, consider adjusting parameters like `max_iter` or `hidden_layer_sizes` if convergence warnings occur.")
    elif uploaded_file:
        st.warning("Please select both target and feature columns to enable model training.")
elif uploaded_file is None:
    st.info("Upload a CSV file to begin.")

# --- Make Predictions Section ---
if st.session_state.model is not None and st.session_state.final_features_for_model:
    st.header("🔮 Make Predictions")
    st.info("Use the trained model to make predictions on new data.")

    prediction_input_method = st.radio(
        "Choose prediction input method:",
        ("Manual Input", "Upload CSV for Prediction")
    )

    if prediction_input_method == "Manual Input":
        st.subheader("Manual Feature Input")
        manual_input_data = {}
        for feature_name in st.session_state.final_features_for_model:
            # Try to get the original feature name if it was one-hot encoded
            original_feat = None
            for orig_col, transformed_cols in st.session_state.original_feature_cols_map.items():
                if isinstance(transformed_cols, list) and feature_name in transformed_cols:
                    original_feat = orig_col
                    break
                elif isinstance(transformed_cols, str) and feature_name == transformed_cols:
                    original_feat = orig_col
                    break
            
            # If it's a one-hot encoded feature, ask for the original categorical value
            if original_feat and original_feat in st.session_state.ohe_encoders:
                # Get unique categories from the stored OHE
                ohe = st.session_state.ohe_encoders[original_feat]
                categories = ohe.categories_[0].tolist()
                manual_input_data[original_feat] = st.selectbox(
                    f"Select value for '{original_feat}' (Original Categorical)",
                    categories,
                    key=f"manual_input_{original_feat}"
                )
            elif original_feat and original_feat in st.session_state.manual_numeric_mappings:
                # If custom numeric mapping was applied, let user choose from original categories
                categories = list(st.session_state.manual_numeric_mappings[original_feat].keys())
                manual_input_data[original_feat] = st.selectbox(
                    f"Select value for '{original_feat}' (Original Categorical)",
                    categories,
                    key=f"manual_input_{original_feat}"
                )
            else:
                # For numeric or label-encoded features, try to guess input type
                original_dtype = st.session_state.original_feature_dtypes.get(feature_name, 'float64')
                if 'int' in str(original_dtype):
                    manual_input_data[feature_name] = st.number_input(
                        f"Enter value for '{feature_name}'",
                        value=0,
                        step=1,
                        key=f"manual_input_{feature_name}"
                    )
                elif 'float' in str(original_dtype):
                    manual_input_data[feature_name] = st.number_input(
                        f"Enter value for '{feature_name}'",
                        value=0.0,
                        step=0.1,
                        key=f"manual_input_{feature_name}"
                    )
                else: # Default to text input
                    manual_input_data[feature_name] = st.text_input(
                        f"Enter value for '{feature_name}'",
                        key=f"manual_input_{feature_name}"
                    )
        
        # Button to trigger prediction
        if st.button("Get Manual Prediction"):
            try:
                # Create a DataFrame from manual input for preprocessing
                # Important: The DataFrame must initially have the original feature columns before OHE.
                # Initialize with original feature names
                input_df = pd.DataFrame([manual_input_data])
                
                # Apply the same preprocessing steps
                processed_input_df = input_df.copy()

                for col, encoder_type in st.session_state.encoder_choices.items():
                    if col in processed_input_df.columns: # Check if the column exists in the input
                        processed_input_df[col] = processed_input_df[col].astype(str) # Ensure string for encoding
                        
                        if encoder_type == "Label Encoding":
                            # LabelEncoder can be fit on string and transform, but we must use a fitted one
                            # For prediction, we need to apply the SAME fitted LabelEncoder from training
                            # However, LabelEncoder is usually not stored explicitly for features;
                            # if a feature was LabelEncoded, it's already numeric. This case
                            # assumes the manual input for LabelEncoded features would be numeric.
                            # If the user inputs a categorical string for a LabelEncoded feature, it would fail.
                            # For manual input, we assume the user inputs the numerically encoded value if it was LabelEncoded.
                            pass # Nothing to do here if the user inputs the numeric label
                        elif encoder_type == "One-Hot Encoding":
                            # Use the stored OneHotEncoder
                            if col in st.session_state.ohe_encoders:
                                ohe = st.session_state.ohe_encoders[col]
                                # Transform input data; ensure it's 2D array
                                # handle_unknown='ignore' ensures it doesn't break for new categories
                                ohe_array = ohe.transform(processed_input_df[[col]])
                                ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out([col]), index=processed_input_df.index)
                                processed_input_df = processed_input_df.drop(columns=[col])
                                processed_input_df = pd.concat([processed_input_df, ohe_df], axis=1)
                        elif encoder_type == "Custom Numeric Mapping":
                            # Use the stored custom mapping
                            mapping = st.session_state.manual_numeric_mappings.get(col, {})
                            processed_input_df[col] = processed_input_df[col].map(mapping)
                            if processed_input_df[col].isnull().any():
                                st.warning(f"Manual input for '{col}' resulted in NaN after custom mapping. Ensure input matches trained categories.")


                # Convert all features to numeric, coercing errors
                for col in processed_input_df.columns:
                    processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce')
                
                # Fill NaNs with 0 or a strategy used in training if applicable (for now, just 0)
                processed_input_df = processed_input_df.fillna(0) # Simple imputation for prediction

                # Ensure the processed input DataFrame has the exact same columns and order as during training
                # This is crucial for scikit-learn models
                # Create a blank DataFrame with the expected training columns
                final_input_for_model = pd.DataFrame(columns=st.session_state.final_features_for_model)
                # Assign values from processed_input_df to the correct columns
                for col in st.session_state.final_features_for_model:
                    if col in processed_input_df.columns:
                        final_input_for_model[col] = processed_input_df[col]
                    else:
                        final_input_for_model[col] = 0 # Default to 0 if an OHE column wasn't set (implies original category wasn't present)
                
                # Handle cases where manual input might not have all expected features (e.g. if a numeric column was input but not selected in original features)
                # It's safer to reconstruct based on final_features_for_model

                prediction = st.session_state.model.predict(final_input_for_model)

                st.subheader("Prediction Result:")
                if st.session_state.is_target_continuous:
                    st.success(f"Predicted Value: {prediction[0]:.2f}")
                else:
                    # Inverse transform the prediction if LabelEncoder was used for the target
                    if st.session_state.le_target is not None:
                        predicted_label = st.session_state.le_target.inverse_transform(prediction)
                        st.success(f"Predicted Label: **{predicted_label[0]}**")
                    else:
                        st.success(f"Predicted Label: **{prediction[0]}**")

            except Exception as e:
                st.error(f"Error during manual prediction: {e}. Please ensure all inputs are valid and match the model's expectations.")

    elif prediction_input_method == "Upload CSV for Prediction":
        st.subheader("Upload CSV for Predictions")
        uploaded_predict_file = st.file_uploader("Upload CSV file with features for prediction", type=["csv"], key="predict_csv_uploader")

        if uploaded_predict_file:
            predict_df = pd.read_csv(uploaded_predict_file)
            st.write("Preview of new data:", predict_df.head())

            if st.button("Get Predictions from CSV"):
                try:
                    processed_predict_df = predict_df.copy()

                    # Apply stored encoder choices and mappings
                    for col, encoder_type in st.session_state.encoder_choices.items():
                        if col in processed_predict_df.columns: # Only apply if column exists in the new CSV
                            processed_predict_df[col] = processed_predict_df[col].astype(str) # Ensure string for encoding

                            if encoder_type == "Label Encoding":
                                # If the original feature was label encoded, it implies it was numeric in training.
                                # For new CSV, if it's still categorical, it means the user should have mapped it to numeric.
                                # Or, if it's already numeric, nothing to do.
                                # For simplicity, we assume if it was LabelEncoded and is still object, it should be mapped
                                # However, LabelEncoder cannot transform unseen labels without error.
                                # The best practice here would be to apply the *fitted* LabelEncoder.
                                # Since we don't store it for features (only target), we rely on manual numeric conversion.
                                pass # This case is handled by pd.to_numeric later

                            elif encoder_type == "One-Hot Encoding":
                                if col in st.session_state.ohe_encoders:
                                    ohe = st.session_state.ohe_encoders[col]
                                    # Ensure column is 2D for transform
                                    ohe_array = ohe.transform(processed_predict_df[[col]])
                                    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out([col]), index=processed_predict_df.index)
                                    processed_predict_df = processed_predict_df.drop(columns=[col])
                                    processed_predict_df = pd.concat([processed_predict_df, ohe_df], axis=1)
                            elif encoder_type == "Custom Numeric Mapping":
                                mapping = st.session_state.manual_numeric_mappings.get(col, {})
                                processed_predict_df[col] = processed_predict_df[col].map(mapping)
                                # Handle categories in new CSV not in mapping (will become NaN)
                                if processed_predict_df[col].isnull().any():
                                    st.warning(f"New data for '{col}' contains categories not seen during training and mapped to NaN. Please ensure new data categories match training categories.")


                    # Convert all features to numeric, coercing errors
                    for col in processed_predict_df.columns:
                        processed_predict_df[col] = pd.to_numeric(processed_predict_df[col], errors='coerce')
                    
                    processed_predict_df = processed_predict_df.fillna(0) # Simple imputation for prediction

                    # Ensure the processed input DataFrame has the exact same columns and order as during training
                    final_predict_input = pd.DataFrame(index=processed_predict_df.index)
                    for col in st.session_state.final_features_for_model:
                        if col in processed_predict_df.columns:
                            final_predict_input[col] = processed_predict_df[col]
                        else:
                            final_predict_input[col] = 0.0 # Add missing columns with default 0.0 for OHE features

                    predictions = st.session_state.model.predict(final_predict_input)

                    if st.session_state.is_target_continuous:
                        predict_df['Predicted_Label'] = predictions
                        st.subheader("Predictions for Uploaded CSV:")
                        st.dataframe(predict_df)
                    else:
                        if st.session_state.le_target is not None:
                            predicted_labels = st.session_state.le_target.inverse_transform(predictions)
                            predict_df['Predicted_Label'] = predicted_labels
                        else:
                            predict_df['Predicted_Label'] = predictions
                        st.subheader("Predictions for Uploaded CSV:")
                        st.dataframe(predict_df)

                except Exception as e:
                    st.error(f"Error during CSV prediction: {e}. Please ensure the CSV file has the same features as the training data and that data types are compatible.")

else:
    if uploaded_file:
        st.info("Train a model first to enable the prediction feature.")
    # No message needed if no file uploaded at all, initial message covers it.
