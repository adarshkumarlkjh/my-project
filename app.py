import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Data Loading and Caching ---
# Load data using caching to improve performance
@st.cache_data
def load_data(file_path):
    """Loads the dataset from the specified path."""
    try:
        data = pd.read_csv(file_path)
        # Drop the unnamed column if it exists
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        return data
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {file_path}. Make sure 'fraudTrain.csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Preprocessing ---
def preprocess_data(df):
    """Preprocesses the input DataFrame."""
    # Drop unnecessary columns (adjust based on your notebook's final feature set)
    columns_to_drop = ['cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Handle missing values - Drop rows with NaN in 'is_fraud' or any feature column
    df_processed.dropna(inplace=True)

    # Convert target to integer if needed (already float/int in notebook)
    if 'is_fraud' in df_processed.columns:
        df_processed['is_fraud'] = df_processed['is_fraud'].astype(int)

    return df_processed

# --- Model Training ---
# Cache the trained model and encoders to avoid retraining on every interaction
@st.cache_resource
def train_model(data):
    """Trains the SVC model and returns the model and encoders."""
    df_processed = preprocess_data(data.copy()) # Preprocess a copy

    if df_processed is None or df_processed.empty:
        st.error("Preprocessing failed or resulted in empty data. Cannot train model.")
        return None, {}

    # Separate features (X) and target (y)
    X = df_processed.drop(columns=['is_fraud'])
    y = df_processed['is_fraud']

    # --- Encoding ---
    encoders = {}
    categorical_cols = ['merchant', 'category', 'gender', 'job']
    for col in categorical_cols:
        if col in X.columns:
            encoder = LabelEncoder()
            # Fit encoder and transform the column
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder # Store fitted encoder

    # --- Train SVC Model ---
    try:
        model = SVC(probability=True) # Enable probability estimates if needed later
        model.fit(X, y)
        st.success("Model trained successfully!")
        return model, encoders, X.columns # Return trained model, encoders, and feature columns used
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, {}, None

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
Welcome to the Credit Card Fraud Detection application.
This app uses a Support Vector Classifier (SVC) model trained on transaction data to predict whether a transaction is fraudulent.
Please enter the transaction details below.
""")

# Load the training data
data = load_data('fraudTrain.csv')

if data is not None:
    # Train the model (or load from cache)
    model, encoders, feature_columns = train_model(data)

    if model and encoders and feature_columns is not None:
        st.sidebar.header("Enter Transaction Details:")

        # --- User Input Fields ---
        input_data = {}

        # Dynamically create input fields based on feature columns
        # Keep track of unique keys for Streamlit widgets
        widget_key_counter = 0

        # Use columns for layout
        col1, col2 = st.sidebar.columns(2)

        for i, col in enumerate(feature_columns):
            target_col = col1 if i % 2 == 0 else col2 # Alternate columns
            widget_key = f"{col}_{widget_key_counter}"
            widget_key_counter += 1

            if col in encoders: # Categorical features - use selectbox
                # Get original categories from the encoder
                try:
                    # Handle cases where encoder might not have classes_ attribute immediately
                    if hasattr(encoders[col], 'classes_'):
                         # Limit the number of options displayed for performance if necessary
                        options = list(encoders[col].classes_)
                        if len(options) > 1000: # Example limit
                             st.sidebar.warning(f"Feature '{col}' has many categories ({len(options)}). Displaying a subset or consider alternative input.")
                             # options = options[:1000] # Or handle differently
                        # Check if options are suitable for selectbox (strings)
                        if all(isinstance(opt, (str, int, float)) for opt in options):
                             input_data[col] = target_col.selectbox(f"Select {col.replace('_', ' ').title()}", options=options, key=widget_key)
                        else:
                             # Fallback or different widget type if classes are complex
                             input_data[col] = target_col.text_input(f"Enter {col.replace('_', ' ').title()} (Encoded)", key=widget_key)
                             target_col.caption(f"Requires encoded value for {col}")
                    else:
                        input_data[col] = target_col.text_input(f"Enter {col.replace('_', ' ').title()} (Encoded)", key=widget_key)
                        target_col.caption(f"Requires encoded value for {col} - Encoder info unavailable.")

                except Exception as e:
                    st.sidebar.error(f"Error creating selectbox for {col}: {e}")
                    # Fallback to text input if selectbox fails
                    input_data[col] = target_col.text_input(f"Enter {col.replace('_', ' ').title()} (Encoded Value)", key=widget_key)

            elif pd.api.types.is_numeric_dtype(data[col]): # Numeric features - use number_input
                # Determine min/max from training data for better input range
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                # Provide a reasonable default or mean
                default_val = float(data[col].mean())
                # Use numpy float types for potentially large ranges
                input_data[col] = target_col.number_input(
                    f"Enter {col.replace('_', ' ').title()}",
                    min_value=np.float64(min_val),
                    max_value=np.float64(max_val),
                    value=np.float64(default_val),
                    step=np.float64(1.0) if pd.api.types.is_integer_dtype(data[col]) else np.float64(0.01), # Adjust step based on type
                    key=widget_key
                )
            else: # Fallback for other types
                input_data[col] = target_col.text_input(f"Enter {col.replace('_', ' ').title()}", key=widget_key)


        # --- Prediction ---
        if st.sidebar.button("Predict Fraud Status", key=f"predict_{widget_key_counter}"):
            try:
                # Create DataFrame for prediction
                predict_df = pd.DataFrame([input_data])

                # Apply encoding using stored encoders
                for col, value in input_data.items():
                    if col in encoders:
                         # Transform the user's selected category string into its encoded integer
                         predict_df[col] = encoders[col].transform([value])[0]

                # Ensure column order matches training data
                predict_df = predict_df[feature_columns]

                 # Handle potential dtype mismatches after encoding/input
                for col in predict_df.columns:
                     if pd.api.types.is_numeric_dtype(data[col]):
                         predict_df[col] = pd.to_numeric(predict_df[col], errors='coerce')
                     # Add more specific dtype conversions if needed

                # Check for NaNs introduced during conversion
                if predict_df.isnull().values.any():
                    st.error("Error: Invalid input detected after processing. Please check your entries.")
                else:
                    # Make prediction
                    prediction = model.predict(predict_df)
                    # prediction_proba = model.predict_proba(predict_df) # Optional: get probabilities

                    # --- Display Results ---
                    st.subheader("Prediction Result:")
                    if prediction[0] == 1:
                        st.error("🚨 Transaction is likely FRAUDULENT")
                    else:
                        st.success("✅ Transaction appears to be LEGITIMATE")

                    # st.write("Prediction Probability:", prediction_proba) # Optional

                    st.markdown("---")
                    st.subheader("Input Data Provided:")
                    # Display user input for confirmation
                    display_input = {}
                    for k, v in input_data.items():
                        display_input[k.replace('_', ' ').title()] = v
                    st.json(display_input)


            except KeyError as e:
                st.error(f"Error during prediction: Feature mismatch - {e}. Check input data. Ensure all required features are provided.")
            except ValueError as e:
                 st.error(f"Error during prediction: Could not process input - {e}. Please ensure inputs match expected formats (e.g., numbers for numeric fields).")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.error("Input data at time of error:")
                st.write(input_data) # Log input data for debugging

    else:
        st.warning("Model could not be trained. Please check the data source and preprocessing steps.")
else:
    st.warning("Dataset could not be loaded. Cannot proceed with the application.")

st.markdown("""
---
*Disclaimer: This prediction is based on a machine learning model and should be used for informational purposes only.*
""")
