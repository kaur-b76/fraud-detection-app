import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the Model and the Data
@st.cache_data
def load_data():
    # Loading the full dataset from Source 4
    df = pd.read_csv('AIML Dataset.csv')
    return df


# Load the saved pipeline
model = joblib.load('fraud_detection_pipeline.pkl')
df = load_data()

st.title("🛡️ Fraud Detection Inspector")

# 2. Front-End: Data Selection
st.header("Step 1: Browse & Select a Transaction")

# We still use .head(20) for the visual table so your browser doesn't freeze
st.write(f"Previewing first 20 rows. Total records available: {len(df):,}")
st.dataframe(df.head(20))

# THIS LINE IS THE FIX: It allows you to select ANY row up to 6,362,619
row_index = st.number_input("Enter any Row Number (Index) from 0 to 6,362,619:",
                            min_value=0,
                            max_value=len(df) - 1,
                            value=0,
                            step=1)

# Pull the specific record the user typed in
selected_record = df.iloc[[row_index]]

# Separate the 'isFraud' answer so the model doesn't "cheat"
if 'isFraud' in selected_record.columns:
    actual_status = selected_record['isFraud'].values[0]
    X_test = selected_record.drop('isFraud', axis=1)
else:
    actual_status = None
    X_test = selected_record

# 3. Execution: Run the Prediction
st.header("Step 2: Analysis")
if st.button('Analyze Transaction'):
    # The model predicts based on the row you selected
    prediction = model.predict(X_test)

    if prediction == 1:
        st.error(f"🚨 AI PREDICTION: FRAUD DETECTED at Row {row_index}!")
    else:
        st.success(f"✅ AI PREDICTION: LEGITIMATE transaction at Row {row_index}.")

    # Compare with the actual truth from the CSV
    if actual_status is not None:
        actual_label = "Fraud" if actual_status == 1 else "Legitimate"
        st.info(f"**Actual Status in Dataset:** {actual_label}")

    # 4. Visualization: Feature Importance
    st.subheader("Clue Analysis")

    try:
        # Using model[-1] to avoid the KeyError: 'model'
        final_estimator = model[-1]
        importance = final_estimator.feature_importances_
        features = X_test.columns

        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
        plt.title(f"Factors influencing Analysis for Row {row_index}")
        st.pyplot(fig)

    except (AttributeError, KeyError, IndexError):
        st.info("Feature importance details are hidden for this model.")