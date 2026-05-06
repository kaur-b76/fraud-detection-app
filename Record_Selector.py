import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the Model and the Data
# We load the dataset so the user can see all the records
@st.cache_data
def load_data():
    df = pd.read_csv('AIML Dataset.csv')
    return df


model = joblib.load('fraud_detection_pipeline.pkl')  # Your saved Rule Book
df = load_data()

st.title("🛡️ Fraud Detection Inspector")

# 2. Front-End: Data Selection
st.header("Step 1: Browse & Select a Transaction")
st.write("Below are all the records from the dataset. Choose one to test:")

# Display the dataframe so the user can browse records
st.dataframe(df.head(20))

# Create a selection slider or box based on the row index
row_index = st.number_input("Enter the Row Number (Index) you want to test:",
                            min_value=0,
                            max_value=len(df) - 1,
                            value=0)

# Pull the specific record
selected_record = df.iloc[[row_index]]
# We drop the 'label' column if it exists so we don't cheat by showing the answer
if 'label' in selected_record.columns:
    actual_status = selected_record['label'].values[0]
    X_test = selected_record.drop('label', axis=1)
else:
    X_test = selected_record

# 3. Execution: Run the Prediction
st.header("Step 2: Analysis")
if st.button('Analyze Transaction'):
    # Get the prediction and the probability (confidence)
    prediction = model.predict(X_test)[cite: 2]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED at Row {row_index}!")
    else:
        st.success(f"✅ Row {row_index} appears to be a LEGITIMATE transaction.")

    # 4. Visualization: Why did it choose this?
    st.subheader("Clue Analysis (Feature Importance)")

    # Getting importance from the model inside the pipeline
    importance = model.named_steps['model'].feature_importances_
    features = X_test.columns

    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
    plt.title("Key Factors Influencing the Guard")
    st.pyplot(fig)

    st.info("The longest bars represent the 'clues' that weighed most heavily in this decision.")
