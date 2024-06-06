import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.markdown(
    """
    <style>
    body {
        background-image: url('background.jpg');
        background-size: cover;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #87CEEB; /* Sky Blue color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Final Project")
# Display dataset
data=pd.read_csv('diabetes.csv')
data.head()
st.write("### DiabetesDataset")
st.write(data.head())
# Data preprocessing
st.write("### Data Preprocessing")
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
st.write(f"### Model Accuracy: {accuracy:.2f}%")


# User input for prediction
st.write("### Predict Diabetes Outcome")
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.write(f"PredictedDiabetes Outcome: {prediction}")


# Load and display the image
col1, col2 = st.columns(2)
with col1:
    image_path = 'diabetespic.png'
    st.image(image_path, caption='', width=350)
with col2:
    image_path = 'diabetes2.png'
    st.image(image_path, caption='', width=250)
