# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def app():
	df = pd.read_csv('student_single.csv')
	st.markdown("<h1 style='text-align: center; color: blue;'>Student performance using single linear regression</h1>", unsafe_allow_html=True)


	X = df[['Previous Scores']]
	y = df['Performance Index']

	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create a linear regression model
	model = LinearRegression()

	# Fit the model on the training data
	model.fit(X_train, y_train)

	# Make predictions on the test data
	y_pred = model.predict(X_test)

	# Evaluate the model
	mse = mean_squared_error(y_test, y_pred)


	# Now you can use the model to make predictions for new data
	st.sidebar.header("Student performance prediction Web App")
	p_len = st.sidebar.slider("Previous Scores", float(df["Previous Scores"].min()), float(df["Previous Scores"].max()))

	if st.sidebar.button("Predict", type="primary"):
		arr = np.array([p_len])
		new_data = arr.reshape(-1, 1)
		prediction = model.predict(new_data)
		st.write("Input Value of Previous Scores is ", p_len)
		st.write("-"*70)
		st.write("Predicted performance Index: ", prediction[0])
		st.write("Mean Squared Error:", mse)
