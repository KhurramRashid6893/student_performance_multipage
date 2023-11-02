# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def app():
	df = pd.read_csv('Student.csv')
	st.markdown("<h1 style='text-align: center; color: blue;'>Student performance using multiple linear regression</h1>", unsafe_allow_html=True)


	X = df[['Hours Studied', 'Previous Scores','Sleep Hours', 'Sample Question Papers Practiced']]
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
	h_len = st.sidebar.slider("Hours Studied", float(df["Hours Studied"].min()), float(df["Hours Studied"].max()))
	p_len = st.sidebar.slider("Previous Scores", float(df["Previous Scores"].min()), float(df["Previous Scores"].max()))
	s_len = st.sidebar.slider("Sleep Hours", float(df["Sleep Hours"].min()), float(df["Sleep Hours"].max()))
	q_len = st.sidebar.slider("Sample Question Papers Practiced", float(df["Sample Question Papers Practiced"].min()), float(df["Sample Question Papers Practiced"].max()))
	if st.sidebar.button("Predict"):
		new_data = np.array([[h_len,p_len, s_len, q_len]])
		prediction = model.predict(new_data)
		st.write("Input Value of Hours Studied is ", h_len)
		st.write("Input Value of Previous Scores is", p_len)
		st.write("Input Value of Sleep Hours is", s_len)
		st.write("Input Value of Sample Question Papers Practiced is ", q_len)
		st.write("-"*100)
		st.write("Predicted performance Index: ", prediction[0])
		st.write("Mean Squared Error:", mse)
