import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score



def app():
  
  df = pd.read_csv("student_single.csv")

  st.set_option("deprecation.showPyplotGlobalUse", False)

  # <ins>Underlined Heading 1</ins>

  st.markdown("<h1 style='text-align: center; color: red;'>Google Developer Student Clubs</h1>", unsafe_allow_html=True)

  st.markdown("<h2 style='text-align: center; color: green;'>Student Performance Visualisation Web App</h2>", unsafe_allow_html=True)


  st.sidebar.title("Student Data Visualisation")
  if st.sidebar.button('Display Raw data', type="primary"):
    st.subheader("Student Data Set", divider='rainbow')
    st.dataframe(df)
    st.write("Number of Rows: ", df.shape[0])
    st.write("Number of Columns: ", df.shape[1])

  x = df.describe()
  if st.sidebar.button("Describe", type="primary"):
    st.write(x)


  # p = df['Performance Index'].min()
  # q = df['Performance Index'].max()
  # pq = st.sidebar.slider("Select Performance Index", p, q)

  # min_ = df["Previous Scores"].min()
  # max_ = df["Previous Scores"].max()
  # min_max = st.sidebar.slider("Select Previous Scores:", min_, max_)

  if st.sidebar.button("Scatter Plot", type="primary"):
    fig, ax = plt.subplots()
    ax.scatter(df["Previous Scores"], df['Performance Index'])
    ax.set_xlabel("Performance Index")
    ax.set_ylabel("Previous Scores")
    ax.set_title(f"Scatter Plot of Previous Scores vs Performance Index", color = "r")
    st.pyplot(fig)


  X = df.drop(['Performance Index'], axis = 1)
  y = df['Performance Index']
  columns = X.columns
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  X = pd.DataFrame(X, columns = columns)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False, stratify = None)

  st.sidebar.title("Visualisation after prediction")
  if st.sidebar.button("Scaled DataFrame", type="primary"):
    st.subheader("DataSet", divider = 'rainbow')
    st.dataframe(X)
    st.write("Number of Rows: ", df.shape[0])
    st.write("Number of Columns: ", df.shape[1])
  #Linear Regression
  #if st.button("Single Linear Regression", type="primary"):
  if st.sidebar.button("Single Linear Regression", type="primary"):
      st.title("Linear Regression")

      lr = LinearRegression()
      lr.fit(X_train, y_train)
      train_prediction = lr.predict(X_train)
      test_prediction = lr.predict(X_test)
      train_mse = mean_squared_error(y_train, train_prediction)
      test_mse = mean_squared_error(y_test, test_prediction)
      st.write("Mean squared error for train set = ", train_mse)
      st.write("Mean squared error for test set = ", test_mse)
      # st.write(X_train.min())
      # st.write(X_train.max())



      plt.figure(figsize = (8,6))
      plt.scatter(y_train, train_prediction, c = 'blue', label = 'Model Prediction')
      plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], c = 'r', linestyle = '--', label = "Perfect Prediction")
      plt.xlabel('True Values')
      plt.ylabel('Predicted Values')
      plt.title('Training vs Predicted Values')
      plt.legend()
      plt.grid(True)
      st.pyplot()

      plt.figure(figsize=(8, 6))
      plt.scatter(y_test, test_prediction, c='blue', label='Model Predictions')
      plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
      plt.xlabel('True Values')
      plt.ylabel('Predicted Values')
      plt.title('Testing True vs Predicted Values')
      plt.legend()
      plt.grid(True)
      st.pyplot()


####............________________----------------------------____________________________________---------------









  



