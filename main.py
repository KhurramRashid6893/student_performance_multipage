import streamlit as st
import numpy as np
import pandas as pd

import home
import student_single
import student_multi
import predict_multi
import predict_single
 
# df = pd.read_csv("student_single.csv")
st.set_page_config(page_title = 'Student Performance',
                    page_icon = ':Student:',
                    layout = 'centered',
                    initial_sidebar_state = 'auto'
                    )

pages_dict = {"Home": home,
		      "Single Linear Regression": student_single,
              "Multiple Linear Regression": student_multi,
              "Student performance using single linear regression": predict_single,
              "Student performance using multiple linear regression": predict_multi

              }

st.sidebar.title("Navigation")
user_choice = st.sidebar.radio("Go to", tuple(pages_dict.keys()))
if user_choice == "Home":
  home.app()
else:
  selected_page = pages_dict[user_choice]
  selected_page.app()