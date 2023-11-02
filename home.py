import streamlit as st
from PIL import Image

def app():
	st.markdown("<h1 style='text-align: center; color: red;'>Student Performance web App</h1>", unsafe_allow_html=True)
	
	st.markdown("<h1 style='text-align: center; color: blue;'>Google Developer Student Clubs</h1>", unsafe_allow_html=True)
	image= Image.open("gdsc_aum.jpg")
	st.image(image,use_column_width = False)
	
	
    