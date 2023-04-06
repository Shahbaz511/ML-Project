from turtle import color
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import math
import joblib
detection1=joblib.load('model.pkl')

def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def detection(mean_radius,mean_texture,mean_area,mean_concavity,mean_symmetry,mean_fractal_dimension,texture_error,perimeter_error,concave_points_error,worst_radius,worst_perimeter,worst_area,worst_compactness):  
   
    prediction=detection1.predict([[float(mean_radius),float(mean_texture),float(mean_area), float(mean_concavity),
       float(mean_symmetry),float(mean_fractal_dimension),float(texture_error),
       float(perimeter_error),float(concave_points_error),float(worst_radius),
       float(worst_perimeter),float(worst_area), float(worst_compactness)]])
    return prediction

  
def add_bg_from_url():
    st.set_page_config(layout="wide")
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://healthitanalytics.com/images/site/article_headers/_normal/Getty_correct_size_AI_lung.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title     
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-White:LavenderBlush;padding:13px">
    <h1 style ="color:white;text-align:center;">Cancer Classification Using Machine Learning </h1>

    </div>
    """
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    #st.date_input("Select Future date")
    mean_radius=st.text_input("Enter the Value Of mean_radius")
    mean_texture=st.text_input("Enter the Value Of mean_texture")
    mean_area=st.text_input("Enter the Value Of mean_area ")
    mean_concavity=st.text_input("Enter the Value Of mean_concavity")
    mean_symmetry=st.text_input("Enter the Value Of mean_symmetry")
    mean_fractal_dimension=st.text_input("Enter the Value Of mean_fractal_dimension")
    texture_error=st.text_input("Enter the Value Of texture_error")
    perimeter_error=st.text_input("Enter the Value Of perimeter_error")
    concave_points_error=st.text_input("Enter the Value Of concave_points_error")
    worst_radius=st.text_input("Enter the Value Of worst_radius")
    worst_perimeter=st.text_input("Enter the Value Of worst_perimeter")
    worst_area=st.text_input("Enter the Value Of worst_areast")
    worst_compactness=st.text_input("Enter the Value Of worst_compactness") 
   
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result

    
    if st.button("Detection Of Cancer "):
        result=detection(mean_radius,mean_texture,mean_area,mean_concavity,mean_symmetry,mean_fractal_dimension,texture_error,perimeter_error,concave_points_error,worst_radius,worst_perimeter,worst_area,worst_compactness) 
   
        if result==0:
            result="Benign (noncancerous)"
            
        elif result==1:
            result="Malignant (cancerous)"
        
        else:
            result="Error"
         
        
      #st.image('image22.jpg')  
  
    st.success('Detection Of the Cancer------->>{}'.format(result))
if __name__=='__main__':
    main()
