"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

#displaying
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
#Read the saved csv of the performance summary into a dataframe
clf_performance_df = pd.read_csv('clf_performance_df.csv', index_col = 0) 
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
def switch_demo(x):

	switcher = {
			0:"Neutral",
			1: "Pro",
			2: "News",
			-1: "Anti"
			
			
		}

	return switcher.get(x[0], x)
def graph_model_performances(df, column):
    """
    A function to graph model performances from a dataframe. 
    :param df: Dataframe
    :param column: String, column to sort by
    return: Graph
    """  
    df = df.sort_values(column, ascending=True)
    
    if column == 'F1-Weighted':
        xlim = [0.5, 0.8]
    if column == 'F1-Accuracy':
        xlim = [0.5, 0.82]
    if column == 'F1-Macro':
        xlim = [0.5, 0.75]  
    if column == 'Execution Time':
        df = df.sort_values(column, ascending=False)
        xlim = [0.6, 291]  
    if column == 'CV_Mean':
        xlim = [0.6, 0.76]
    if column == 'CV_Std':
        xlim = [0.002, 0.009]
        
    graph = df.plot(y=column, 
                    kind='barh', 
                    xlim=xlim, 
                    color= 'cyan', 
                    edgecolor = 'blue',
                    figsize=(10, 8), 
                    fontsize=16)
    return  graph

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LogReg.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Predicted sentiment as : "+switch_demo(prediction))
			display(clf_performance_df)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
