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
import numpy as np
import matplotlib.pyplot as plt
from streamlit.caching import cache

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Model metrics
@st.cache 
def load_data(df):
	dataframe = pd.read_csv(df + '.csv', index_col = 0)
	return dataframe

def switch_demo(x):
	switcher = {
			0:"Neutral",
			1: "Pro",
			2: "News",
			-1: "Anti" }
	return switcher.get(x[0], x)

def graph_model_performances(df, column):
		"""
		A function to graph model performances from a dataframe. 
		:param df: Dataframe
		:param column: String, column to sort by
		return: Graph
		"""  
		if column == ' ': 
			return 
		else:	
			df = df.sort_values(column, ascending=True)
			
			if column == 'F1-Weighted':
				xlim = [0.5, 0.8]
			if column == 'F1-Accuracy':
				xlim = [0.6, 0.82]
			if column == 'F1-Macro':
				xlim = [0.2, 0.73]  
			if column == 'Execution Time':
				df = df.sort_values(column, ascending=False)
				xlim = [0.6, 295]  
			if column == 'CV_Mean':
				xlim = [0.6, 0.76]			
			if column == 'CV_Std':
				xlim = [0.002, 0.009]
			if 'Flair_TextClassifier' in df.index: 
				figsize = (14, 5.5) 			
				title = column
	
			else:
				figsize = (10, 5.5) 
				title = False 

			fig, ax = plt.subplots(figsize=figsize, dpi = 550)

			df.plot(y=column, kind='barh', xlim=xlim, color= '#18330C', edgecolor = '#8C1010', 
							fontsize=16, title= title, ax = ax, width =0.3)
		
		
		return  st.pyplot(fig), st.dataframe(df.sort_values(column, ascending= False))

def graph_model_improvement(tuned_models_performance, models_performance, column):
    """
    A function to visualise model improvements after hyperparameter tuning 
    :param tuned_models_performance: Dataframe of model performances after tuning
    :param models_performance:       Dataframe of model performances beofre tuning
    :column:                         String, column to sort by
    return: Graph
    """  
    after = tuned_models_performance.sort_values(column,ascending=True)
    before = models_performance.sort_values(column,ascending=True)
    
    if column == 'Execution Time':
        xlim = [0.9, 220]
        after = tuned_models_performance.sort_values(column,ascending=False)
        before = models_performance.sort_values(column,ascending=False)
    else:
        xlim = [0.55, 0.8]
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi = 550)
    ax.set_xlim(xlim)
    plt.rcParams['font.size'] = '12'

    models_after_tuning = after[column].index
    metrics_after = after[column]
    metrics_before = before[column][models_after_tuning]

    after_tuning = ax.barh(y= models_after_tuning, width= metrics_after, height =0.3, color= 'blue', 
                                   edgecolor = 'red',label = 'AFTER TUNING')
    before_tuning = ax.barh(y=models_after_tuning, width= metrics_before, height =0.3, color= '#18330C', 
                        edgecolor = 'red', label = 'BEFORE TUNING')
    ax.set_title(column)

    return st.pyplot(fig), st.dataframe(tuned_models_performance.sort_values(column, ascending= False))

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
	options = ["Prediction", "Information", "Model Assessment"]
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
		options = ['Linear Support Vector', 'Logistic Regression', 'SGD Classifier', ' Ridge Classifier']	
		st.sidebar.selectbox('Choose Model', options)
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Predicted sentiment as : "+switch_demo(prediction))


	if selection == "Model Assessment":
		st.header('\n \n')
		
		st.header("Model Assessment")
		st.subheader("Graphing the Performance of Different Machine Learning Models")

		clf_performance_df = load_data('clf_performance_df')
		ordered_CV_clf_performance_df = load_data('ordered_CV_clf_performance_df')
		best_performing_df = load_data('best_performing_df')
		CV_best_performing_df = load_data('CV_best_performing_df')
		metrics_new_data_split_df = load_data('metrics_new_data_split_df')

		options = ["All models", "Top 4 off the shelf", "Hyperparameter tuned Top 4", "The Best"]
		option = st.selectbox('1. Select models to evaluate:', options)	
		methods = [' ', 'Train Test Split', 'Cross Validation']
		method = st.selectbox('2. Select an evaluation method:', methods)
		metrics = [' ', 'F1-Accuracy', 'F1-Macro', 'F1-Weighted', 'Execution Time', 'CV_Mean', 'CV_Std']	

		if option == options[0]:
			if method == methods[1]:
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[:5])
				if options[0]:				 
						a = 1+1
				
				graph_model_performances(clf_performance_df, column)
			
			elif method == methods[2]:
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[4:])
				if options[0]:				 
						a = 1+1
				
				graph_model_performances(ordered_CV_clf_performance_df, column)

		
		elif option == options[1]:
			if method == methods[1]:	
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[:5])
				if options[0]:				 
						a = 1+1
				graph_model_performances(clf_performance_df.sort_values('F1-Accuracy')[-4:], column)
			
			elif method == methods[2]:
				metrics000 = [' ', 'F1-Accuracy', 'F1-Macro', 'F1-Weighted', 'Execution Time', 'CV_Mean', 'CV_Std']	
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[4:])
				if options[0]:				 
						a = 1+1
				
				graph_model_performances(ordered_CV_clf_performance_df[:-5], column)
		
		elif option == options[2]: 
			if method == methods[1]:	
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[:5])
				if column == metrics[0]:
					a = 1+1
				
				else:
				    graph_model_improvement(best_performing_df, clf_performance_df, column)
			
			elif method == methods[2]:
				metrics000 = [' ', 'F1-Accuracy', 'F1-Macro', 'F1-Weighted', 'Execution Time', 'CV_Mean', 'CV_Std']	
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[4:])
				if options[0]:				 
						a = 1+1
				
				graph_model_improvement(CV_best_performing_df, ordered_CV_clf_performance_df, column)

		elif option == options[3]: 
			if method == methods[1]:	
				column = st.selectbox('3. Select an evaluation metric:',
						     metrics[:5])
				if column == metrics[0]:
					a = 1+1
				
				else:
				    graph_model_performances(metrics_new_data_split_df, column)
			
			elif method == methods[2]:
				st.write('Comapring the models to the flair text classifier neural network by means of cross validation \
				is too computationally expensive and therefore only a train test split was carried out.')
				

			

			


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
