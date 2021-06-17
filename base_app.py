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

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
			0:"Neutral:",
			1: "Pro: Believes in man-made climate change",
			2: "News",
			-1: "Anti: Doesn't believe in man-made climate change" }
	return switcher.get(x[0], x)

def clean_tweets(message):
    """
    A function to preprocess tweets for model training and exploratory data analysis
    :param message: String, message to be cleaned
    :param remove_stopwords: Bool, defualt is False, set to true to remove stopwords
    :param eda: Bool, defualt is False, set to true to return cleaned but readable string
    :param lemma: Bool, deafautl is True, lemmatize.
    return: String, message
    """    
    # replace all url-links with url-web
    url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    message = re.sub(url, 'web', message)
    # removing all punctuation and digits
    message = re.sub(r'[-]',' ',message)
    message = re.sub(r'[_]', ' ', message)
    message = re.sub(r'[^\w\s]','',message)
    message = re.sub('[0-9]+', '', message) 
    message = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~âã¢¬¦¢’‘‚…]', ' ', message)
    message = re.sub("â|ã|Ã|Â", " ", message)  # removes strange character 
    message = re.sub("\\s+", " ", message)  # fills white spaces
    message = message.lstrip()  # removes whitespaces before string
    message = message.rstrip()  # removes whitespaces after string 
   
    # lemmatizing all words
    lemmatizer = WordNetLemmatizer()
    message = [lemmatizer.lemmatize(token) for token in message.split(" ")]
    message = [lemmatizer.lemmatize(token, "v") for token in message]
    message = " ".join(message)

    return message

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
				xlim = [0.6, 0.81]
			if column == 'F1-Accuracy':
				xlim = [0.6, 0.81]
			if column == 'F1-Macro':
				xlim = [0.5, 0.8]  
			if column == 'Execution Time':
				df = df.sort_values(column, ascending=False)
				xlim = [0.6, 295]  
			if column == 'CV_Mean':
				xlim = [0.6, 0.76]			
			if column == 'CV_Std':
				xlim = [0.002, 0.009]
			if 'Flair_TextClassifier' in df.index: 
				figsize = (14, 5.8) 			
				title = column
	
			else:
				figsize = (10, 5.8) 
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
        xlim = [0.6, 0.8]
    
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi = 550)
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
	st.header('\n')
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Make A Prediction",  "Gain Insight", "Assess Our Models"]
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
	if selection == "Make A Prediction":
		options = ['Linear Support Vector Classifier', 'Logistic Regression', 'Stochastic Gradient Descent Classifier', 'Ridge Classifier']	
		st.info("Select a classification model and enter some text to predict the sentiment.")
		model = st.selectbox('Select A Model', options)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		cleaned_tweet_text = clean_tweets(tweet_text)
		classify = st.button("Classify")

		if classify and model == options[0]:   # Prediction with LinearSVC
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([cleaned_tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Predicted sentiment as: "+ switch_demo(prediction))
		
		elif classify and model == options[1]:   # Logistic Regression
			vect_text = tweet_cv.transform([cleaned_tweet_text]).toarray()
			predictor = joblib.load(open(os.path.join("resources/LogReg.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Predicted sentiment as: "+ switch_demo(prediction))
		
		elif classify and model == options[2]:   # SGDClassifier
			vect_text = tweet_cv.transform([cleaned_tweet_text]).toarray()
			predictor = joblib.load(open(os.path.join("resources/SGDClassifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Predicted sentiment as: "+ switch_demo(prediction))
		
		elif classify and model == options[3]:   # Ridge
			vect_text = tweet_cv.transform([cleaned_tweet_text]).toarray()
			predictor = joblib.load(open(os.path.join("resources/RidgeClassifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Predicted sentiment as: "+ switch_demo(prediction))
			

	if selection == "Assess Our Models":
		st.header('\n \n')
		st.header("Model Assessment")
		st.info("Graph the performance of our trained machine learning models below")

		clf_performance_df = load_data('clf_performance_df')
		ordered_CV_clf_performance_df = load_data('ordered_CV_clf_performance_df')
		best_performing_df = load_data('best_performing_df')
		CV_best_performing_df = load_data('CV_best_performing_df')
		metrics_new_data_split_df = load_data('metrics_new_data_split_df')

		options = ["All models", "Top 4", "Hyperparameter tuned Top 4", "The Best"]
		option = st.selectbox('1. Select models to evaluate:', options)	
		methods = [' ', 'Train Test Split', 'Cross Validation']
		method = st.selectbox('2. Select the training method:', methods)
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
				

	if selection == "Gain Insight":
		st.header('\n')
		st.header("Exploratory Data Analysis")
		st.info('Explore the labled data.')
		
		sentiment1 = st.sidebar.checkbox('Anti')
		sentiment2 = st.sidebar.checkbox('Neutral')
		sentiment3 = st.sidebar.checkbox('Pro')
		sentiment4 = st.sidebar.checkbox('News')
		wordcloud = st.sidebar.checkbox('Wordclouds')
		hashtags = st.sidebar.checkbox('Hashtags')
		mentions = st.sidebar.checkbox('Mentions')
		message_len = st.sidebar.checkbox('Message length')

		if sentiment1:
			st.write('Information on tweets labled Anti')
		if sentiment2:
			st.write('Information on tweets labled Neutral')
		if sentiment3:
			st.write('Information on tweets labled Pro')
		if sentiment4:
			st.write('Information on tweets labled News')
		if wordcloud: 
			
		if hashtags:
		if mentions:
		if message_len:
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
