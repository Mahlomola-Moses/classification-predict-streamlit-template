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
import seaborn as sns
sns.set(font_scale=1.4)
from streamlit.caching import cache

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

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

def clean_tweets(message, remove_stopwords=False, eda = False, lemma=True):
    """
    A function to preprocess tweets for model training and exploratory data analysis
    :param message: String, message to be cleaned
    :param remove_stopwords: Bool, defualt is False, set to true to remove stopwords
    :param eda: Bool, defualt is False, set to true to return cleaned but readable string
    :param lemma: Bool, deafautl is True, lemmatize.
    return: String, message
    """    
    if eda == False:
        # change all words into lower case
        message = message.lower()
    
    if eda == True:
        message = re.sub('RT|rt','retweet',message)

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
    
    if remove_stopwords == True:     
        # remove stopwords if wordcloud
        stop_words = stopwords.words('english')
        stop_words.append('web')
        stop_words.append('climate')
        stop_words.append('change')
        stop_words.append('global')
        stop_words.append('warming')
        stop_words.append('retweet')
        stop_words.append('u')
        message = ' '.join([word for word in message.split(' ') if not word in stop_words])
    
    if lemma == True:    
      # lemmatizing all words
        lemmatizer = WordNetLemmatizer()
        message = [lemmatizer.lemmatize(token) for token in message.split(" ")]
        message = [lemmatizer.lemmatize(token, "v") for token in message]
        message = " ".join(message)

    return message

def hashtag_extract(x):
    """
    Function to extract the hashtags from the messages column
    """
    hashtags = []    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

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

def plot_message_len(messages):
	fig, ax = plt.subplots(figsize=(30,10))

	#Positive 
	sns.distplot(messages, hist=True, kde=True,
				bins=10, color = 'blue', 
				ax = ax)
	fig, ax = plt.subplots(dpi = 600, figsize=(6, 9))
	ax.set_xlabel('Message Length')
	ax.set_ylabel('Density')
	return st.pyplot(fig)

def plot_hashtags(df, n):
	# selecting top 10 most frequent hashtags     
	df = df.nlargest(columns="Count", n = n) 
	fig, ax = plt.subplots(figsize = (12.5, 10), dpi = 500)
	ax = sns.barplot(data=df, x= "Count", y = "Hashtag", palette='winter')
	ax.set(xlabel = 'Count')
	return st.pyplot(fig)
	
def create_wordcloud(tweets, n):
	fig, ax = plt.subplots(figsize=(22, 15), dpi = 1000)
	wc = WordCloud(width=800, height=500, 
               background_color='black',
               max_words = n,
               max_font_size=150, random_state=42)
	wc.generate(tweets)
	plt.imshow(wc, interpolation='bilinear')
	plt.axis("off")

	return st.pyplot(fig)

# Load your raw data
train = pd.read_csv("train.csv")
@st.cache
def load_bulk_data():

	pro_len = train[train['sentiment']==1]['message'].str.len()
	news_len = train[train['sentiment']==2]['message'].str.len()
	anti_len = train[train['sentiment']==-1]['message'].str.len()
	neutral_len = train[train['sentiment']==0]['message'].str.len()

	# extracting hashtags from train tweets
	anti_hashtag = hashtag_extract(train['message'][train['sentiment'] == -1])
	neutral_hashtag = hashtag_extract(train['message'][train['sentiment'] == 0])
	pro_hashtag = hashtag_extract(train['message'][train['sentiment'] == 1])
	news_hashtag = hashtag_extract(train['message'][train['sentiment'] == 2])

	anti_hash = nltk.FreqDist(sum(anti_hashtag,[]))
	anti_hash_df = pd.DataFrame({'Hashtag': list(anti_hash.keys()),
					'Count': list(anti_hash.values())})
	pro_hash = nltk.FreqDist(sum(pro_hashtag,[]))
	pro_hash_df = pd.DataFrame({'Hashtag': list(pro_hash.keys()),
					'Count': list(pro_hash.values())})
	neutral_hash = nltk.FreqDist(sum(neutral_hashtag,[]))
	neutral_hash_df = pd.DataFrame({'Hashtag': list(neutral_hash.keys()),
					'Count': list(neutral_hash.values())})
	news_hash = nltk.FreqDist(sum(news_hashtag,[]))
	news_hash_df = pd.DataFrame({'Hashtag': list(news_hash.keys()),
					'Count': list(news_hash.values())})

	train['message_clean_eda']=train['message'].apply(lambda x: clean_tweets(message =x, remove_stopwords=True, 
																			eda=True, lemma=False))

	news_tweets = ' '.join([text for text in train['message_clean_eda']
							[train['sentiment'] == 2]])
	pro_tweets = ' '.join([text for text in train['message_clean_eda']
						[train['sentiment'] == 1]])
	neutral_tweets = ' '.join([text for text in train['message_clean_eda']
							[train['sentiment'] == 0]])
	anti_tweets = ' '.join([text for text in train['message_clean_eda']
							[train['sentiment'] == -1]])
	tweet_list = [news_tweets, pro_tweets,neutral_tweets, anti_tweets]

	return pro_len, news_len, anti_len, neutral_len, anti_hash_df, pro_hash_df, neutral_hash_df, news_hash_df, tweet_list

pro_len, news_len, anti_len, neutral_len, anti_hash_df, pro_hash_df, neutral_hash_df, news_hash_df, tweet_list = load_bulk_data()

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Make A Prediction",  "Gain Insight", "Assess Our Models"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.header('\n')
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Make A Prediction":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.header('\n')
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
		st.subheader("Climate change tweet classification")
		st.title("Model Assessment")
		st.info("Graph the performance of our trained machine learning models below")

		clf_performance_df = load_data('clf_performance_df')
		ordered_CV_clf_performance_df = load_data('ordered_CV_clf_performance_df')
		best_performing_df = load_data('best_performing_df')
		CV_best_performing_df = load_data('CV_best_performing_df')
		metrics_new_data_split_df = load_data('metrics_new_data_split_df')

		options = ["All Models", "Top 4", "Hyperparameter Tuned Top 4", "The Best"]
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
		st.header("Exploratory Data Analysis")
		st.subheader('Explore the labled data.')
		st.subheader('\n ')
		st.sidebar.markdown('Select sentiment:')
		ANTI = st.sidebar.checkbox('Anti')
		NEUTRAL = st.sidebar.checkbox('Neutral')
		PRO = st.sidebar.checkbox('Pro')
		NEWS = st.sidebar.checkbox('News')
		st.sidebar.markdown('Select info:')
		wordcloud = st.sidebar.checkbox('Wordclouds')
		hashtags = st.sidebar.checkbox('Hashtags')
		mentions = st.sidebar.checkbox('Mentions')
		message_len = st.sidebar.checkbox('Message length')

		if ANTI:
			st.write('\+ tweets labled Anti')
		if NEUTRAL:
			st.write('\+ tweets labled Neutral')
		if PRO:
			st.write('\+ tweets labled Pro')
		if NEWS:
			st.write('\+ tweets labled News')
		if wordcloud: 
			st.title('Wordcloud')
			n = st.slider('Max Words',15, 60, 30, 15)
			if ANTI:
				st.subheader('Most Popular Words For Anti Tweets')
				create_wordcloud(tweet_list[3], n)
			if NEUTRAL:
				st.subheader('Most Popular Words For Neutral Tweets')
				create_wordcloud(tweet_list[2], n)
			if PRO:
				st.subheader('Most Popular Words For Pro Tweets')
				create_wordcloud(tweet_list[1], n)
			if NEWS:
				st.subheader('Most Popular Words For News Tweets')
				create_wordcloud(tweet_list[0], n)
		if hashtags:
			st.title('Popular Hashtags')
			h = st.slider('Max Hashtags',8, 32, 12, 4)
			if ANTI:
				st.subheader('Most Popular Hashtags for Anti Tweets')
				plot_hashtags(anti_hash_df, h)
			if NEUTRAL:
				st.subheader('Most Popular Hashtags For Neutral Tweets')
				plot_hashtags(neutral_hash_df, h)
			if PRO:
				st.subheader('Most Popular Hashtags For Pro Tweets')
				plot_hashtags(pro_hash_df, h)
			if NEWS:
				st.subheader('Most Popular Hashtags For News Tweets')
				plot_hashtags(news_hash_df, h)
		if message_len:
			st.title('Tweet Lengths')
			if ANTI:
				st.subheader('Tweet length Distribution - Anti')
				plot_message_len(anti_len)
			if NEUTRAL:
				st.subheader('Tweet length Distribution - Neautral')
				plot_message_len(neutral_len)
			if PRO:
				st.subheader('Tweet length Distribution - Pro')
				plot_message_len(pro_len)
			if NEWS:
				st.subheader('Tweet length Distribution - News')
				plot_message_len(news_len)

			
				

			
		
			
				

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
