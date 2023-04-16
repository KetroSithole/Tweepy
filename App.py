
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import streamlit as st
import pandas as pd
import plost
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#from sklearn.feature_extraction.text import CountVectorizer
import string
import re
#matplotlib inline
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
from PIL import Image

#import text2emotion as te
import os 
import base64
import warnings
import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
st.set_option('deprecation.showPyplotGlobalUse', False)
from nltk.corpus import stopwords


main_bg = "download (5).jpg"
main_bg_ext = "jpg"

#############


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  
st.title("Crypto Analysis Tool ")
st.header("Detection of  Crypto Fraud")
 
st.sidebar.header('OverView AI')    
st.sidebar.subheader('Navigation')

#reading the dataset
data1=pd.read_excel("Clean_twitter_data.xlsx")
data2=pd.read_csv("Twitter_text_data.csv")

Options = ["Sentiment Analysis", "Data Visualisation","Intelligence"]
bar = st.sidebar.selectbox("Navigation", Options)
if bar == "Sentiment Analysis" :
    upload_file = st.file_uploader('Upload a file')
    if st.button("Print dataset"):
        st.write(data2.head(15))
    if st.button("Number of rows & columns"):
        print(st.write(data2.shape))
    if st.button("Attributes names"):
        st.write(data2.columns)
    if st.button("Print clean dataset"):
        st.write(data1.head(10))   
        
    st.write("Original & clean twitter comments") 
    if st.button("Print original tweets"):
        st.write(data2["text"].head(10))
    if st.button("Print clean tweets"):
        st.write(data1["text"].head(10))
        

    st.write('Upload dataset or text to clean and analyse')
    with st.expander('Analyze text'):
        text = st.text_input('Input text here: ')
        if text:
            blob = TextBlob(text)
            st.write('Polarity: ', round(blob.sentiment.polarity,2))
            st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


        pre = st.text_input('Clean text: ')
        if pre:
            st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                     stopwords=True ,lowercase=True ,numbers=True , punct=True))

    with st.expander('Analyze CSV'):
        upl = st.file_uploader('Upload file')

        def score(x):
            blob1 = TextBlob(x)
            return blob1.sentiment.polarity

    #
        def analyze(x):
            if x >= 0.5:
                return 'Positive'
            elif x <= -0.5:
                return 'Negative'
            else:
                return 'Neutral'

    #
        if upl:
            df = pd.read_excel(upl)
            del df['Unnamed: 0']
            df['score'] = df['tweets'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
            st.write(df.head(10))

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )


      
    
if bar =="Data Visualisation":
    st.text("Twitter Data")
 
 #Sentiments bar graph
    
    if st.button("Sentiment counts "):
        fig=plt.figure(figsize = (10,5))
        plt.title("Sentiment counts Over-View")
        sns.countplot(x = 'sentiment', data = data1)
        st.write(fig)

#Sentiment pie chart

    if st.button("Sentiments pie chart"):
        fig = plt.figure(figsize = (7,7))
        colors = ("yellowgreen", "gold", "red")
        wp = {'linewidth':2, "edgecolor" : "black"}
        tags = data1['sentiment'].value_counts()
        explode = (0.1, 0.1, 0.1)
        tags.plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, colors = colors, startangle=90, wedgeprops = wp, explode = explode, label = '')
        plt.title(' Percentage distribution of sentiments')
        st.write(fig)
        
# word cloud
    if st.button("Word cloud"):
        
        df1 = pd.read_excel("clean_protest_data.xlsx")
        #df1 = pd.read_excel("Book1.xlsx")

        df1.drop_duplicates(subset = "text", keep = "first", inplace = True)

        df1['text'] = df1['text'].str.replace(r"RT", " ")

        

        def data_processing(text):
            text = text.lower()
            text = re.sub(r"https\S+\www\S+",'', text, flags = re.MULTILINE)
            text = re.sub(r'@\S+','', str(text))
            text = re.sub(r'[^\w\s]','', text)
            text_tokens = word_tokenize(text)
            filtered_text = [w for w in text_tokens if not w in stop_words]
            return " ".join(filtered_text)

        df1.text = df1['text'].apply(data_processing)

        #stemming
        import nltk
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()

        df1['text'] = df1['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))


        from nltk.stem.wordnet import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        df1['text'] = df1['text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
        df1['text'].head()

        def polarity(text):
            return TextBlob(text).sentiment.polarity

        df1['polarity'] = df1['text'].apply(polarity)

        def sentiment(label):
            if label < 0:
                return "negative"
            elif label == 0:
                return "neutral"
            elif label > 0:
                return "positive"

        df1['sentiment'] = df1['polarity'].apply(sentiment)

        # wordcloud library
        from wordcloud import WordCloud

        # joing the different text together
        long_string = ','.join(list(df1['text'].values))
        
        
        # Create some sample text
        text = long_string

        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()
        
#Sentiments over time
    st.write("Sentiments over time")
    if st.button("Show sentiments over time plots"):
        # Create data
        x = ["1","20","21","22","23","24","25","26","27","28","30"]
        pos=[520,405,600,469,473,499,571,625,529,497,502]
        neu=[1945,1203,1267,900,841,916,1403,1920,1502,1030,1005]
        nev=[320,500,369,201,276,301,401,550,561,461,327]
        
        plt.plot(x,pos)
        plt.plot(x,nev)
        plt.plot(x,neu)
        plt.ylabel("number of tweets about the Currency")
        plt.xlabel("date(April-2023)")
        plt.title("Sentiments over time")
        plt.legend(labels=['Bitcoin','Arbitrum','Shiba Inu'])

        st.pyplot() 
        
if bar == "Intelligence" :
    
    st.image('download.png')
    pos=[520,405,600,469,473,499,571,625,529,497,502]
    neu=[1945,1203,1267,900,841,916,1403,1920,1502,1030,1005]
    nev=[320,500,369,201,276,301,401,550,561,461,327]
    red=[.4,.4,.4,.4,.4,.4,.4,.4,.4,.4,.4]
    x = ["19","20","21","22","23","24","25","26","27","28","29"]
    y = [(nev[0]/(nev[0]+pos[0]+neu[0])), (nev[1]/(nev[1]+pos[1]+neu[1])), (nev[2]/(nev[2]+pos[2]+neu[2])), (nev[3]/(nev[3]+pos[3]+neu[3])),
        (nev[4]/(nev[4]+pos[4]+neu[4])), (nev[5]/(nev[5]+pos[5]+neu[5])), (nev[6]/(nev[6]+pos[6]+neu[6])), (nev[7]/(nev[7]+pos[7]+neu[7])), 
        (nev[8]/(nev[8]+pos[8]+neu[8])), (nev[9]/(nev[9]+pos[9]+neu[9])), (nev[10]/(nev[10]+pos[10]+neu[10]))]
    plt.title("Fraud Detection November 2023 Safe Zone:0-0.40  Danger Zone:0.41-100")
    plt.ylabel("Fraud Detection Rate  ")
    plt.xlabel("Date(November)")
    plt.plot(x,y)
    plt.plot(x,red)
    st.pyplot()
    
    st.image('download.png')
   


    
    st.write("Ask Our Bot for More Information")
    
