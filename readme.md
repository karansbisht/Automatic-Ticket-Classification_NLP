# Case Study: Automatic Ticket Classification

## Problem Statement

For a financial company, customer complaints carry a lot of importance, as they are often an indicator of the shortcomings in their products and services. If these complaints are resolved efficiently in time, they can bring down customer dissatisfaction to a minimum and retain them with stronger loyalty. This also gives them an idea of how to continuously improve their services to attract more customers. 

 

These customer complaints are unstructured text data; so, traditionally, companies need to allocate the task of evaluating and assigning each ticket to the relevant department to multiple support employees. This becomes tedious as the company grows and has a large customer base.

 

In this case study, you will be working as an NLP engineer for a financial company that wants to automate its customer support tickets system. As a financial company, the firm has many products and services such as credit cards, banking and mortgages/loans. 


# Objective: 
Business goal
You need to build a model that is able to classify customer complaints based on the products/services. By doing so, you can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.

 

With the help of non-negative matrix factorization (NMF), an approach under topic modelling, you will detect patterns and recurring words present in each ticket. This can be then used to understand the important features for each cluster of categories. By segregating the clusters, you will be able to identify the topics of the customer complaints. 

 

You will be doing topic modelling on the .json data provided by the company. Since this data is not labelled, you need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:

Credit card / Prepaid card

Bank account services

Theft/Dispute reporting

Mortgages/loans

Others 

With the help of topic modelling, you will be able to map each ticket onto its respective department/category. You can then use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, you can classify any new customer complaint support ticket into its relevant department.


## Table of contents
Problem Statement
Pipelines that needs to be performed:
Importing the necessary libraries
Loading the data
Data preparation
Prepare the text for topic modeling
Exploratory data analysis to get familiar with the data.
Find the top 40 words by frequency among all the articles after processing the text.
Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text.
The personal details of customer has been masked in the dataset with xxxx. Let's remove the masked text as this will be of no use for our analysis
Feature Extraction
Create a document term matrix using fit_transform
Topic Modelling using NMF
Manual Topic Modeling
After evaluating the mapping, if the topics assigned are correct then assign these names to the relevant topic:
Supervised model to predict any new complaints to the relevant Topics.
Apply the supervised models on the training data created. In this process, you have to do the following:
Clearly Logistic Regression is performing better
Infering the best model
Conclusion



## library
# Import Basic libariries
import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

nlp = en_core_web_sm.load(disable=['parser','ner'])

from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import swifter
from sklearn.feature_extraction.text import TfidfTransformer

%matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
nltk.download('wordnet')

--- 

# Result


Clearly Logistic Regression is performing better

# conclusion :

As expected 5 topics were indetified namely:

Account Services

Others

Mortgage/Loan

Credit card or prepaid card

Theft/Dispute Reporting

Tried 4 models on the data with accuracies as follows:

| Model | Accuracy | | ----------- | ----------- | | Logistic Regression | 0.95 | | Decision Tree | 0.77 | | Random Forest | 0.74 | | Naive Bayes | 0.36 |

Logistic Regression has highest accuracy of 0.95, Hence is a good fit for this particular case study.

# Submitted by 
karan singh bisht
karan pandey