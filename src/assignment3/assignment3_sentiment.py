#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3 - Sentiment Analysis
# __DESCRIPTION__
# 
# Dictionary-based sentiment analysis with Python
# <br>
# 
# 
# Download the following CSV file from Kaggle:
# <br>
# 
# 
# https://www.kaggle.com/therohk/million-headlines
# <br>
# 
# 
# This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).
# <br>
# 
# 
# Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python. <br>
# Create and save a plot of sentiment over time with a 1-week rolling average <br>
# Create and save a plot of sentiment over time with a 1-month rolling average <br>
# Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y  axes; and a legend for the plot <br>
# Write a short summary (no more than a paragraph) describing what the two plots show. 
# 
# <br> 
# You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?
# 
# <br>
# HINT: You'll probably want to calculate an average score for each day first, before calculating the rolling averages for weeks and months.
# 
# <br>
# 
# __General instructions__
# 
# <br>
# For this assignment, you should upload a standalone .py script which can be executed from the command line or a Jupyter Notebook
# Save your script as sentiment.py or sentiment.ipynb
# Make sure to include a requirements.txt file and details about where to find the data
# You can either upload the scripts here or push to GitHub and include a link - or both!
# Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line
# 
# <br>
# 
# __Purpose:__
# 
# <br>
# This assignment is designed to test that you have a understanding of:
# <br>
# how to perform dictionary-based sentiment analysis in Python; <br>
# how to effectively use pandas and spaCy in a simple NLP workflow; <br>
# how to present results visually, working with datetime formats to show trends over time <br>

# In[67]:


import os
import spacy
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob
#initialise spacy
nlp = spacy.load("en_core_web_sm")


# In[68]:


news_data = os.path.join("..","data", "abcnews-date-text.csv") #First we load in the dataset


# In[72]:


data = pd.read_csv(news_data) #defining it and reading the csv


# In[74]:


#Adding spaCyTextBlob to spaCy pipeline: 
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


# In[75]:


#This is primary code for creating the sentiment scores, but this proces takes over 15 mins so for it to be easier I will include
# the csv file this code creates instead, which can be loaded below this code. 
"""# Creating a container for the sentiment scores:
output = []

#The loop runs through the headlines in the csv-file.
for doc in nlp.pipe(data["headline_text"], batch_size = 5000,
                   disable=["tagger", "parser", "ner"]): #Disabeling to make the code run faster, since there is a lot of data and the processing time is long.
    output.append(doc._.sentiment.polarity) #appending the scores to our output container.

# Creating a new dataframe
data_df = pd.DataFrame(data)

# Appending a new series/column to the new df, with the scores
data_df["scores"] = output"""


# In[83]:


data_df.to_csv("data_scores") #Here we wrote the dataframe to the csv


# In[97]:


data_df_path = os.path.join("data_scores") #Load this instead of running the code above as it takes less time
data_df = pd.read_csv(data_df_path) # with this the data 


# In[96]:


data_df #Here we should be able to see the dataframe. You want to make sure you have the path for the data_scores as the same 
# as the actual script or change the directory above. 


# In[25]:


daily_avg_sentiment = data_df.groupby("publish_date").mean() #Here we group the data_df by the date of publish. 


# In[31]:


daily_avg_sentiment #here we check if we have the average sentiments scores for the individual dates


# In[43]:


#To make the figures we create a mean_scores list in which we loop the scores and append on index
mean_scores = []
for i in daily_avg_sentiment ["scores"]:
    mean_scores.append(i)
    
    


# In[64]:


#First we create the weekly averages with the pandas series function
smoothed_sentiment_weeks = pd.Series(mean_scores).rolling(7).mean()

    #Create figure
fig = plt.figure(figsize = (15,10)) 
    #Create a grid for readability
plt.grid()

    #Plot average sentiment for 7 days rolling
plt.plot(smoothed_sentiment_weeks)

    #Create title, labels and legend
plt.title("Headline sentiment since 2003",fontsize= 20)
plt.xlabel("Years since 2003", fontsize= 15, labelpad=10)
plt.ylabel("Sentiment Score", fontsize= 15)
plt.legend(["Weekly rolling average"],
               loc='upper right',
               fontsize= 12)

    #Set x ticks to be in years rather than days, this meaning the plot shows years (0 being 2003 and 18 being 2021) 
plt.xticks(np.arange(0, len(mean_scores)+1,365), range(0,18))


# ## What do we see? 
# From what I understand there is a radical negative sentiment in the middle of year 3 and 4, which would be 2006 and 2007, which could be around the financial crisis meaning news were more negative. Opposite can be said just before year 12 (2015) where there is a spike in positve sentiment.  

# In[66]:


#Same process as above for the monthly but 30 as the rolling key instead of 7
smoothed_sentiment_months = pd.Series(mean_scores).rolling(30).mean()

    #Create figure
fig = plt.figure(figsize = (15,10)) 
    #Create a grid for readability
plt.grid()
    #Plot average sentiment for 30 days rolling
plt.plot(smoothed_sentiment_months)
    #Create title, labels and legend
plt.title("Headline sentiment since 2003",fontsize= 20)
plt.xlabel("Years since 2003", fontsize= 15, labelpad=10)
plt.ylabel("Sentiment Score", fontsize= 15)
plt.legend(["Monthly rolling average"],
               loc='upper right',
               fontsize= 12)

    #Set x ticks to be in years again. This does however create a visual of the list, which is unnecessary but I don't know how 
    # remove this without also removing the correct years on the plot
plt.xticks(np.arange(0, len(mean_scores)+1,365), range(0,18))


# ## What do we see? 
# Similar patterns to above but it is worth noting that the sentiment never goes to a negative point but instead only touches a neutral in between year 3 and 4. It also displays a positvive spike at year 12 (2015).   
