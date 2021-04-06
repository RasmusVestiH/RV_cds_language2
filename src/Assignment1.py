#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1

# Basic scripting with Python
# 
# Using the corpus called 100-english-novels found on the cds-language GitHub repo, write a Python programme which does the following:
# 
# Calculate the total word count for each novel
# Calculate the total number of unique words for each novel
# Save result as a single file consisting of three columns: filename, total_words, unique_words
# 
# 
# General instructions
# 
# For this exercise, you can upload either a standalone script OR a Jupyter Notebook
# Save your script as word_counts.py OR word_counts.ipynb
# You can either upload the script/notebook here or push to GitHub and include a link - or both!
# Your code should be clearly documented in a way that allows others to easily follow the structure of your script.
# Similarly, remember to use descriptive variable names! A name like word_count is more readable than wcnt.
# 
# 
# Purpose
# 
# This assignment is designed to test that you have a understanding of:
# 
# how to structure, document, and share a Python script;
# how to effectively make use of native Python data structures, functions, and flow control;
# how to load, save, and process text files.
# 

# ## Script

# In[ ]:


# To start of we need to import the 100-english-novels in the data folder of the cds-language github repo.


# In[1]:


import os


# In[2]:


from pathlib import Path


# In[3]:


data_path = os.path.join("..", "data", "100_english_novels", "corpus") #path.join lets us set the path for our directory that is being used


# In[4]:


data_path


# In[5]:


for filename in Path (data_path).glob("*.txt"): #Here we have a loop that checks if the textfiles are present and correct.
    print (filename)


# In[24]:


#We create a table for the data on filename, word counts and unique words
table = [f"[filename], [word_count], [unique_words]"]


# In[35]:


for filename in Path (data_path).glob("*.txt"):
    with open(filename, "r", encoding="utf-8") as file: 
        loaded_text = file.read()
        word_count = loaded_text.split()
        unique_words = set(word_count)
     
        table.append(f"{filename}, {len(word_count)}, {len(unique_words)}")
        


# In[36]:


# by appending to the table we get the dataset of textfiles, the word count and the unique words in this. 


# In[37]:


table


# In[44]:


#In this code I tried to save it as a new file in the folder of 100 english novels, but could not make it work. 
        
outpath = os.path.join("..", "data", "100_english_novels", "wordcounts.txt")
    with open(outpath, "w", encoding = "utf-8") as file:
        file.write(table)

                

