#!/usr/bin/env python
# coding: utf-8

# ## String Processing

# String processing with Python
# 
# Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.
# 
# 
# 
# The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv
# 
# These parameters can be defined in the script itself
# 
# Find out how often each word collocates with the target across the corpus
# 
# Use this to calculate mutual information between the target word and all collocates across the corpus
# 
# Save result as a single file consisting of four columns: collocate, raw_frequency, MI
# 
# 
# BONUS CHALLENGE: Use argparse to take inputs from the command line as parameters
# 
# 
# General instructions
# 
# For this assignment, you should upload a standalone .py script which can be executed from the command line.
# Save your script as collocation.py
# Make sure to include a requirements.txt file and your data
# You can either upload the scripts here or push to GitHub and include a link - or both!
# Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

# __Disclaimer:__ <br>
# I struggled with the assignment a lot! That also means that I have been seeking inspiration and understanding in my peers work on github. Despite this I still haven't been able to understand how to get the function for computing the data working, in a sense as well as how to create a .csv after running the code.

# In[166]:


import os
import re #regex
import string #regex
import math
import numpy as np
import pandas as pd
import csv
from pathlib import Path


# In[167]:


def main():
    path = os.path.join((".."), "data", "100_english_novels", "corpus") 
    output = os.path.join("..", "data", "output.csv") #making the output file
    #running the collocates function on the specified path using the keyword
    collocates_df = collocates(path, "so", 5)
    #saving the collocates data frame as csv 
    collocates_df.to_csv("collocates.csv", index = False)
    main()


def tokenize(input_string):
    #split at all characters except for letters (both lowercase and uppercase) and apostrophes with regex
    tokenizer = re.compile(r"[^a-zA-Z']+") 
    # Tokenize
    token_list = tokenizer.split(input_string) # makes a token list by splitting the input string
    # Return list of tokens
    return token_list

def collocates (path, keyword, window_size): 
    data = pd.DataFrame(columns=["collocate", "raw_frequency", "MI", "keyword"])
    collocates_list = [] #empty lists for collocates
    token_list_all = [] #same for tokens
    u = 0 #This is the count for keywords
    
    for filename in Path(path).glob("*.txt"): 
        with open (filename, "r", encoding = "utf-8") as file:
            text = file.read()
            
            token_list = tokenize(text.lower())
            #appending the temporary token_list to an "all" list
            token_list_all.extend(token_list)
            #finding the index for all occurrences of the keyword and using enumerate to create indices
            indices = [index for index, x in enumerate(token_list) if x == keyword]
            #adding the number of keywords to the keyword count (u)
            u = u + len(indices)
           
            #looping over occurences of the keyword
            for index in indices:
                #we need a window start and end for the words surrounding the keyword: 
                window_start = max(0, index - window_size) # if index - window_size is a negative value, it will be 0.
                window_end = index + window_size
                #using token_list to see the tokens within window space of keyword
                keyword_string = token_list[window_start : window_end + 1]
                #extending these to a list
                collocates_list.extend(keyword_string)
                #to make sure the keyword isn't in the list we remove it
                collocates_list.remove(keyword)
        
        
        
#calculating collocate frequency and mutual information (MI)    
        unique_collocates = set(collocates_list)
    for collocate in unique_collocates:
        #calculating v - the occurrences of collocate with the keyword
        v = token_list_all.count(collocate)
        #calculating O11 - all occurrences of collocate as a collocate
        O11 = collocates_list.count(collocate)
        #calculating O12 - occurrences of keyword without collocate
        O12 = u - O11
        #calculating O21 - occurrences of collocate without keyword
        O21 = v - O11
        #calculating R1 - observed frequency of O11 and O12
        R1 = O11 + O12
        #calculating C1 - observed frequency of O11 and O21
        C1 = O11 + O21
        #calculating N - number of words in corpus 
        N = len(token_list_all)
        #calculating E11 - expected frequency of keyword and collocate
        E11 = R1*C1/N
        #calculating MI - mutual information
        MI = np.log(O11/E11)
        #appending data to data frame
        data = data.append({"keyword": keyword, 
                     "collocate": collocate, 
                     "raw_frequency": O11,
                     "MI": MI}, ignore_index = True)
    #sorting the MI values in ascending order 
    data = data.sort_values("MI", ascending = False)
    # Returning data frame
    
    # Writing the file
    
    with open(output, "w", encoding = "utf-8") as file:
        file.write(collocates_df.to_csv("collocates.csv", index = False))
    return data
         
  

    if __name__=="__main__":
        main()

