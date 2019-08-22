# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:34:48 2019

#### Matching between bills and Faculty ######

@author: Alvaro Gonzalez Magnolfi
email: alvarogo@andrew.cmu.edu
"""
#%%
###Importing neccesary libraries 
from pathlib import Path
import glob
import os
import pandas as pd
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#%% 
### Setting path for bills or text. 
### files has to be .txt.
### Setting the path for CMU elements keywords. 
### file has to be an excel file
data_folder = Path("C:/Users/algon/OneDrive/Desktop/Dean office/Brook")
bloc = pd.read_excel(r'C:\Users\algon\OneDrive\Desktop\Dean office\Clust\bloc.xlsx', sheet_name = 'Data')
os.chdir("C:/Users/algon/OneDrive/Desktop/Dean office/Brook")
#%%
def matching(data_folder,bloc):
        def get_bills_text(data_folder): ### Get text and label according the filename
            bills = []
            for filename in glob.glob(os.path.join(data_folder,'*.txt')):
                with open(filename, 'r', encoding = "ISO-8859-1") as infile:
                    bills.append(infile.read())
                
            names = []
            for filename in glob.glob(os.path.join(data_folder,'*.txt')):
                names.append(filename)
            filename_split = []
            for i in names:
                t = i.split('\\')
                filename_split.append(t)
            labels = []
            for i in filename_split:
                a = i[-1].split('.txt')
                b = a[0]
                labels.append(b)
            return(labels, bills)    
           
        def Elem_keywords(bloc): ### Getting faculty information, assign to a dict and keywords
            faculty = set()
            for i in bloc['Faculty Name']:
                faculty.add(i)
            faculty = list(faculty)
            faculty.sort()
            dict_fact = { i : faculty[i] for i in range(0, len(faculty) ) }
            bloc2 = bloc.groupby('Faculty Name')['Keywords'].apply(lambda x: "{%s}" % ', '.join(x))
            bloc_kwds = []
            for i in range(0,(len(bloc2))):
                bloc_kwds.append(bloc2[i])
            return (dict_fact, bloc_kwds)
        
        def is_letter_only(word): ### Function to clean names and only includer words (erase numbers)
            return word.isalpha()
        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()
        
        def clean_text(docs):   ### Setting clean text function ###       
            bills_cleaned = []
            for doc in docs:
                doc = doc.lower()
                bill_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word) and word not in all_names) ### Lemmatize, only include letters except names
                bills_cleaned.append(bill_cleaned)
            return bills_cleaned
        
        x = get_bills_text(data_folder) ### Run function
        z = Elem_keywords(bloc)         ### Run function  
        
        tfidf_vector = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.7, min_df =1)     # frequency-inverse document frequency (tf-idf) function
        data_cleaned_tfid_ma = tfidf_vector.fit_transform(clean_text(x[1]))                               # fitting tf-idf to text files
        prof_cleaned_tfid_ma = tfidf_vector.transform(clean_text(z[1]))                                  # transform keywords to a tf-idf matrix (set before)
        
        aux1=[]                                                                                          #Matching function
        for o in range(0,len(x[0])):
            aux2=[]
            for j in range(0,len(z[0])):
                aux2.append(np.linalg.norm(prof_cleaned_tfid_ma[j].todense()-data_cleaned_tfid_ma[o].todense())) #Calculating the eucliedean distance between each bill and faculty keywords
            aux1.append(aux2)
            
        aux3=[]
        for i in aux1:
            dict1 = { t : i[t] for t in range(0, len(i) ) }    
            dict1_order = [sorted(dict1, key=dict1.get)]
            aux3.append(dict1_order)
            
        dict_rk_CMUE =dict(zip(x[0],aux3))
        aux4 = []
        for i in x[0]:                                                                              # Ranked faculty using the eucliedean distance
            t = dict_rk_CMUE.get(i)
            a=t[0][0:5]
            aux4.append(a)
        aux5= []
        for i in aux4:
            aux6= []
            for j in i:
                aux6.append(z[0].get(j))
            aux5.append(aux6)
        dict_CMU_score = dict(zip(x[0],aux5))
        df2 = pd.DataFrame.from_dict(dict_CMU_score, orient='index')
        df2 = df2.rename(columns = {0: "Rank1",1: "Rank2",2: "Rank3",3: "Rank4",4: "Rank5"})
        return df2.to_excel("output.xlsx")                                                        # Return a excel file that for a given label, return the first 5 faculty members closer    
#%%
matching(data_folder,bloc)
