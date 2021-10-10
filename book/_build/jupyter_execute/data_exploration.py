#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# ## Import Libraries & Read In Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import itertools


# In[2]:


data = pyreadr.read_r('../data/IGTdata.rdata')


# **Overview of dataset**

# In[3]:


print("{:12}| {:57}| {:30}".format('DF NAME','COLUMN NAMES','ROW NAMES'))
print('-'*110)
for key, value in data.items():
    if value.shape[1] > 5:
        column_names = ''.join([', '.join(value.columns[:3]), ',...,', ', '.join(value.columns[-2:])])
    else:
        column_names = ', '.join(value.columns)
    row_names = [str(v) for v in list(value.index.values)]
    row_names = ''.join([', '.join(row_names[:2]), ',...,',', '.join(row_names[-2:])])
    print("{:12}| {:57}| {:30}".format(key,column_names,row_names))


# * The studies are grouped by the number of trials (t) completed; 95, 100 or 150. 
# * For each group, there are 4 data frames, where each row correspnds to a subject (s):
#   * choice_t : 
#     * Entries are either 1, 2, 3 or 4 which correspond to deck A, B, C and D, respectively.
#     * Dimensionality is s x t. 
#     * The entry of the second row and third column indicates the choice made by the second subject on the third trial.
#   * wi_t :
#     * Contains the win achieved as a result of each choice.
#     * Dimensionality is s x t. 
#     * The entry of the second row and third column corresponds to the reward received by the second subject on third trial.
#   * lo_t :
#     * Contains the loss incurred as a result of each choice.
#     * Dimensionality is s x t. 
#     * The entry of the second row and third column corresponds to the loss incurred by the second subject on third trial.
#   * index_t :
#     * Entries are the name of the first author of the study that reports the data name of the first author of the study that reports the data of the corresponding participant.
#     * Dimensionality is s x 2. 
#     * The entry of the second row indicates which study the second subject participated in.
#     

# ## Data Cleaning & Validation

# **Update index_t row names for confirmity.**<br>
# All other data frames have consistent row names of the form Subj_1, Subj_2, Subj_3 etc.

# In[4]:


for key, value in data.items():
        if not key[0:5] == 'index':
            continue
        data[key] = value.drop(columns=['Subj'])
        data[key].index = ['Subj_'+str(i) for i in range(1,value.shape[0]+1)]
data['index_150'].head()


#  

# **Verify table 1 (include link  to it).** <br> cite:t}`Steingroever_Fridberg_Horstmann_Kjome_Kumari_Lane_Maia_McClelland_Pachur_Premkumar` speculates that the sample size may be less than 617 due to "missing data for one participant in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010`, and for two participants in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`". According to table 1 (link again), there should be 19 participants in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010` and 153 in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`.

# In[5]:


print("Total number of subjects:", data['choice_95'].shape[0] + data['choice_100'].shape[0] + data['choice_150'].shape[0])


# Appears in order, let's take a closer look to be sure.

# In[6]:


print("Subjects in Kjome study:", len(data['index_100'][data['index_100']['Study'] == 'Kjome']))
print("Subjects in Wood study:", len(data['index_100'][data['index_100']['Study'] == 'Wood']))


# Confirms correct number of subjects reported.

#  

# **Check for nulls/ unexpected entries**

# Sanity checking the data frames for unusual entries, such as null values and unexpected data types. Additionally, confirming the data frames are structured as expected, e.g. checking that all entries in lo_t are negative integers and that 1, 2, 3 and 4 are the only entries in choice_t.

# In[7]:


for key, value in data.items():
    try:
        uniq_entries = ', '.join([("{:.2f} ({:.2f}%)".format(entry, count*100)) for entry, count in value.stack().value_counts(normalize=True).sort_index().iteritems()])
    except:
        uniq_entries = ', '.join([("{:}".format(entry)) for entry, count in value.stack().value_counts().sort_index().iteritems()])

    print("\033[1mUnique entries (and their frequency) in {:}:\033[0m \n{:}".format(key, uniq_entries))


# No unexpected entries.

# In[ ]:





# In[8]:


# data preparation section
# df for win - loss (outcome of each trial)
# cumulative profit/loss


# In[ ]:





# In[9]:


# maybe contrast payoff scheme 3 -> if so add payoff scheme to study data frame


# In[10]:


# verify good/bad decks
# when do players develop a tendency/aversion to certain decks -> do they correctly identify them

