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
# * For each group, there are 4 dataframes, where each row correspnds to a subject (s):
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
# All other dataframes have consistent row names of the form Subj_1, Subj_2, Subj_3 etc.

# In[4]:


for key, value in data.items():
    if not key[0:5] == 'index':
        continue
    data[key] = value.drop(columns=['Subj'])
    data[key].index = ['Subj_'+str(i) for i in range(1,value.shape[0]+1)]
data['index_150'].head()


#  

# **Verify table 1 (include link  to it).** <br> {cite:t}`Steingroever_Fridberg_Horstmann_Kjome_Kumari_Lane_Maia_McClelland_Pachur_Premkumar` speculates that the sample size may be less than 617 due to "missing data for one participant in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010`, and for two participants in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`". According to table 1 (link again), there should be 19 participants in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010` and 153 in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`.

# In[5]:


print("Total number of subjects:", data['choice_95'].shape[0] + data['choice_100'].shape[0] + data['choice_150'].shape[0])


# Appears in order, let's take a closer look to be sure.

# In[6]:


print("Subjects in Kjome study:", len(data['index_100'][data['index_100']['Study'] == 'Kjome']))
print("Subjects in Wood study:", len(data['index_100'][data['index_100']['Study'] == 'Wood']))


# Confirms correct number of subjects reported.

#  

# **Check for nulls/ unexpected entries**

# Sanity checking the dataframes for unusual entries, such as null values and unexpected data types. Additionally, confirming the dataframes are structured as expected, e.g. checking that all entries in lo_t are negative integers and that 1, 2, 3 and 4 are the only entries in choice_t.

# In[7]:


for key, value in data.items():
    try:
        uniq_entries = ', '.join([("{:.2f} ({:.2f}%)".format(entry, count*100)) for entry, count in value.stack().value_counts(normalize=True).sort_index().iteritems()])
    except:
        uniq_entries = ', '.join([("{:}".format(entry)) for entry, count in value.stack().value_counts().sort_index().iteritems()])

    print("\033[1mUnique entries (and their frequency) in {:}:\033[0m \n{:}".format(key, uniq_entries))


# No unexpected entries.

#  

# **Rename the columnns of choice_t, wi_t & lo_t for confirmity.**

# In[8]:


for key,value in data.items():
    if key[0:5] == 'index':
        continue
    data[key].columns = ['Trial_'+str(i) for i in range(1,value.shape[1]+1)]

data['choice_95'].head()


#  

# **Make dataframe for net outcome of each trial.**

# In[9]:


net_95 = data['wi_95'] + data['lo_95']
net_100 = data['wi_100'] + data['lo_100']
net_150 = data['wi_150'] + data['lo_150']
net_100.head()


# **Make dataframe for net cumulative outcome of each trial.**

# In[10]:


cum_out_95 = net_95.cumsum(axis=1)
cum_out_100 = net_100.cumsum(axis=1)
cum_out_150 = net_150.cumsum(axis=1)
cum_out_150.head()


# **Verify the good & bad decks.**

# As explained in the Introduction (add link), there are 2 advantageous decks and 2 disadvantageous decks. Let's verify that C and D are the good decks by checking the net outcome of each deck.

# In[11]:


print("{:} \t| {:} | {:}".format('choice_95 net outcome', 'choice_100 net outcome', 'choice_150 net outcome'))
print('-' * 73)

decks = ['A', 'B', 'C', 'D']
for i in range (1,5):
    deck = decks[i-1]
    out_95 = net_95[data['choice_95'].isin([i])].fillna(0).values.sum()
    out_100 = net_100[data['choice_100'].isin([i])].fillna(0).values.sum()
    out_150 = net_150[data['choice_150'].isin([i])].fillna(0).values.sum()
        
    print("deck {:}: {:10.2f} \t| deck {:}: {:10.2f} \t | deck {:}: {:10.2f}".format(deck, out_95, deck, out_100, deck, out_150))


# Decks A and B have negative net outcomes, while decks C and D have positive net outcome. Thus, confiriming that C and D are the good decks.

# ## Data Exploration

# **How many subjects made a profit?**

# In[12]:


profiters_95 = len(cum_out_95.loc[cum_out_95.Trial_95 > 0])
print('95 Trial Variation: ', profiters_95, 'subjects (or',round(profiters_95 / cum_out_95.shape[0] * 100,2), '%)')
profiters_100 = len(cum_out_100.loc[cum_out_100.Trial_100 > 0])
print('100 Trial Variation:', profiters_100, 'subjects (or',round(profiters_100 / cum_out_100.shape[0] * 100,2), '%)')
profiters_150 = len(cum_out_150.loc[cum_out_150.Trial_150 > 0])
print('150 Trial Variation:', profiters_150, 'subjects (or',round(profiters_150 / cum_out_150.shape[0] * 100,2), '%)')


# A higher percentange of subjects who completed 150 trials made a profit, than those who completed either 95 or 100 trials. The most obvious explanation for this, is that these subjects are afforded more trials to recognize the good decks and bad decks, and adjust their choices accordingly. Of course, there are other factors that could be responsible for this, such as the demographics of subjects or the environment of the study.

# **How long does it take, on average, to identify the good decks?**

# Produce a graph that shows the frequency of the good and bad decks over the course of the game. To do this, compute the frequency for a rolling group of 10 trials, i.e. calculate the frequency the of the subset of the 10 previous choices.

# In[13]:


grouped_choices = {}
for key, value in data.items():
    if not key[0:6] == 'choice':
        continue
    for i in range(1,5):
        grouped_choices[str('deck' + str(i) + '_' + key)] = value.T.rolling(10).apply(lambda x: x.isin([i]).fillna(0).values.sum()).astype('Int64').T
grouped_choices['deck4_choice_150'].head()


# In[14]:


grouped_keys = list(grouped_choices.keys())
for i in range(0,3):
    ind = i * 4
    df_range = list(range(1, grouped_choices[grouped_keys[ind]].shape[1] + 1))
    
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(df_range,(grouped_choices[grouped_keys[ind + 2]].fillna(0).mean() + grouped_choices[grouped_keys[ind + 3]].fillna(0).mean()) / 10 * 100, color = 'limegreen', label = 'good decks')
    plt.plot(df_range,(grouped_choices[grouped_keys[ind]].fillna(0).mean() + grouped_choices[grouped_keys[ind + 1]].fillna(0).mean()) / 10 * 100, color = 'crimson', label = 'bad decks')
    plt.xlabel('trial number', fontsize = 14)
    plt.ylabel('frequency [%]', fontsize = 14)
    fig.suptitle(grouped_keys[ind][-3:].strip('_') + ' Trial Variation - Avg Deck Selection', fontsize = 14)
    plt.xlim(10, df_range[-1])
    plt.ylim(0, 100)
    plt.legend()
    ax.xaxis.set_ticks(np.arange(10, df_range[-1] + 1, 20))
    plt.grid()
    plt.show()


# In the above graphs, the bad decks (deck A and B) are the most popular at the beginning. However, subjects appear to begin to recognize the bad decks (decks C and D) after approximately 13 trials, as the frequency of bad decks declines. For both, the 95 and 100 Trial Variations, the good decks overtake the bad decks after around 30 trials. On average, it appears to take the subjects of the 150 Trial Variation longer to recognize and tend towards the good decks, with the frequency of the good decks surpassing the bad deck after 57 trials. Interestingly, at the 90 trial mark, subjects in the 150 Trial Variation are selecting good decks with a similar or worse frequency to the subjects in the other 2 trial variations. Performance, for those in the 150 Trial Variation, continues to improve after 100 trials, with the gap between the green and red line increasing. This supports our earlier speculation of why a larger proportion of subjects made a profits in the 150 Trial Variation - they're more likely to identify the good and bad decks, as they do more repetitions of the task.

# In[15]:


# pick out a good and bad player and show their individual graphs


# In[16]:


# maybe contrast payoff scheme 3 to the others -> if so add payoff scheme to study data frame


# In[17]:


# prepare the data for clustering


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




