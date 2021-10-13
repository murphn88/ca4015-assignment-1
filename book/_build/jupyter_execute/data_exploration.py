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
from sklearn.preprocessing import StandardScaler


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

# **Add payoff scheme to index_t**

# In[5]:


study_payscheme = {'Fridberg':1, 'Horstmann':2, 'Kjome':3, 'Maia':1, 'Premkumar':3, 'SteingroverInPrep':2, 'Wood':3, 'Worthy':1, 'Steingroever2011':2, 'Wetzels':2}
for key, value in data.items():
    if not key[0:5] == 'index':
        continue
    payscheme = [study_payscheme[val[0]] for val in value.values]
    data[key]['PayScheme'] = payscheme
data['index_150'].head()


#  

# **Verify Sample Size.** <br> {cite:t}`Steingroever_Fridberg_Horstmann_Kjome_Kumari_Lane_Maia_McClelland_Pachur_Premkumar` speculates that the sample size may be less than 617 due to "missing data for one participant in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010`, and for two participants in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`". According to {cite:t}`Steingroever_Fridberg_Horstmann_Kjome_Kumari_Lane_Maia_McClelland_Pachur_Premkumar`, there should be 19 participants in {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010` and 153 in {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005`.

# In[6]:


print("Total number of subjects:", data['choice_95'].shape[0] + data['choice_100'].shape[0] + data['choice_150'].shape[0])


# Appears in order, let's take a closer look to be sure.

# In[7]:


print("Subjects in Kjome study:", len(data['index_100'][data['index_100']['Study'] == 'Kjome']))
print("Subjects in Wood study:", len(data['index_100'][data['index_100']['Study'] == 'Wood']))


# Confirms correct number of subjects reported.

#  

# **Check for nulls/ unexpected entries**

# Sanity checking the dataframes for unusual entries, such as null values and unexpected data types. Additionally, confirming the dataframes are structured as expected, e.g. checking that all entries in lo_t are negative integers and that 1, 2, 3 and 4 are the only entries in choice_t.

# In[8]:


for key, value in data.items():
    try:
        uniq_entries = ', '.join([("{:.2f} ({:.2f}%)".format(entry, count*100)) for entry, count in value.stack().value_counts(normalize=True).sort_index().iteritems()])
        print("\033[1mUnique entries (and their frequency) in {:}:\033[0m \n{:}".format(key, uniq_entries))        
    except:
        uniq_entries = ', '.join([("{:}".format(entry)) for entry, count in value.stack().apply(str).value_counts(normalize=True).sort_index().iteritems()])
        print("\033[1mUnique entries in {:}:\033[0m \n{:}".format(key, uniq_entries))


# No unexpected entries.

#  

# **Rename the columnns of choice_t, wi_t & lo_t for confirmity.**

# In[9]:


for key,value in data.items():
    if key[0:5] == 'index':
        continue
    data[key].columns = ['Trial_'+str(i) for i in range(1,value.shape[1]+1)]

data['choice_95'].head()


#  

# **Make dataframe for net outcome of each trial.**

# In[10]:


net_95 = data['wi_95'] + data['lo_95']
net_100 = data['wi_100'] + data['lo_100']
net_150 = data['wi_150'] + data['lo_150']
net_100.head()


# **Make dataframe for net cumulative outcome of each trial.**

# In[11]:


cum_out_95 = net_95.cumsum(axis=1)
cum_out_100 = net_100.cumsum(axis=1)
cum_out_150 = net_150.cumsum(axis=1)
cum_out_150.head()


# **Verify the good & bad decks.**

# As explained in the Introduction (add link), there are 2 advantageous decks and 2 disadvantageous decks. Let's verify that C and D are the good decks by checking the net outcome of each deck.

# In[12]:


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

# **Check the average outcome for each deck**

# In[13]:


print("{:} | {:}  | {:}".format('choice_95 average outcome', 'choice_100 average outcome', 'choice_150 average outcome'))
print('-' * 85)

decks = ['A', 'B', 'C', 'D']
for i in range (1,5):
    deck = decks[i-1]
    avg_out_95 = net_95[data['choice_95'].isin([i])].fillna(0).values.sum() / data['choice_95'].stack().value_counts().sort_index()[i]
    avg_out_100 = net_100[data['choice_100'].isin([i])].fillna(0).values.sum() / data['choice_100'].stack().value_counts().sort_index()[i]
    avg_out_150 = net_150[data['choice_150'].isin([i])].fillna(0).values.sum() / data['choice_150'].stack().value_counts().sort_index()[i]

    print("deck {:}: {:10.2f} \t  | deck {:}: {:11.2f} \t| deck {:}: {:10.2f}".format(deck, avg_out_95, deck, avg_out_100, deck, avg_out_150))


# For the 95 and 100 Trial Varations it appears that, at least in some of the studies, the net outcome of the bad decks are not equal. Deck B appears to be the worst deck. The 150 Trial Variation, only includes studies with Payoff Scheme 2, where the net loss appears to consistent among the bad decks.

# ## Data Exploration

# **How many subjects made a profit?**

# In[14]:


profiters_95 = len(cum_out_95.loc[cum_out_95.Trial_95 > 0])
print('95 Trial Variation: ', profiters_95, 'subjects (or',round(profiters_95 / cum_out_95.shape[0] * 100,2), '%)')
profiters_100 = len(cum_out_100.loc[cum_out_100.Trial_100 > 0])
print('100 Trial Variation:', profiters_100, 'subjects (or',round(profiters_100 / cum_out_100.shape[0] * 100,2), '%)')
profiters_150 = len(cum_out_150.loc[cum_out_150.Trial_150 > 0])
print('150 Trial Variation:', profiters_150, 'subjects (or',round(profiters_150 / cum_out_150.shape[0] * 100,2), '%)')


# A higher percentange of subjects who completed 150 trials made a profit, than those who completed either 95 or 100 trials. The most obvious explanation for this, is that these subjects are afforded more trials to recognize the good decks and bad decks, and adjust their choices accordingly. Of course, there are other factors that could be responsible for this, such as the demographics of subjects or the environment of the study.

# **How long does it take, on average, to identify the good decks?**

# Produce a graph that shows the frequency of the good and bad decks over the course of the game. To do this, compute the frequency for a rolling group of 10 trials, i.e. calculate the frequency the of the subset of the 10 previous choices.

# In[15]:


grouped_choices = {}
for key, value in data.items():
    if not key[0:6] == 'choice':
        continue
    for i in range(1,5):
        grouped_choices[str('deck' + str(i) + '_' + key)] = value.T.rolling(10).apply(lambda x: x.isin([i]).fillna(0).values.sum()).astype('Int64').T
grouped_choices['deck4_choice_150'].head()


# In[16]:


grouped_keys = list(grouped_choices.keys())
for i in range(0,3):
    ind = i * 4
    df_range = list(range(1, grouped_choices[grouped_keys[ind]].shape[1] + 1))
    
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(df_range,(grouped_choices[grouped_keys[ind + 2]].fillna(0).mean() + grouped_choices[grouped_keys[ind + 3]].fillna(0).mean()) / 10 * 100, color = 'limegreen', linewidth=2.5, label = 'good decks')
    plt.plot(df_range,(grouped_choices[grouped_keys[ind]].fillna(0).mean() + grouped_choices[grouped_keys[ind + 1]].fillna(0).mean()) / 10 * 100, color = 'crimson', linewidth=2.5, label = 'bad decks')
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

# **Let's look at an individual plot for a good & bad subject.**

# In[17]:


good_player = cum_out_100['Trial_100'].idxmax()

fig = plt.figure()
ax = fig.gca()
plt.plot(list(range(1,101)),((grouped_choices['deck3_choice_100'].loc[good_player].fillna(0) + grouped_choices['deck4_choice_100'].loc[good_player].fillna(0))/ 10 * 100).tolist(), color = 'limegreen', label = 'good decks', linewidth=2.5)
plt.plot(list(range(1,101)),((grouped_choices['deck1_choice_100'].loc[good_player].fillna(0) + grouped_choices['deck2_choice_100'].loc[good_player].fillna(0))/ 10 * 100).tolist(), color = 'crimson', label = 'bad decks', linewidth=2.5)
plt.xlabel('trial number', fontsize = 14)
plt.ylabel('frequency [%]', fontsize = 14)
fig.suptitle('Good player - ' + str(good_player) + ', Net outcome of ' + str(cum_out_100.loc[[good_player]].Trial_100.values[0]), fontsize = 14)
plt.xlim(10, 100)
plt.ylim(0, 100)
plt.legend()
ax.xaxis.set_ticks(np.arange(10, 100, 20))
plt.grid()
plt.show()
    
bad_player = cum_out_100['Trial_100'].idxmin()

fig = plt.figure()
ax = fig.gca()
plt.plot(list(range(1,101)),((grouped_choices['deck3_choice_100'].loc[bad_player].fillna(0) + grouped_choices['deck4_choice_100'].loc[bad_player].fillna(0))/ 10 * 100).tolist(), color = 'limegreen', label = 'good decks', linewidth=2.5)
plt.plot(list(range(1,101)),((grouped_choices['deck1_choice_100'].loc[bad_player].fillna(0) + grouped_choices['deck2_choice_100'].loc[bad_player].fillna(0))/ 10 * 100).tolist(), color = 'crimson', label = 'bad decks', linewidth=2.5)
plt.xlabel('trial number', fontsize = 14)
plt.ylabel('frequency [%]', fontsize = 14)
fig.suptitle('Bad player - ' + str(bad_player) + ', Net outcome of ' + str(cum_out_100.loc[[bad_player]].Trial_100.values[0]), fontsize = 14)
plt.xlim(10, 100)
plt.ylim(0, 100)
plt.legend()
ax.xaxis.set_ticks(np.arange(10, 100, 20))
plt.grid()
plt.show()


# The good player clearly has a preference for the good decks, while the bad player mainly sticks to the bad decks.

# **Try to identify, when, if at all, a subject recognizes the good decks.**

# In[18]:


player = 'Subj_188'

fig = plt.figure()
ax = fig.gca()
plt.plot(list(range(1,101)),((grouped_choices['deck3_choice_100'].loc[player].fillna(0) + grouped_choices['deck4_choice_100'].loc[player].fillna(0))/ 10 * 100).tolist(), color = 'limegreen', label = 'good decks', linewidth=2.5)
plt.plot(list(range(1,101)),((grouped_choices['deck1_choice_100'].loc[player].fillna(0) + grouped_choices['deck2_choice_100'].loc[player].fillna(0))/ 10 * 100).tolist(), color = 'crimson', label = 'bad decks', linewidth=2.5)
plt.xlabel('trial number', fontsize = 14)
plt.ylabel('frequency [%]', fontsize = 14)
fig.suptitle('Player - ' + str(player) + ', Net outcome of ' + str(cum_out_100.loc[[player]].Trial_100.values[0]), fontsize = 14)
plt.xlim(10, 100)
plt.ylim(0, 100)
plt.legend()
ax.xaxis.set_ticks(np.arange(10, 100, 20))
plt.grid()
ax.add_patch(plt.Circle((45.5, 50), 3, color='black', fill=False, linewidth=3))
plt.show()


# The green line overtakes the red line, and stays on top, at the 46 trial mark. So, from the 35th trial onwards (remember we're working with rolling groups of 10), this subject chooses more form the good decks, than the bad decks. In fact, this subject develops a complete aversion after 35 trials.  
# Let's write a function to extract the circled point, the point where the green line overtakes the red line and remains on top. If the point doesn't exist (i.e., the green line isn't above the red line at the end), we'll choose the last point across all variations (i.e., the number of the last trial, which is 150).

# In[19]:


trial_variations = ['95', '100', '150']
aha_moments = []

for i in range(0,3):
    deck3 = grouped_choices[('deck3_choice_' + trial_variations[i])]
    deck4 = grouped_choices[('deck4_choice_' + trial_variations[i])]
    subj_aha_moments = []

    for subj in deck3.index.tolist():
        freq_good_decks = (deck3.loc[subj].fillna(0) + deck4.loc[subj].fillna(0)) * 10
        last_trial = freq_good_decks.index.tolist()[-1]
        aha_moment = 'trial_150'
        good_majority_bin = ((freq_good_decks > 50).to_frame()[subj].fillna(False).astype(int)).to_frame()
        for k, v in good_majority_bin[good_majority_bin[subj] == 1].groupby((good_majority_bin[subj] != 1).cumsum()):
            if v.index.tolist()[-1] == last_trial:
                aha_moment = v.index.tolist()[0]
        subj_aha_moments.append(int(aha_moment.split('_')[-1]))
    aha_moments.append(pd.DataFrame(subj_aha_moments, columns = ['AhaMoment'], index = deck3.index))


# **Check the function**

# In[20]:


player = 'Subj_89'

fig = plt.figure()
ax = fig.gca()
plt.plot(list(range(1,151)),((grouped_choices['deck3_choice_150'].loc[player].fillna(0) + grouped_choices['deck4_choice_150'].loc[player].fillna(0))/ 10 * 100).tolist(), color = 'limegreen', label = 'good decks', linewidth=2.5)
plt.plot(list(range(1,151)),((grouped_choices['deck1_choice_150'].loc[player].fillna(0) + grouped_choices['deck2_choice_150'].loc[player].fillna(0))/ 10 * 100).tolist(), color = 'crimson', label = 'bad decks', linewidth=2.5)
plt.xlabel('trial number', fontsize = 14)
plt.ylabel('frequency [%]', fontsize = 14)
fig.suptitle('Player - ' + str(player) + ', Net outcome of ' + str(cum_out_150.loc[[player]].Trial_150.values[0]), fontsize = 14)
plt.xlim(10, 150)
plt.ylim(0, 100)
plt.legend()
ax.xaxis.set_ticks(np.arange(10, 150, 20))
plt.grid()
ax.add_patch(plt.Circle((aha_moments[2].loc['Subj_89'].values[0], 50), 3, color='black', fill=False, linewidth=3, zorder=3))
plt.show()


# Works as expected.

# **Redefine Aha moment as the trial from which the subsequent majority of choices are good.**  

# In[21]:


aha_moments = [aha_moment - 9 for aha_moment in aha_moments]


# **Investigate correlation between Aha moment and profit/loss.**

# In[22]:


plt.plot(aha_moments[1],cum_out_100['Trial_100'],'o', color = 'lightgreen', label = '100 Trial Variation')
plt.plot(aha_moments[0],cum_out_95['Trial_95'],'o', color = 'coral', label = '95 Trial Variation')
plt.plot(aha_moments[2],cum_out_150['Trial_150'],'o', color = 'mediumpurple', label = '150 Trial Variation')
plt.legend()
plt.xlabel('Aha Moment', fontsize = 14)
plt.ylabel('Profit/ Loss', fontsize = 14)
plt.grid()
plt.show()


# In[23]:


print('Correlation Coefficient:', np.corrcoef(aha_moments[0].AhaMoment.tolist(), cum_out_95['Trial_95'].tolist())[0][1])
print('Correlation Coefficient:', np.corrcoef(aha_moments[1].AhaMoment.tolist(), cum_out_100['Trial_100'].tolist())[0][1])
print('Correlation Coefficient:', np.corrcoef(aha_moments[2].AhaMoment.tolist(), cum_out_150['Trial_150'].tolist())[0][1])


# A clear negative correlation can be seen in the above graph between the profit/ loss made by a subject and when their Aha moment is. Subjects who have earlier Aha mometnts, tend to have better results in the IGT. While the simple function we used to determine the Aha moment could probably benefit from increased sophistication and more complex logic, the function can be used as an indication of how well a subject will perform.

# ## Preprocessing for Clustering

# Let's combine all of the data needed for clustering into a single dataframe. We do not want to choose too many features, as the number of variables (dimensions) increases, the distance-based similarity measure, used to cluster, converges to a constant value. Thus, the higher the dimensionality, the more difficult it becomes to find strict differences between instances. To fully prepare the data, we will standardize it, to ensure all features are considered on an even playing field. 

# ```{admonition} Caution
# :class: warning
# We cannot use the k-means clustering algorithm on categorical variables, such as study or payoff scheme. Additionally, we cannot can simply convert these feautures into numerical features, as if we attempted to cluster on 'PayScheme', for example, k-means would assume that schemes 2 and 3 are more similar than schemes 1 and 3.
# ```

# **Feature Concatenation.**

# In[24]:


# frequency of each deck chosen per subject

deck1_95 = data['choice_95'][data['choice_95'] == 1].count(1) / 95
deck2_95 = data['choice_95'][data['choice_95'] == 2].count(1) / 95
deck3_95 = data['choice_95'][data['choice_95'] == 3].count(1) / 95
deck4_95 = data['choice_95'][data['choice_95'] == 4].count(1) / 95
goodfreq_95 = deck3_95 + deck4_95

deck1_100 = data['choice_100'][data['choice_100'] == 1].count(1) / 100
deck2_100 = data['choice_100'][data['choice_100'] == 2].count(1) / 100
deck3_100 = data['choice_100'][data['choice_100'] == 3].count(1) / 100
deck4_100 = data['choice_100'][data['choice_100'] == 4].count(1) / 100
goodfreq_100 = deck3_100 + deck4_100

deck1_150 = data['choice_150'][data['choice_150'] == 1].count(1) / 150
deck2_150 = data['choice_150'][data['choice_150'] == 2].count(1) / 150
deck3_150 = data['choice_150'][data['choice_150'] == 3].count(1) / 150
deck4_150 = data['choice_150'][data['choice_150'] == 4].count(1) / 150
goodfreq_150 = deck3_150 + deck4_150


# In[25]:


prepared_95 = pd.concat([deck1_95, deck2_95, deck3_95, deck4_95, goodfreq_95, aha_moments[0], cum_out_95['Trial_95'], cum_out_95['Trial_95']], axis=1)
prepared_100 = pd.concat([deck1_100, deck2_100, deck4_100, deck4_100, goodfreq_100, aha_moments[1], cum_out_100['Trial_95'], cum_out_100['Trial_100']], axis=1)
prepared_150 = pd.concat([deck1_150, deck2_150, deck3_150, deck4_150, goodfreq_150, aha_moments[2], cum_out_150['Trial_95'], cum_out_150['Trial_150']], axis=1)

prepared = pd.DataFrame(np.concatenate((prepared_95.values, prepared_100.values, prepared_150), axis=0))
prepared.columns = ['Freq1Deck1', 'Freq1Deck2', 'Freq1Deck3', 'Freq1Deck4', 'FreqGoodDecks', 'AhaMoment', 'CumOutTrial95', 'NetOut']
prepared


# **Standardization**

# Standardization is an important preprocessing step, prior to running the k-means algorithm, as it is a distance-based algorithm. It involves shifting the scales of each feature, so that the values for each feautre have a mean of 0 and a standard deviation of 1. Our dataset contains features with different scales or ranges. If we skipped this step of standardization, the clustering algorithm would not give equal importance to all features, e.g. NetOut would be considered much more important than FreqGoodDecks, as FreqGoodDecks ranges from 0 to 1, and NetOut is often in the thousands.

# In[26]:


scaled = StandardScaler().fit_transform(prepared)
scaled = pd.DataFrame(scaled, index = prepared.index, columns = prepared.columns)
scaled


# **Save dataframe**

# Add Study and Payoff Scheme features, as we will want to reference them in the next section.

# In[27]:


study_scheme = pd.concat([pd.concat([data['index_95']['Study'], data['index_95']['PayScheme']], axis=1), pd.concat([data['index_100']['Study'], data['index_100']['PayScheme']], axis=1), pd.concat([data['index_150']['Study'], data['index_150']['PayScheme']], axis=1)], axis=0).reset_index(drop=True)
processed = pd.concat([study_scheme, scaled], axis=1)


# In[28]:


processed.to_csv('../data/processed.csv')


# In[ ]:





# In[ ]:





# In[29]:


# maybe contrast payoff scheme 3 to the others


# In[ ]:





# In[30]:


# what is it about deck B, high % of picks despite being bad, higher highs? less frequent losses?

