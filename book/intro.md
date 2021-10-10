# Introduction

## Iowa Gambling Task Overview

The Iowa Gambling Task (IGT), designed by {cite:t}`bechara_damasio_damasio_anderson_1994`, studies decision-making under uncertainty, in a context of punishment and reward. Originally physical playing cards were used, whereas nowadays, the IGT is primarily computer based. The IGT requires participants to repeatedly draw cards from four possible decks (A, B, C and D). Associated with each card picked is a monetary gain and/or loss. However, the expected values of the four decks differ, in that two of the decks (i.e. the good decks) yield lower immediate rewards but long-term overall gains, and two of the decks (i.e., the bad decks) yield higher immediate rewards but long-term overall loss. Additionally, the decks differ on two other features: the relative number of gains vs. losses (i.e., the gain frequency), and the relative number of net losses (i.e., the loss frequency). After every selection, participants receive feedback about how much money they’ve won and/or lost as a result of that selection. Participants start with a “loan” and are instructed to try to make a profit. 

To perform well in the IGT, participants must identify which are the good and bad decks, and choose to forgo the short-term benefits of the bad decks for the long-term benefits of the good decks. Hence, the success of a participant depends on their ability to focus on the long-term outcome. Accordingly, {cite:t}`brand_recknor_grabenhorst_bechara_2007` has shown that the IGT can be used as a measure of impaired decision-making in a range of psychiatric and neurological conditions. 

```{figure} ./igt_screenshot.jpg
---
height: 300px
name: igt-fig
---
What a participant might see
```

## Procedures

The analysis conducted in this paper (book/report?) uses data collected during 10 independent published studies. All participants were deemed healthy (i.e., no known neurological impairments) and performed a computerized version of the IGT. Depending on the study, participants completed 95, 100 or 150 trials. The payoff scheme used was not consistent between IGT studies, each study was based on one of the following payoff schemes:

* *Payoff Scheme 1* is the traditional payoff scheme, with the following features:
	* Decks C and D are the good decks, with a net outcome of 10 cards being +250. Decks A and B are the bad decks with the net outcome of 10 cards being -250.
	* Decks B and D have infrequent losses, while decks A and C have frequent losses.
	* There is a fixed win within each deck.
	* There is a variable loss in deck C of -25, -50, or -75.
	* Fixed sequence of rewards and losses.  
&nbsp;
* *Payoff Scheme 2* is very similar to Payoff Scheme 1, except with the following **differences**:
	* The loss in deck C is held constant at -50.
	* Randomly shuffled sequence of rewards and losses.  
&nbsp;
* *Payoff Scheme 3* was first introduced by {cite:t}`bechara_damasio_2002`. This payoff scheme has the following **differences** to Payoff Scheme 1:
	* The schedules of rewards and losses change in both the good and bad decks. The net outcome of the good decks (decks C and D) increases by 25 every block of 10 cards (i.e., in the first block, the net outcome is 250, in the second it is 275, and in the sixth block it is 375). Conversely, the net outcome of the bad decks (decks A and B) decreases by 150 every block of 10 cards (i.e., in the first block, the net outcome is -250, in the second it is -400, and in the sixth block it is -1000). Thus, the good decks become gradually better, whereas the bad decks become gradually worse.
	* The wins vary within each deck.  
:::{note}
A detailed description of the payoff schemes can be found here (include link?).
:::


## Data Source

The data used in this analysis was compiled by {cite:t}`Steingroever_Fridberg_Horstmann_Kjome_Kumari_Lane_Maia_McClelland_Pachur_Premkumar`, and can be readily downloaded from [https://osf.io/8t7rm/](https://osf.io/8t7rm/) (at time of publishing 13/010/2021). It consists of data collected from 617 participants across 10 independent IGT studies. For each participant, each deck selection and the resulting win/loss has been captured.  
An overview of the studies can be seen in the below table.



| Study | Number of Participants | Number of Trials | Payoff |
| :- | :- | :- | :- |
| {cite:t}`fridberg_queller_ahn_kim_bishara_busemeyer_porrino_stout_2010` | 15 | 95 | 1 |
| {cite:t}`Steingroever_Wetzels_Wagenmakers_2013` | 162 | 100 | 2 |
| {cite:t}`maia_mcclelland_2004` | 19 | 100 | 3 |
| {cite:t}`kjome_lane_schmitz_green_ma_prasla_swann_moeller_2010` | 40 | 100 | 1 |
| {cite:t}`premkumar_fannon_kuipers_simmons_frangou_kumari_2008` | 25 | 100 | 3 |
| {cite:t}`steingroever_pachur_šmíra_lee` | 70 | 100 | 2 |
| {cite:t}`steingroever_2011` | 57 | 150 | 2 |
| {cite:t}`wetzels_vandekerckhove_tuerlinckx_wagenmakers_2010` | 41 | 150 | 2 |
| {cite:t}`Wood_Busemeyer_Koling_Cox_Davis_2005` | 153 | 100 | 3 |
| {cite:t}`Worthy_Pang_Byrne_2013` | 35 | 100 | 1 |

```{admonition} Caveat
:class: warning
As each of the studies were conducted in different environments, inconsistent factors such as compensation methods, incentives, and instructions received may have affected the performance of participants.
```


