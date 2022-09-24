# POLitical POlarization and COVID-19: A Case Study on Italian Twitter

This repository contains the code to reproduce the results of the paper "Political Polarization and COVID-19: A Case Study on Italian Twitter".
The initial dataset was provided by CIVICA, and it is a dataset of Italian Twitter scraped from 2020 till 2021 using COVID-19 related keywords.

## Preprocessing:

### 1. Pre COVID-19 Scraping:

The first step to our analysis will be to scrape all the users we have in the CIVICA dataset. A total of 19 milion tweets were scraped from the beginning of January 2020 till the end of February 2020. No keywords were used. The goal is to have a pre-COVID dataset to compare the differences in political polarization of the same users pre and post shock.
The code *1_pre_covid_scrape.py* does that.

### 2. Distant Supervision: URL Extraction

To perform distant supervision and create a set of weak labels to train our models on, we extracted all links present in tweets, and then unshortened them.
Codes *2_URL_extraction.py* and *3_URL_unshortening.py* do that.