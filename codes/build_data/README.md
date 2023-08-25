# Code Structure

### 1_pre_covid_scrape.py

Function to scrape tweets from January till February 2020. It uses the module *snscrape*, [link](https://github.com/JustAnotherArchivist/snscrape).

### 2_URL_extraction.py

Function to extract all the links shared in our dataset of tweets.

### 3_URL_unshortening_mpi.py

Function to iteratively unshorten the links previously found. The code is built to run on a SLURM server using MPI4py. The companion batch file is **3_URL_unshortening.qsub**.

### 4_pre_covid_topics.py

Our newly scraped data, from January till February 2020, does not have any topic classification, whereas the original CIVICA dataset splits tweets based on content (*economics, politics, vaccines, health, art and entertainment, none*). We will train a model based on the CIVICA dataset tweets to predict whether a tweet is related to *economics/politics* or none.

### 5_split_topics.py

Due to the large size of the pre-COVID scrape, we split the tweets in batches.

### 6_topics_pred.py

Using the model trained in step (4), we predict each batch created in step (5). We will now have finished scraping and classifying our pre-covid sample.

### 7_training_set_pol.py

We prepare, using our distant supervision method, our training set, and split it into train/test/validation.
We will only use tweets classified as either *economics* or *politics* (as the others are deemed irrelevant for our purposes). Finally, we will use only users sharing more than 3 links, to avoid using tweets whose ideological alignment is difficult to approximate.

### 8_pol_pred.py

Using the training set from step (7), we fine-tune an AlBERTo model, see also their Hugging Face page ([link](https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0)) or GitHub repository ([link](https://github.com/marcopoli/AlBERTo-it)).

### 9_ml_baseline.py

We train a series of classical Machine Learning Models with a variety of text-vectorization processes to provide a baseline comparison for our main BERT-based model.

### 10_predict_pol.py

Using our best model we predict in batches our full dataset.