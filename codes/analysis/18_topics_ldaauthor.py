###########################################################
# TOPIC MODELING - Author Topic Model
# Author: Luca Adorni
# Date: May 2023
###########################################################

# 0. Setup -------------------------------------------------

import re
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import sys
import spacy
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import json
from gensim.models import AuthorTopicModel
from gensim.models import TfidfModel
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Phrases
import warnings
import datetime as dt
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore")



pd.options.display.max_columns = 200
pd.options.display.max_rows = 1000
pd.set_option('max_info_columns', 200)
pd.set_option('expand_frame_repr', False)
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',1000)
pd.set_option('display.width',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Setup Repository
try:
    # Setup Repository
    with open("repo_info.txt", "r") as repo_info:
        path_to_repo = repo_info.readline()
except:
    path_to_repo = f"{os.getcwd()}/polpo/"
    sys.path.append(f"{os.getcwd()}/.local/bin") # import the temporary path where the server installed the module

  
print(path_to_repo)

path_to_data = f"{path_to_repo}data/"
path_to_raw = f"{path_to_data}raw/"
path_to_links = f"{path_to_data}links/"
path_to_processed = f"{path_to_data}processed/"
path_to_figures = f"{path_to_repo}figures/"
path_to_figure_odds = f"{path_to_figures}logit_res/"
path_to_tables = f"{path_to_repo}tables/"
path_to_ctm = f"{path_to_data}ctm/"
path_to_lda = f"{path_to_data}lda/"
path_to_gsdmm = f"{path_to_data}gsdmm/"
path_to_author = f"{path_to_data}author/"

os.makedirs(path_to_ctm, exist_ok=True)
os.makedirs(path_to_lda, exist_ok=True)
os.makedirs(path_to_gsdmm, exist_ok=True)
os.makedirs(path_to_author, exist_ok=True)

# define a string of punctuation symbols
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
# Load italian stopwords
stop_words = stopwords.words('italian')
# Load SpaCy IT model
nlp = spacy.load("it_core_news_sm")

# 1. Define main functions --------------------------------------------------------



def prepare_tokens(docs, min_bigrams = 20):
    bigram = Phrases(docs, min_count = min_bigrams)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


def create_corpus(data_words, tf_idf = True):
  # Create Dictionary
  id2word = corpora.Dictionary(data_words)
  print(len(id2word))
  id2word.filter_extremes(no_below = 50, no_above = 0.8)
  print(len(id2word))
  # Create Corpus
  texts = data_words
  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]
  if tf_idf:
    model = TfidfModel(corpus)  # fit model
    corpus = [model[text] for text in corpus] # apply model to the first corpus document
  return id2word, texts, corpus


# 2. Load our dataset --------------------------------------------

# PARAMETER ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
period = 'periods'
# Number of topics
n_topics = 5

# Load the dataset
final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')

# Create also 0/1 for being in a certain group
final_df = pd.concat([final_df, pd.get_dummies(final_df.polarization_bin)], axis = 1)

# 3. Apply Topic Modelling to each period --------------------------------------------

week_dict = {'first_lock': [3, 4],
            'post_lock': [5, 6],
             'summer': [7, 8, 9], 
            'second_lockdown': [10,11,12]}

store = []

for key_w, week_date in week_dict.items():
    print(key_w)
    
    sub_df = final_df.loc[final_df.week_start.dt.month.isin(week_date)]

    # Train a topic for politic tweets
    pol_df = sub_df.loc[(sub_df.topic == 'politics')|(sub_df.topic == 'economics')]
    # And for health tweets
    vac_df = sub_df.loc[(sub_df.topic == 'health')|(sub_df.topic == 'vaccine')]

    iter_dict = {'_pol': pol_df}

    for tag, df in iter_dict.items():
        print(tag)
        # convert string of tokens into tokens list
        df['tokens'] = df.tokens.apply(lambda x: re.split('\s', x))
        # create a single list of tweet tokens
        docs = df['tokens'].tolist()
        docs = prepare_tokens(docs)

        id2word, texts, corpus = create_corpus(docs, tf_idf = False)

        # Get all author names and their corresponding document IDs.
        author2doc = dict()
        doc_id = 0
        for i, row in df.iterrows():
                if not author2doc.get(row.polarization_bin):
                    # This is a new author.
                    author2doc[row.polarization_bin] = []

                # Add document IDs to author.
                author2doc[row.polarization_bin].append(doc_id)
                doc_id += 1

        try:
            # Load model.
            author_model = AuthorTopicModel.load(f"{path_to_author}final/authorlda_{key_w}{tag}{stem_tag}_n{n_topics}ext.atmodel")
            print("Model already trained")
        except:
            author_model = AuthorTopicModel(corpus= corpus, 
                                            author2doc=author2doc, 
                                            id2word=id2word, 
                                            num_topics=n_topics,
                                            eta = 0.01,
                                            random_state=42,
                                            alpha = 0.5
            )
            
            # Save model.
            author_model.save(f"{path_to_author}final/authorlda_{key_w}{tag}{stem_tag}_n{n_topics}ext.atmodel")

        
        # get the topic descriptions
        topic_sep = re.compile("0\.[0-9]{3}\*") # getting rid of useless formatting
        # extract a list of tuples with topic number and descriptors from the model
        author_model_topics = [(topic_no, re.sub(topic_sep, '', model_topic).split(' + ')) for topic_no, model_topic in
                        author_model.print_topics(num_topics=n_topics, num_words=30)]

        # Print the top 30 words
        author_descriptors = []
        for i, m in sorted(author_model_topics):
            print(i, ", ".join(m[:10]))
            author_descriptors.append(", ".join(m[:3]).replace('"', '') + f' - Topic {i}')

        # initialize mapping from covariate(=author/country) to topic distro, set all to 0.0
        author_vecs = {author: {author_descriptors[t]: 0.0
                                for t in range(author_model.num_topics)}
                    for author in author_model.id2author.values()
                    }
        # update mappings from model
        for author in author_model.id2author.values():
            for (t, v) in author_model.get_author_topics(author):
                author_vecs[author][author_descriptors[t]] = v

        # make a DataFrame
        author_df = pd.DataFrame.from_dict(author_vecs)

        # Reorder the author df
        author_df = author_df[['far_left', 'center_left', 'center', 'center_right', 'far_right']]

        # Get the index
        author_df['topic_n'] = author_df.index
        author_df['topic_n'] = author_df.topic_n.apply(lambda x: re.findall(r'\d+',x)[0]).astype(int)
        author_df.reset_index(drop = True, inplace = True)

        # Now attach the first 20 words
        author_df['top_words'] = author_df.topic_n.apply(lambda x: author_model_topics[x][1])

        # And finally flag month/type
        author_df['month'] = key_w
        author_df['types'] = tag
        store.append(author_df)


store = pd.concat(store)

store.to_csv(f"{path_to_author}final/resulting_topics_periods_final.csv", index = False)