###########################################################
# TOPIC MODELING - GSDMM
# Author: Luca Adorni
# Date: March 2023
###########################################################

# 0. Setup -------------------------------------------------

import re
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords
import os
import sys

sys.path.append(f"{os.getcwd()}gsdmm/")

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

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

os.makedirs(path_to_ctm, exist_ok=True)
os.makedirs(path_to_lda, exist_ok=True)
os.makedirs(path_to_gsdmm, exist_ok=True)

# define a string of punctuation symbols
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
# Load italian stopwords
stop_words = stopwords.words('italian')
# Load SpaCy IT model
nlp = spacy.load("it_core_news_sm")

# 1. Define main functions --------------------------------------------------------

from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import json

class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.
        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a MovieGroupProcess from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
            _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        '''
        Score a document
        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p),max(p)

# functions to clean tweets
def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet


def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet


def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def remove_keywords(tweet):
    """Takes a string and removes the main keywords"""
    tweet = re.sub('(coronavirus|covid19|COVID-19|COVID19italia|Coronavid19|pandemia|corona|virus)', '', tweet)  # remove hash tags
    return tweet

def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet


def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in stop_words \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(nlp(token)[0].lemma_)
    return result


def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = remove_keywords(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    return tweet


def basic_clean(tweet):
    """Main master function to clean tweets only without tokenization or removal of stopwords"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub('📝 …', '', tweet)
    return tweet


def tokenize_tweets(df):
    """Main function to read in and return cleaned and preprocessed dataframe.
    This can be used in Jupyter notebooks by importing this module and calling the tokenize_tweets() function
    Args:
        df = data frame object to apply cleaning to
    Returns:
        pandas data frame with cleaned tokens
    """

    df['tokens'] = df.tweet_text.apply(preprocess_tweet)
    num_tweets = len(df)
    print('Complete. Number of Tweets that have been cleaned and tokenized : {}'.format(num_tweets))
    return df



# define helper functions
def top_words(cluster_word_distribution, top_cluster, values):
    '''prints the top words in each cluster'''
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s'%(cluster,sort_dicts))
        print(' - - - - - - - - -')
        
def cluster_importance(mgp):
    '''returns a word-topic matrix[phi] where each value represents
    the word importance for that particular cluster;
    phi[i][w] would be the importance of word w in topic i.
    '''
    n_z_w = mgp.cluster_word_distribution
    beta, V, K = mgp.beta, mgp.vocab_size, mgp.K
    phi = [{} for i in range(K)]
    for z in range(K):
        for w in n_z_w[z]:
            phi[z][w] = (n_z_w[z][w]+beta)/(sum(n_z_w[z].values())+V*beta)
    return phi

def topic_allocation(df, docs, mgp, topic_dict):
    '''allocates all topics to each document in original dataframe,
    adding two columns for cluster number and cluster description'''
    topic_allocations = []
    for doc in tqdm(docs):
        topic_label, score = mgp.choose_best_label(doc)
        topic_allocations.append(topic_label)

    df['cluster'] = topic_allocations

    df['topic_name'] = df.cluster.apply(lambda x: get_topic_name(x, topic_dict))
    print('Complete. Number of documents with topic allocated: {}'.format(len(df)))

def get_topic_name(doc, topic_dict):
    '''returns the topic name string value from a dictionary of topics'''
    topic_desc = topic_dict[doc]
    return topic_desc


# helper functions to extract all data needed to create bubble charts for exploring words in each topic
def top_words_dict(cluster_word_distribution, top_cluster, n_words):
    '''returns a dictionary of the top n words and the number of docs they are in;
    cluster numbers are the keys and a tuple of (word, word count) are the values'''
    top_words_dict = {}
    for cluster in top_cluster:
        top_words_list = []
        for val in range(0, n_words):
            if len(mgp.cluster_word_distribution[cluster]) == 0:
                continue
            # If we have reached the limit of the words, stop here
            elif len(mgp.cluster_word_distribution[cluster]) <= val:
                continue
            top_n_word = sorted(mgp.cluster_word_distribution[cluster].items(), 
                                key=lambda item: item[1], reverse=True)[:n_words][val]    #[0]
            top_words_list.append(top_n_word)
        top_words_dict[cluster] = top_words_list

    return top_words_dict

def get_word_counts_dict(top_words_nclusters):
    '''returns a dictionary that counts the number of times a word 
    appears only in the top n words list across all the clusters;
    words are the keys and a count of the word is the value'''
    word_count_dict = {}
    for key in top_words_nclusters:
        words_score_list = []
        for word in top_words_nclusters[key]:
            if word[0] in word_count_dict.keys():
                word_count_dict[word[0]] += 1
            else:
                word_count_dict[word[0]] = 1
    return word_count_dict

def get_cluster_importance_dict(top_words_nclusters, phi):
    '''returns a dictionary that of all top words and their cluster
    importance value for each cluster;
    cluster numbers are the keys and a list of word 
    importance computed scores are the values'''
    cluster_importance_dict = {}
    for key in top_words_nclusters:
        words_score_list = []
        for word in top_words_nclusters[key]:
            importance_score = phi[key][word[0]]
            words_score_list.append(importance_score)
        cluster_importance_dict[key] = words_score_list
    return cluster_importance_dict

def get_doc_counts_dict(top_words_nclusters):
    '''returns a dictionary of only the doc counts of each top n word for each cluster;
    cluster numbers are the keys and a list of doc counts are the values'''
    doc_counts_dict = {}
    for key in top_words_nclusters:
        doc_counts_list = []
        for word in top_words_nclusters[key]:
            num_docs = word[1]
            doc_counts_list.append(num_docs)
        doc_counts_dict[key] = doc_counts_list
    return doc_counts_dict

def get_word_frequency_dict(top_words_nclusters, word_counts):
    '''returns a dictionary of only the number of occurences across all 
    clusters for each word in a particular cluster's top n words;
    cluster numbers are the keys and a list of 
    word occurences counts are the values'''
    word_frequency_dict = {}
    for key in top_words_nclusters:
        words_count_list = []
        for word in top_words_nclusters[key]:
            words_count_list.append(word_counts[word[0]])
        word_frequency_dict[key] = words_count_list

    return word_frequency_dict

from gensim.models import Phrases

def prepare_tokens(docs, min_bigrams = 20):
    bigram = Phrases(docs, min_count = min_bigrams)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


# 2. Load our dataset --------------------------------------------

# PARAMETER ---------
stem = True
stem_tag = np.where(stem, "_stemm","")
period = 'periods'
if period == 'months':
    # Do LDA for the most relevant weeks during the pandemic
    week_dict = {'feb': [2], 'mar': [3], 'apr': [4], 'may': [5], 
                'june': [6], 'jul': [7],'aug': [8], 'sep': [9], 
                'oct': [10],'nov': [11], 'dec': [12]}

    n_topics = 50
elif period == 'periods':
    # Do LDA for the most relevant weeks during the pandemic
    week_dict = {'first_lockdown': [2,3,4], 
                'end_lock': [5,6],
                'summer': [7,8,9], 
                'second_lockdown': [10,11,12]}
    n_topics = 100

# ----------------------------------------------------------------

final_df = pd.read_pickle(f"{path_to_processed}cleaned_gsdmm_tweets{stem_tag}.pkl.gz", compression = 'gzip')

# 3. Apply Topic Modelling to each period --------------------------------------------

# Create also 0/1 for being in a certain group
final_df = pd.concat([final_df, pd.get_dummies(final_df.polarization_bin)], axis = 1)

store = []

for key_w, week_date in week_dict.items():
    print(key_w)
    sub_df = final_df.loc[final_df.week_start.dt.month.isin(week_date)]

    # Train a topic for politic tweets
    pol_df = sub_df.loc[(sub_df.topic == 'politics')|(sub_df.topic == 'economics')]
    # And for health tweets
    vac_df = sub_df.loc[(sub_df.topic == 'health')|(sub_df.topic == 'vaccine')]
    vac_df2 = sub_df.loc[(sub_df.topic == 'vaccine')]

    iter_dict = {'_pol': pol_df, '_vacc': vac_df}

    for tag, df in iter_dict.items():
        print(tag)
        # convert string of tokens into tokens list
        df['tokens'] = df.tokens.apply(lambda x: re.split('\s', x))
        # create a single list of tweet tokens
        docs = df['tokens'].tolist()
        docs = prepare_tokens(docs)

        # Train STTM model
        #    K = number of potential topics
        #    alpha = controls completeness
        #    beta =  controls homogeneity 
        #    n_iters = number of iterations
        try:
            # load in trained model 
            filehandler = open(f'{path_to_gsdmm}/{n_topics}clusters_{key_w}{tag}{stem_tag}.model', 'rb')
            mgp = pickle.load(filehandler)
            print("Model already trained")
            vocab = set(x for doc in docs for x in doc)
            n_terms = len(vocab)
        except:
            mgp = MovieGroupProcess(K=n_topics, alpha=0.1, beta=0.5, n_iters=10)
            vocab = set(x for doc in docs for x in doc)
            n_terms = len(vocab)
            y = mgp.fit(docs, n_terms)

            # Save model
            with open(f'{path_to_gsdmm}/{n_topics}clusters_{key_w}{tag}{stem_tag}.model', 'wb') as f:
                pickle.dump(mgp, f)
                f.close()

        doc_count = np.array(mgp.cluster_doc_count)
        print('Number of documents per topic :', doc_count)
        print('*'*20)

        # topics sorted by the number of documents they are allocated to
        top_index = doc_count.argsort()[-10:][::-1]
        print('Most important clusters (by number of docs inside):', top_index)
        print('*'*20)

        # show the top 5 words in term frequency for each cluster 
        topic_indices = np.arange(start=0, stop=len(doc_count), step=1)
        top_words(mgp.cluster_word_distribution, topic_indices, 10)


        # declare any static variables needed 
        nwords = 20
        phi = cluster_importance(mgp)

        # define and generate dictionaries that hold each topic number and its values
        top_nwords = top_words_dict(mgp.cluster_word_distribution, topic_indices, nwords)
        word_count = get_word_counts_dict(top_nwords)
        word_frequency = get_word_frequency_dict(top_nwords, word_count)
        cluster_importance_dict = get_cluster_importance_dict(top_nwords, phi)
            
        # add all values for each topic to a list of lists
        rows_list = []
        for cluster in range(0, n_topics):
            words = [x[0] for x in top_nwords[cluster]]
            doc_counts = [x[1] for x in top_nwords[cluster]]
            
            # create a list of values which represents a 'row' in our data frame 
            rows_list.append([int(cluster), cluster, words, doc_counts, 
                            word_frequency[cluster], cluster_importance_dict[cluster]])
                
        topic_words_df = pd.DataFrame(data=rows_list, 
                                    columns=['cluster','topic_name', 'top_words',
                                                'doc_count', 'num_topic_occurrence', 'word_importance'])

        # save data frame to pickle file
        topic_words_df.to_csv(f'{path_to_gsdmm}/{n_topics}clusters_df_{key_w}{tag}{stem_tag}.csv')

#         df[f'{key_w}{tag}_{n_topics}'] = ""
#         df['ind'] = range(len(df))
#         df[f'{key_w}{tag}_{n_topics}'] = df.ind.apply(lambda x: mgp.choose_best_label(docs[x]))

#         store.append(df)

# store = pd.concat(store)
# store.to_pickle(f"{path_to_gsdmm}/final_gsdmm_df_{n_topics}{stem_tag}.pkl.gz", compression = 'gzip')