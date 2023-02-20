###########################################################
# Code to create plot graphs
# Author: Luca Adorni
# Date: January 2023
###########################################################

# 0. Setup -------------------------------------------------

import numpy as np
import pandas as pd
import os
import re
import pickle
import sys
from collections import Counter
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import random
from wordcloud import WordCloud
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from nltk.probability import FreqDist
import gensim.corpora as corpora
import gensim
from gensim.models.coherencemodel import CoherenceModel
import dill
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter("ignore")
# set seaborn whitegrid theme
sns.set_style('ticks')
sns.set_context("paper", font_scale = 1.5)
# set custom color palette
colors = ['#ca0020','#f4a582','#bdc9c1','#92c5de','#0571b0']
my_palette = sns.color_palette(colors)
params = {'axes.labelsize': 25,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'legend.fontsize': 20,
          'legend.title_fontsize': 20}
plt.rcParams.update(params)


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

os.makedirs(path_to_ctm, exist_ok=True)
os.makedirs(path_to_lda, exist_ok=True)

# 2. Load User Predictions ----------------------------------------------------------------------


try:
    final_df = pd.read_pickle(f"{path_to_processed}tweet_for_topic.pkl.gz", compression = 'gzip')
    print("Loaded Data")
except:
    print("Creating data")
    # Load our predicted dataset
    df = pd.read_pickle(f"{path_to_processed}final_df_clean.pkl.gz", compression = 'gzip')
    # Keep only the info we want
    df = df[['week_start', 'scree_name', 'polarization_bin', 'prediction', 'extremism_toleft', 'extremism_toright', 'orient_change_toleft', 'orient_change_toright']]

    # Load back the tweets
    post_df = pd.read_csv(f"{path_to_raw}tweets_without_duplicates_regions_sentiment_demographics_topic_emotion.csv")
    post_df['dates'] = pd.to_datetime(post_df.dates)
    post_df['week_start'] = post_df['dates'].dt.to_period('W').dt.start_time
    # Drop columns we do not need
    post_df.drop(columns = ['tweet_ids', 'user_description' ,'locations', 'dates', 'place', 'polygon', 'user_id', 'profile_url', 'is_org'], inplace = True)

    # Restrict to 2020
    df = df.loc[(df.week_start <= pd.datetime(2020, 12, 31))& (df.week_start > pd.datetime(2019, 12, 31))]

    # Now merge the two sets back
    final_df = post_df.merge(df, how = 'inner', on = ['week_start', 'scree_name'], validate = 'm:1')

    # Save it
    final_df.to_pickle(f"{path_to_processed}tweet_for_topic.pkl.gz", compression = 'gzip')

# 3. Initialize Topic Model ----------------------------------------------------------------------

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import nltk
from unidecode import unidecode
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# STOPWORDS = stopwords.words("italian")
# # we initialize our stemmer
# stemmer = SnowballStemmer("italian", ignore_stopwords=True)

# def text_prepare(text) :
#     """
#         text: a string        
#         return: modified initial string
#     """    
#     text = text.lower() # lowercase text
#     text = emoji.demojize(text) # convert emojis to text
#     text = unidecode((text))
#     text = re.sub("#|@", "", text) # take away hashtags or mentions but keep the word
#     text = re.sub(r'(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+', "", text)
#     text =  " ".join([x for x in text.split()if x not in STOPWORDS]) # delete stopwords from text
#     # text =  " ".join([stemmer.stem(x) for x in text.split()])
#     # text =  " ".join([x for x in text.split()])
#     return text

# # Get a list of them
# documents = pol_df.tweet_clean.tolist()

# # Preprocess them
# sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=STOPWORDS)
# preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

# # Initialize our topic model
# tp = TopicModelDataPreparation("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
# # Then Train it
# training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

# ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=20, num_epochs=10)
# ctm.fit(training_dataset) # run the model

# path_to_models = f"{path_to_data}models/"
# ctm.save(models_dir=f"{path_to_models}topic_ctm_pol")


# # Get a list of them
# documents = vac_df.tweet_clean.tolist()

# # Preprocess them
# sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=STOPWORDS)
# preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

# # Initialize our topic model
# tp = TopicModelDataPreparation("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
# # Then Train it
# training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

# ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=20, num_epochs=10)
# ctm.fit(training_dataset) # run the model

# path_to_models = f"{path_to_data}models/"
# ctm.save(models_dir=f"{path_to_models}topic_ctm_vac")

import matplotlib.colors as mcolors
session_seed = 42
# OLD LDA -------------------------

def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = emoji.demojize(tweet) # convert emojis to text
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return tweet

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


stop_words = stopwords.words('italian')

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def tokenize(text):
    """
    Performs tokenization on each text we pass through, also eliminating the stems, punctuation, and lowercasing everything
    """
    
    final_text = [stem.stem(word) for word in text]
    return final_text

stem = nltk.stem.SnowballStemmer('italian')

def stemming(texts):
    texts_out = []
    for sent in texts:
      texts_out.append(tokenize(sent))
    return texts_out

# supporting function
def compute_coherence_values(corpus, dictionary,topic_num, data_words):
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num, random_state=session_seed, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence(), lda_model


def prepare_tokens(filtered, min_bigrams = 20):

  data_words = list(sent_to_words(filtered.tweet_text.apply(lambda x: cleaner(x))))
  data_words = remove_stopwords(data_words)
  data_words = stemming(data_words)

  # Compute bigrams.
  # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
  bigram = Phrases(data_words, min_count = min_bigrams)
  for idx in range(len(data_words)):
      for token in bigram[data_words[idx]]:
          if '_' in token:
              # Token is a bigram, add to document.
              data_words[idx].append(token)
  return data_words


# define a function that saves figures


def get_top_tokens(data_words, path, n_words = 30, title = 'Top Words'):
  #iterate through each tweet, then each token in each tweet, and store in one list
  flat_words = [item for sublist in data_words for item in sublist]

  word_freq = FreqDist(flat_words)

  #retrieve word and count from FreqDist tuples

  most_common_count = [x[1] for x in word_freq.most_common(n_words)]
  most_common_word = [x[0] for x in word_freq.most_common(n_words)]

  #create dictionary mapping of word count
  top_30_dictionary = dict(zip(most_common_word, most_common_count))

  #Create Word Cloud of top 30 words
  wordcloud = WordCloud(colormap = 'Paired', background_color = 'white', width=800, height=400)\
  .generate_from_frequencies(top_30_dictionary)

  #plot with matplotlib
  plt.figure(figsize=(20, 10))

  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.title(title, fontsize = 30)
  plt.tight_layout(pad=0)

  plt.tight_layout()
  plt.savefig(path, format='pdf', dpi=300)

def create_corpus(data_words):
  # Create Dictionary
  id2word = corpora.Dictionary(data_words)
  print(len(id2word))
  id2word.filter_extremes(no_below = 3, no_above = 0.8)
  print(len(id2word))
  # Create Corpus
  texts = data_words
  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]
  return id2word, texts, corpus


def get_best_topic(corpus, id2word, data_words, tune = True, topics = 5):
  # Topics range
  min_topics = 10
  max_topics = 100
  step_size = 10
  topics_range = range(min_topics, max_topics, step_size)
  old_coh = 0
  if tune == True:
    lda_list = {}
    for i in topics_range:
      topic_coh, lda_topic = compute_coherence_values(corpus, id2word, i, data_words)
      lda_list[i] = lda_topic
      if topic_coh > old_coh + 0.02:
        old_coh = topic_coh
        best_topic = i
    save_lda = lda_list[best_topic]
    print(f'Best Topic: {best_topic} - {old_coh}')
  
  else:
    save_lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=topics, random_state=session_seed, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    best_topic = topics
  return save_lda, best_topic

def get_lda(corpus, id2word, lda_model, data_words, print_topics = True):
  # Build LDA model
  #%time lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=best_topic, random_state=session_seed, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
  topic_df = pd.DataFrame({'Topic_' + str(i): [token for token, score in lda_model.show_topic(i, topn=30)] for i in range(0, lda_model.num_topics)}).T
  topic_df['Topic_num'] = [topic[0] for topic in lda_model.show_topics(num_topics = 100)]
  # Compute Perplexity
  print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

  # Compute Coherence Score
  coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  print('\nCoherence Score: ', coherence_lda)
  if print_topics:
    for idx, topic in lda_model.print_topics(-1, num_words = 30):
      print('Topic: {} \nWords: {}'.format(idx, topic))
  return lda_model, topic_df

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def display_top_tweets(lda_model, corpus, data_words, filtered):
  df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts = data_words)
  df_topic_sents_keywords["tweet"] = list(filtered.tweet_text)

  # Format
  #df_dominant_topic = df_topic_sents_keywords.reset_index()
  #df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Tweet']
  #df_dominant_topic.head(10)

  # Display setting to show more characters in column
  pd.options.display.max_colwidth = 100

  sent_topics_sorteddf_mallet = pd.DataFrame()
  sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

  for i, grp in sent_topics_outdf_grpd:
      sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                              grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                              axis=0)

  # Reset Index    
  sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

  # Format
  sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text", "Tweet"]
  sent_topics_sorteddf_mallet.drop('Representative Text', axis = 1, inplace = True)
  return sent_topics_sorteddf_mallet

def word_cloud_lda(lda_model, top_topics, path, max_topics = 100):
  cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

  h_x = int(len(top_topics)/5)

  cols = cols*h_x

  cloud = WordCloud(
                    background_color='white',
                    width=750,
                    height=750,
                    max_words=20,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

  topic_uncleaned = lda_model.show_topics(num_topics = max_topics, formatted=False)
  topics = []
  for top in topic_uncleaned:
    if top[0] in top_topics:
      topics.append(top)

  fig, axes = plt.subplots(h_x, 5, figsize=(40,45), sharex=True, sharey=True)

  for i, ax in enumerate(axes.flatten()):
      fig.add_subplot(ax)
      topic_words = dict(topics[i][1])
      cloud.generate_from_frequencies(topic_words, max_font_size=300)
      plt.gca().imshow(cloud)
      plt.gca().set_title('Topic ' + str(topics[i][0]), fontdict=dict(size=30, fontweight = 'bold'))
      plt.gca().axis('off')


  plt.axis('off')
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.05, hspace=0.15)

  plt.tight_layout()
  plt.savefig(path, format='pdf', dpi=300)

def word_topic_imp(lda_model, data_ready, best_topic, top_topics, path, max_topics = 100):

  topic_uncleaned = lda_model.show_topics(num_topics = max_topics, formatted=False)
  topics = []
  for top in topic_uncleaned:
    if top[0] in top_topics:
      topics.append(top)

  data_flat = [w for w_list in data_ready for w in w_list]
  counter = Counter(data_flat)

  out = []
  for i, topic in topics:
      for word, weight in topic:
          out.append([word, i , weight, counter[word]])

  df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

  # Plot Word Count and Weights of Topic Keywords
  fig, axes = plt.subplots(len(top_topics)//2, 2, figsize=(30,45), sharey=True, dpi=160)
  cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
  cols = cols*(best_topic//10)
  for i, ax in enumerate(axes.flatten()):
      ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
      ax_twin = ax.twinx()
      ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
      ax.set_ylabel('Word Count', color=cols[i])
      ax_twin.set_ylim(0, df.importance.max()*1.1); ax.set_ylim(0, int(df.word_count.max()*1.1))
      ax.set_title('Topic: ' + str(topics[i][0]), color=cols[i], fontsize=16)
      ax.tick_params(axis='y', left=False)
      ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
      ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

  fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
  
  fig.tight_layout(w_pad=2)    
  plt.savefig(path, format='pdf', dpi=300)


def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

def get_top_topics(model, corpus):

  dominant_topics, topic_percentages = topics_per_document(model=model, corpus=corpus, end=-1)      
  df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
  dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
  top_topics = list(dominant_topic_in_each_doc.sort_values(ascending = False).head(30).index)
  return top_topics


def top_topics_overall(lda_model, corpus, best_topic, top_topics, path, n = 30, title_tag = ''):
  dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

  # Distribution of Dominant Topics in Each Document
  df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
  dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
  df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

  # Total Topic Distribution by actual weight
  topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
  df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

  # Top 3 Keywords for each Topic
  topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False, num_topics = best_topic) 
                                  for j, (topic, wt) in enumerate(topics) if j < 3]

  df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
  df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
  df_top3words.reset_index(level=0,inplace=True)

  df_dominant_topic_in_each_doc['count'] = df_dominant_topic_in_each_doc['count']/df_dominant_topic_in_each_doc['count'].sum()

  df_dominant_topic_in_each_doc = df_dominant_topic_in_each_doc[df_dominant_topic_in_each_doc.Dominant_Topic.isin(top_topics)]
  df_topic_weightage_by_doc = df_topic_weightage_by_doc[df_topic_weightage_by_doc.index.isin(top_topics)]
  df_top3words = df_top3words[df_top3words.index.isin(top_topics)]
  df_top3words.reset_index(inplace = True)
  # Plot
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(45, 35), dpi=160, sharey=False)

  # Topic Distribution by Dominant Topics
  sns.barplot(data = df_dominant_topic_in_each_doc, x = 'Dominant_Topic', y = 'count', ax = ax1, color = 'firebrick')
  sns.despine()
  #tick_formatter = FuncFormatter(lambda x, pos: '' + str(int(df_top3words.loc[df_top3words.index==x, 'topic_id']))+ '\n' + df_top3words.loc[df_top3words.index==x, 'words'].values[0])
  tick_formatter = FuncFormatter(lambda x, pos: '' + str(int(df_top3words.loc[df_top3words.index==x, 'topic_id'])))
  ax1.xaxis.set_major_formatter(tick_formatter)
  ax1.tick_params(axis='x', which='major', labelsize=20)
  ax1.set_ylabel('% of Tweets')
  ax1.text(x=0.125, y=0.905, s=f"Tweets by Dominant Topic {title_tag}", fontsize=45, ha="left", transform=fig.transFigure)
  ax1.text(x=0.125, y=0.885, s= "Number of tweets by dominant topic - labels are percentage over total sample", fontsize=30, ha="left", transform=fig.transFigure)
  rects = ax1.patches

  # Make some labels.
  labels = list(df_dominant_topic_in_each_doc.loc[:, 'count'])

  for rect, label in zip(rects, labels):
      height = rect.get_height()
      ax1.text(
          rect.get_x() + rect.get_width() / 2, height + 0.001, round(label,2), ha="center", va="bottom", fontsize = 22
      )

  # Topic Distribution by Topic Weights
  sns.barplot(data = df_topic_weightage_by_doc, x = 'index', y = 'count', ax = ax2, color = 'steelblue')
  sns.despine()
  ax2.xaxis.set_major_formatter(tick_formatter)
  ax2.text(x=0.125, y=0.484, s=f"Tweets by Topic Weightage {title_tag}", fontsize=30, ha="left", transform=fig.transFigure)
  ax2.text(x=0.125, y=0.471, s= "Number of tweets by topic weightage", fontsize=20, ha="left", transform=fig.transFigure)

  plt.tight_layout()    
  plt.savefig(path, format='pdf', dpi=300)


def filter_words(tweet, id2word):
  tweet_clean = []
  for text in tweet:
    if text in list(id2word.token2id.keys()):
      tweet_clean.append(text)

  return tweet_clean



# # Keep only extremism tweets
# topic_df = final_df.loc[final_df.extremism_toright == 1]

# # Clean our tweets
# topic_df['tweet_clean'] = topic_df.tweet_text.apply(lambda x: text_prepare(x))

# create dumies for left/right/etc.
subset_df = pd.concat([pd.get_dummies(final_df.polarization_bin), final_df[["scree_name","extremism_toleft","extremism_toright","orient_change_toleft","orient_change_toright"]]], axis = 1)
subset_df = subset_df.groupby('scree_name').max()

# extract users which switched to being extremists at least once
right_switch_list = list(subset_df[(subset_df.extremism_toright == 1)].index.unique())
len(right_switch_list)

# extract users which never switched to the far right, but represents the right
right_noswitch = list(subset_df[(subset_df.extremism_toright == 0)  & (subset_df.far_right == 0) & (subset_df.center_right == 1) & (subset_df.far_left == 0)  & (subset_df.center_left == 0)].index.unique())
len(right_noswitch)


final_df["switch"] = final_df.scree_name.isin(right_switch_list)
final_df["moderate"] = final_df.scree_name.isin(right_noswitch)

final_df['target_extr'] = 0
final_df.loc[final_df.extremism_toright == 1, 'target_extr'] = 'Extremist'
final_df.loc[(final_df.extremism_toright == 0) & (final_df.switch == 1), 'target_extr'] = 'Extremist - Pre Switch'
final_df.loc[(final_df.extremism_toright == 0) & (final_df.switch == 0), 'target_extr'] = 'Moderate'
final_df["irrelevant"] = final_df.scree_name.isin(list(set(right_switch_list + right_noswitch)))


topic_df = final_df.loc[final_df.irrelevant == True]

# Train a topic for politic tweets
pol_df = topic_df.loc[(topic_df.topic == 'politics')|(topic_df.topic == 'economics')]
# And for health tweets
vac_df = topic_df.loc[(topic_df.topic == 'health')|(topic_df.topic == 'vaccine')]
vac_df2 = topic_df.loc[(topic_df.topic == 'vaccine')]

del final_df, topic_df

iter_dict = {'_vacc': vac_df, '_vacc2': vac_df2, '_pol': pol_df}


for tag, df in iter_dict.items():
  data_words = prepare_tokens(df, min_bigrams = 5)
  id2word, texts, corpus = create_corpus(data_words)

  df['tokenized'] = data_words
  df['tokenized_clean'] = df.tokenized.apply(lambda x: filter_words(x, id2word))

  for i in df.target_extr.unique(): 
    get_top_tokens(list(df.loc[df.target_extr == i,'tokenized_clean']), path = f'{path_to_lda}top_tokens_{i}{tag}.pdf', n_words = 30)


  try:
    lda_model = gensim.models.ldamodel.LdaModel.load(f"{path_to_lda}lda_model{tag}")
    print("Model Loaded")
    best_topic = len(lda_model.show_topics(num_topics = 100))
    print(f'Best Topic: {int(best_topic)}')
  except:
    lda_model, best_topic = get_best_topic(corpus, id2word, data_words)
    lda_model.save(f"{path_to_lda}lda_model{tag}")
    print("Model saved")


  top_topics = get_top_topics(lda_model, corpus)

  lda_model, topic_df = get_lda(corpus, id2word, lda_model, data_words, print_topics = False)
  pd.set_option('display.max_columns', 30)
  topic_df.loc[topic_df.Topic_num.isin(top_topics)]

  top_tweets = display_top_tweets(lda_model, corpus, data_words, df)

  top_tweets.loc[top_tweets.Topic_Num.isin(top_topics),:].head(45).to_csv(f"{path_to_lda}sample_topics{tag}.csv")

  word_cloud_lda(lda_model, top_topics = top_topics, path = f'{path_to_lda}lda_cloud{tag}.pdf')

  word_topic_imp(lda_model, data_words, best_topic, top_topics = top_topics, path = f'{path_to_lda}lda_word_imp{tag}.pdf')

  top_topics_overall(lda_model, corpus, best_topic, top_topics = top_topics, path = f'{path_to_lda}lda_top_topics{tag}.pdf')

  df['corpus'] = corpus

  for i in vac_df.target_extr.unique():
    top_topics_overall(lda_model, list(df.loc[df.target_extr == i, 'corpus']), best_topic, top_topics = top_topics, path = f'{path_to_lda}lda_top_topics_{i}{tag}.pdf')