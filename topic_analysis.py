import pandas as pd
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from wordfreq import top_n_list
import numpy as np
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import swifter
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk import word_tokenize
from corextopic import corextopic as ct

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True
mpl.style.use('ggplot')

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_excel(os.path.join(path, 'rli-articles-clean.xlsx'))

df.drop_duplicates(subset=['text', 'source'], inplace=True) # drop articles with same text and source
df.drop(columns=['source'], inplace=True)
df.rename(columns={'source_agg': 'source'}, inplace=True)
df = df.groupby('text').source.apply(list).reset_index().merge(
    df.drop(columns='source').drop_duplicates(subset=['text']), on='text')
df['text'] = df.text.str.strip()

# # Spacy lemmatization
# # load Dutch model
nlp = spacy.load("nl_core_news_lg")

def lemmatize(input_text: str, allowed_postags=['NOUN', 'ADJ', 'ADV']): # 'VERB'
    ''' Returns list of lemmatized tokens, filtering for stopwords and tokens shorter than 3 characters. '''
    doc = nlp(input_text)

    return [token.lemma_ for token in doc if token.is_stop == False and len(token) > 2]


# split into words
df['processed_text'] = df.text.apply(word_tokenize)

# deduplicate
df['processed_text'] = df.processed_text.apply(lambda x: ' '.join(x))
df = df.groupby('processed_text').source.apply(sum).reset_index().merge(
    df.drop(columns='source').drop_duplicates(subset=['processed_text']), on='processed_text')
df['processed_text'] = df.processed_text.apply(word_tokenize)

# remove non alphanumerical tokens
df['processed_text'] = df.processed_text.apply(lambda x: [word for word in x if word.isalnum()])

# remove tokens shorter than three characters
df['processed_text'] = df.processed_text.apply(lambda x: [elem for elem in x if len(elem)>2])

## define stop words (which are taken out of the analyis)
# stop words related to energy sources
stop_words = ['kernenergie', 'kerncentral', 'nucleaire energie', 'nucleaire stroom', 'nucleaire elektriciteit', 'windmolens',
    'atoomenergie', 'atoomstroom', 'nucleair', 'kerncentrale', 'atoomcentrale', 'kernreactor', 'kerncentrales',
    'nucleaire centrale', 'atoomreactor', 'fossiel', 'schaliegas', 'waterstof', 'kolen', 'gas', 'aardgas',
    'steenkool', 'aardolie', 'kolencentrale', 'windmolen', 'windenergie', 'zonnestrom', 'zonnepanelen', 'zon',
    'zonnepanel', 'zonne', 'zonnecel', 'biomassa', 'bruinkool', 'windpark', 'windturbine',
    'zonneenergie', 'molen', 'kolencentrales', 'zonneparken', 'zonneweide']
stop_words = stop_words + [' '.join(lemmatize(x)) for x in stop_words]
stop_words += top_n_list('nl', 2000) # add the 2000 most common words in Dutch
# remove relevant words
remove_from_stop_words = ['miljoen', 'veiligheid', 'cost', 'europa', 'euro',
    'leeftijd', 'organisatie', 'nationale','inwoners', 'president', 'politiek', 'project', 'staten',
    'europese', 'omgeving', 'internationale','politieke', 'partijen', 'universiteit', 'kiezen', 'ziekenhuis',
    'economie', 'verkiezingen', 'land', 'groen', 'overheid', 'geld', 'partij', 'generatie',
    'belgië', 'verenigd koninkrijk', 'groot brittannië', 'frankrijk', 'duitsland', 'spanje', 'italië',
    'verenigde Staten', 'gevaar', 'kost', 'miljard', 'economisch', 'prijs', 'bank', 'duur', 'kinderen',
    'links', 'rechts', 'kamer', 'groningen', 'zeeland', 'brabant', 'provincie', 'gemeente', 'middenoosten']
remove_from_stop_words = remove_from_stop_words + [' '.join(lemmatize(x)) for x in remove_from_stop_words]
stop_words = [item for item in stop_words if item not in remove_from_stop_words]

# Count vectorization
vec = CountVectorizer(min_df=5, ngram_range=(1,2), stop_words=stop_words, max_features=50000) # use bigrams
X = vec.fit_transform(df.processed_text.apply(lambda x: ' '.join(x)))
vocab = vec.get_feature_names()

## Anchored topic model
# Define seed words
topic_keywords = {'Climate impact': ['klimaat', 'co2', 'duurzaam', 'uitstoot', 'hernieuwbaar',
                'duurzaamheid', 'opwarmen', 'atmosfeer', 'temperatuurstijging', 'leefomgeving',
                'energietransitie', 'emissie', 'opwarming', 'ipcc', 'duurzame',
                'verduurzaming', 'verduurzamen', 'duurzamer', 'broeikasgassen', 'broeikasgas',
                'broeikaseffect', 'broeikasgasuitstoot', 'klimaatverandering', 'klimaatakkoord', 'klimaatprobleem',
                'klimaatneutraal', 'klimaatbeleid'],
        'Waste and storage': ['afval', 'opslag',
                'zoutkoepel', 'opbergen', 'covra', 'eindberging', 'ondergronds',
                'halfwaardetijd', 'restwarmte', 'opslaan', 'berging'],
        'Geopolitics': ['geopolitiek', 'iran', 'conflict', 'poetin', 'arsenaal',
                'centrifuge', 'ontwapening', 'verrijking', 'kernwapens', 'sancties', 'militair', 'militairen',
                'kernwapens'],
        'Safety': ['veiligheid', 'ramp', 'tsjernobyl', 'fukushima', 'straling', 'tsunami',
                'aardbeving', 'evacueren', 'explosie', 'catastrofe', 'radioactief', 'kanker', 'dosis',
                'dode', 'aanslag', 'kernsmelting', 'meltdown', 'millisievert', 'beveiliging',
                'ongeval', 'terrorisme', 'terroristen'],
        'Cost': ['kost', 'ontmanteling', 'levensduur', 'goedkoop', 'geld', 'subsidie', 'euro', 'cent',
                'miljoen', 'miljard', 'kilowattuur', 'kwh', 'economisch', 'rendabel',
                'capaciteit', 'prijs', 'uitgaven', 'financieel', 'financiën', 'financiering', 'financieren',
                'financieele', 'gefinancierd', 'financiers', 'investeren', 'investeringen', 'investering',
                'investeerders', 'investeert', 'investeerder', 'investeringsbank', 'bank', 'banken', 'duur',
                 'business case'],
        'Technology': ['zoutreactor', 'thorium', 'kernsplitsing', 'uranium',
                'splijtstof', 'neutronen', 'kettingreactie', 'koeling', 'koelmiddel', 'reactorvat',
                'kernsplijting', 'reactor', 'kernreactoren', 'reactoren', 'thoriumreactor', 'fusiereactor',
                'ontwerpen', 'smr', 'modulaire', 'technologie'],
        'Ethics': ['ethiek', 'ethisch', 'principe', 'principes', 'filosofie', 'filosofen',
                   'waarden', 'generatie', 'generaties', 'kinderen', 'kleinkinderen',
                   'moreel', 'moraal', 'afweging', 'mensheid', 'filosoof', 'religie', 'wetenschap',
                   'intellectueel', 'socioloog'],
        'Politics': ['vvd', 'partij', 'd66', 'groenlinks', 'cda', 'kabinet', 'pvda', 'verkiezingsprogramma',
                     'verkiezing', 'verkiezingen', 'rutte', 'coalitie', 'links', 'rechts', 'kamer',
                     'democratie', 'kabinet', 'formatie'],
        'Choice of site': ['groningen', 'zeeland', 'brabant', 'provincie', 'gemeente', 'borssele',
                           'eemshaven', 'maasvlakte', 'locatiekeuze', 'locaties', 'bouwlocaties']}

# Unused: broeikas graden duurzamheid duurzamheid duurzamheid zoutlaag kleilaag halveringstijd  ongeluk slachtoffer grafiet tritium waarde intergenerationeel middenoosten

# print seed words
for topic in topic_keywords:
    print(topic + ': ')
    print(', '.join(topic_keywords[topic]))

## Some methods used to extend the list of anchoring words
# find similar words via co-occurence matrix
# Xc = (X.T * X)
# Xc = Xc.todense()
# co_occ = pd.DataFrame(Xc, columns=vocab, index=vocab)
# co_occ['bouwlocaties'].sort_values(ascending=False).head(60)

# find similar words via pre-trained word2vec
# model = KeyedVectors.load_word2vec_format('C:/Users/Jakob/Downloads/wikipedia-160.txt')
# [x[0] for x in model.most_similar('energiezekerheid', topn=20)]

# example texts for keyword
# df[df.text.str.lower().str.contains('invest')].text

# find keywords containing sub-phrase
# word_counts = pd.Series(X.toarray().sum(axis=0), index=vocab)
# word_counts[word_counts.index.str.contains('sancties')].sort_values(ascending=False).head(50)

# Fit anchored topic model
topic_model = ct.Corex(n_hidden=15) # using 6 un-anchored topics
topic_model.fit(X, words=vocab, anchors=list(topic_keywords.values()), anchor_strength=10)

# print top n words per topic
topic_names = list(topic_keywords.keys())
topics = topic_model.get_topics(n_words=100)
topics_res = []
for topic_n,topic in enumerate(topics[:-6]): # [:-6] so we don't print sponge topics
    words, mis, signs = zip(*topic)
    # topic_str = topic_names[topic_n] +': '+', '.join([word for word in words if not word in ['eric']])
    # print(topic_str)
    # print words above mutual information score threshold
    # print('\n' + topic_names[topic_n] + ': ')
    for word_n, word in enumerate(words):
        topics_res.append([word, mis[word_n], topic_names[topic_n]])
        # if mis[word_n] > 0.209:
        #     print(word, '(' + str(mis[word_n]) + ')', end=', ')

topics_res = pd.DataFrame(topics_res, columns=['word', 'mis', 'topic'])
topics_res['voc_id'] = topics_res.word.apply(lambda x: vec.vocabulary_[x])
topics_res['num_hits'] = topics_res.voc_id.apply(lambda x: X.T[x].sum())
topics_res['anchor'] = topics_res.word.apply(lambda x: x in [item for sublist in list(topic_keywords.values()) for item in sublist])
for topic in topic_names:
    topic_res = topics_res[topics_res.topic == topic]
    print('\n' + topic + ': ')
    for index, row in topic_res.iterrows():
        print(row['word'] + str(['*' if row['anchor'] else ''][0]) + ' (' + str(row['num_hits']) + ')', end=', ')

# export topic model
# topic_model.save(os.path.join(path, 'corex-topic-model.pkl'), ensure_compatibility=False)

# get topic predictions per article
df_predictions = pd.DataFrame(topic_model.predict(X))
df_predictions.drop(columns=df_predictions.columns[-6:], inplace=True) # drop predictions for sponge topics
df_predictions.columns = topic_names
df_predictions = df_predictions.merge(df, left_index=True, right_index=True)

# export
df_predictions.to_pickle(os.path.join(path, 'rli-articles-with-topic-predictions.pkl'))

# predict topics on sentence level
df_sentences = pd.read_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner.pkl'))

X_sent = vec.transform(df_sentences.sentence)

df_sentences_predictions = pd.DataFrame(topic_model.predict(X_sent))
df_sentences_predictions.drop(columns=df_sentences_predictions.columns[-6:], inplace=True) # drop predictions for sponge topics
df_sentences_predictions.columns = topic_names
df_sentences_predictions = df_sentences_predictions.merge(df_sentences, left_index=True, right_index=True)

# export
df_sentences_predictions.to_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner-topics.pkl'))








## Unused code

# from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words=stop_words, max_features=10000)
# X = vec.fit_transform(df.processed_text.apply(lambda x: ' '.join(x)))
# vocab = vec.get_feature_names()

# from gensim.models import Phrases
# from gensim.models import Word2Vec
# from nltk import tokenize
# test = df.text
# test = test.apply(tokenize.sent_tokenize)
# test = test.apply(lambda x: [elem.split() for elem in x])
# test = test.agg(sum)
# bigram_transformer = Phrases(test)
# own_model = Word2Vec(bigram_transformer[test], epochs=20, min_count=10, workers=-1, sg=1) # skip-gram
# own_model = Word2Vec(bigram_transformer[test], epochs=20, min_count=5, workers=-1) # skip-gram
# own_model.wv.most_similar('veiligheid')
#
# own_model2 = Word2Vec(df.processed_text.to_list(), epochs=20, min_count=10, workers=-1, sg=1)
# own_model2.wv.most_similar('veiligheid')
#
# test2 = df.processed_text.to_list()
# bigram_transformer = Phrases(test2)
# own_model = Word2Vec(bigram_transformer[test2], epochs=10, min_count=10)
# own_model.wv.most_similar('veiligheid')
#
# from gensim.models import FastText
# model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
# model.build_vocab(test)
# model.train(test, total_examples=len(test), epochs=3)
# model.wv.most_similar('co2')
#
# from wikipedia2vec import Wikipedia2Vec
# wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
#
# from gensim.models import LsiModel
# lsi_model = LsiModel(corpus, id2word=d)
#
# lsi_model.print_topics()
# cm = CoherenceModel(model=lsi_model, corpus=corpus, dictionary=d, coherence='u_mass')
# cm.get_coherence()  # get coherence value

# texts = doc_list
#
# # Count word frequencies
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
#
# # Only keep words that appear more than once
# processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
#
# # Creates, which is a mapping of word IDs to words.
# words = corpora.Dictionary(processed_corpus)
#
# # Turns each document into a bag of words.
# corpus = [words.doc2bow(doc) for doc in processed_corpus]
#
# bigram = Phrases(corpus, min_count=1, threshold=3,delimiter=b' ')
#
# bigram_phraser = Phraser(bigram)
#
# bigram_token = []
# for doc in corpus:
#     bigram_token.append(bigram_phraser[doc])


# from string_grouper import group_similar_strings
#
# def group_strings(strings: List[str]) -> Dict[str, str]:
#     series = group_similar_strings(pd.Series(strings))
#
#     name_to_canonical = {}
#     for i, s in enumerate(strings):
#         deduped = series[i]
#         if (s != deduped):
#             name_to_canonical[s] = deduped
#
#     return name_to_canonical

# def lemmatize(input_text: str, allowed_postags=['NOUN', 'ADJ', 'ADV']): # 'VERB'
#     ''' Returns list of lemmatized tokens, filtering for stopwords and tokens shorter than 3 characters. '''
#     doc = nlp(input_text)
#
#     return [token.lemma_ for token in doc if token.is_stop == False and len(token) > 2]
#     # if token.is_stop
#     #     (token.pos_ in allowed_postags and token.is_stop == False) and token.is_punct == False]
#     # return [token.lemma_ for token in doc if (token.pos_ in allowed_postags and token.is_stop == False)]

# stem/lemmatize experiments
# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("dutch")
#
# stemmer.stem(' '.join(['duurzame', 'verduurzaming', 'verduurzamen', 'duurzamer', 'duurzamheid']))
# stemmer.stem(' '.join(['broeikasgassen', 'broeikasgas', 'broeikaseffect', 'broeikasgasuitstoot']))
# stemmer.stem(' '.join((lemmatize(' '.join(['financieel', 'financiën', 'financiering', 'financieren', 'financieele', 'gefinancierd', 'financiers'])))))
# lemmatize(' '.join(['investeren', 'investeringen', 'investering', 'investeerders', 'investeert', 'investeerder', 'investeringsbank', 'bank', 'banken']))

# Klimaatimpact
# Milieuimpact
# Kennisinfrastructuur
# Energiezekerheid
# Veiligheid
# Kernafval
# Kernreactoren als transitietechnologie
# Rol kernenergie in energiesysteem
# Geopolitiek
# Intergenerationeel
# Opbouw nucleaire industrie
# Ruimtelijke impact: locatiekeuze plaatsvinden
# Financiering kosten bouw
# Business case
# Besluitvormingsproces
# Technologie
# Draagvlak
# Continuering Borssele

# LDA model
# corpus = matutils.Sparse2Corpus(X.T)
# id2word = {v: k for k, v in vec.vocabulary_.items()}
# d = corpora.Dictionary()
# d.id2token = id2word
# d.token2id = {v: k for k, v in id2word.items()}
#
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=d,
#                                            num_topics=15,
#                                            random_state=2,
#                                            update_every=1,
#                                            passes=20,
#                                            alpha='auto',
#                                            eta=0.9,
#                                            per_word_topics=False)
#
# lda_model.print_topics()
#
# vis = gensimvis.prepare(lda_model, corpus, d)
# pyLDAvis.save_html(vis, os.path.join(path, 'Plots', 'lda_vis.html'))
#
# cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=d, coherence='u_mass')
# cm.get_coherence()  # get coherence value

# focus on paragraphs rather than whole articles
# df['text'] = df.text.str.split('\n')
# df = df.explode('text')
# df.reset_index(inplace=True)
# df = df[df.text.apply(len) > 15]


# lemmatize text
# df['processed_text'] = df.text.swifter.apply(lemmatize)

# calculate share of overlap for fuzzy deduplication
# def share_overlap(list_1, list_2):
#     set_1, set_2 = set(list_1), set(list_2)
#     len_overlap = len(set_1 & set_2)
#     len_total = (len(set_1) + len(set_2))/2
#
#     return len_overlap/len_total

# def form_query(keyword_list: list):
#     """ Returns regex query that matches on all keywords in the keyword list """
#
#     return '|'.join(keyword_list)
#
# # count occurences
# for topic in topic_keywords:
#     keywords = topic_keywords[topic]
#     # search both lemmatized and unlemmatized text
#     df[topic + '_matched'] = df.text.str.findall(form_query(topic_keywords[topic]), flags=re.I)
#     df[topic + '_matched'] = df[topic + '_matched'] + df.processed_text.apply(lambda x: ' '.join(x)).str.findall(
#             form_query(topic_keywords[topic]), flags=re.I)
#
# # example
# df[df['locatiekeuze'].str.len() > 0].text.sample().values
#
#
# def print_most_frequent(flag):
#     value_counts = df[df[flag].str.len() > 0][flag].explode().str.lower().value_counts()
#     for key, val in zip(value_counts.index, value_counts):
#         print(key + ' ('+ str(val) + '),', end = ' ')
#     print('\n')
#     return
#
# for topic in topic_keywords:
#     print(topic + ':', str(len(df[df[topic + '_matched'].str.len() > 0])) + ' articles')
#     print_most_frequent(topic + '_matched')
#
# # dummy flag for topics
# for topic in topic_keywords:
#     df[topic] = df[topic + '_matched'].str.len() > 0
#
# monthly = df.groupby(df.date.dt.to_period('Y'))[list(topic_keywords.keys())].agg(sum)
# monthly = monthly.div(df.groupby(df.date.dt.to_period('Y')).size(), axis=0)
# # monthly.rolling(window=12).mean().plot()
# monthly.plot()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xticks(rotation=45, ha="right")
# plt.xlabel('Year')
# plt.ylabel('Share of articles mentioning topic')
# plt.show()


# # Corextopic visualization
# from corextopic import vis_topic as vt
# vt.vis_rep(topic_model, column_label=vocab, prefix='topic-model-example')
#
# ## Guided LDA solution (Corex topics are more coherent)
# import guidedlda
#
# model = guidedlda.GuidedLDA(n_topics=9, random_state=7, refresh=20)
#
# word2id = vec.vocabulary_
#
# seed_topics = {}
# for t_id, st in enumerate(topic_keywords.values()):
#     for word in st:
#         seed_topics[word2id[word]] = t_id
#
# model.fit(X.toarray(), seed_topics=seed_topics, seed_confidence=0.3)
#
# # pretty print topics
# n_top_words = 10
# topic_word = model.topic_word_
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#
# ## Guided LDA vis
# import pyLDAvis
#
# # calculate doc lengths as the sum of each row of the dtm
# tef_dtm = pd.DataFrame(X.toarray())
# doc_lengths = tef_dtm.sum(axis=1, skipna=True)
#
# # transpose the dtm and get a sum of the overall term frequency
# dtm_trans = tef_dtm.T
# dtm_trans['total'] = dtm_trans.sum(axis=1, skipna=True)
#
# # create a data dictionary as per this tutorial https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/Movie%20Reviews%2C%20AP%20News%2C%20and%20Jeopardy.ipynb
# data = {'topic_term_dists':model.topic_word_, 'doc_topic_dists':model.doc_topic_, 'doc_lengths':doc_lengths, 'vocab':vocab, 'term_frequency':list(dtm_trans['total'])}
#
# # prepare the data
# tef_vis_data = pyLDAvis.prepare(**data)
#
# # this bit needs to be run after running the earlier code for reasons
# pyLDAvis.display(tef_vis_data)
#
# # save to HTML
# pyLDAvis.save_html(tef_vis_data, os.path.join(path, 'Plots', 'guided_lda_vis.html'))

# topic_predictions_plot = df_predictions.groupby(df_predictions.date.dt.to_period('Y'))[topic_names].sum() # groupby time period and sum
# # divide by total in each perido to get share of articles mentioning topic
# topic_predictions_plot = topic_predictions_plot.div(df_predictions.groupby(df_predictions.date.dt.to_period('Y')).size(), axis=0)
# # topic_predictions_plot = topic_predictions_plot.rolling(window=6).mean() # smooth via rolling average#
# # topic_predictions_plot = topic_predictions_plot.rolling(window=6, win_type='gaussian', center=True).mean(std=50)
# # resample and interpolate to smooth curves
# # topic_predictions_plot = topic_predictions_plot.resample('M', kind='timestamp').mean()
# # topic_predictions_plot = topic_predictions_plot.interpolate(method='cubicspline')
# topic_predictions_plot.plot(colormap='Set1', figsize=(7,7))
# plt.xlabel('Year')
# plt.ylabel('Share of articles')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# plt.ylim(ymin=0)
# plt.xticks(rotation=45, ha="right")
# plt.show()