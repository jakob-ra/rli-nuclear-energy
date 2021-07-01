import pandas as pd
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim import matutils
from wordfreq import top_n_list
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import swifter
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


path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_excel(os.path.join(path, 'rli-articles-clean.xlsx'))

# df = df.sample(100)

# focus on paragraphs rather than whole articles
# df['text'] = df.text.str.split('\n')
# df = df.explode('text')
# df.reset_index(inplace=True)

df.drop_duplicates(subset=['text', 'source'], inplace=True) # drop articles with same text and source
df.drop(columns=['source'], inplace=True)
df.rename(columns={'source_agg': 'source'}, inplace=True)
df = df.groupby('text').source.apply(list).reset_index().merge(
    df.drop(columns='source').drop_duplicates(subset=['text']), on='text')
df['text'] = df.text.str.strip()

# load Dutch model
nlp = spacy.load("nl_core_news_sm", disable=['parser', 'ner'])
# nlp.Defaults.stop_words |= set(top_n_list('nl', 1000))

def lemmatize(input_text: str, allowed_postags=['NOUN', 'ADJ', 'ADV']): # 'VERB'
    ''' Returns list of lemmatized tokens, and only tokens that are tagged as one of the allowed_postags and
     aren't punctuation or stopwords. '''
    doc = nlp(input_text)
    # print([(token.lemma_, token.pos_) for token in doc])
    return [token.lemma_ for token in doc if
        (token.pos_ in allowed_postags and token.is_stop == False)]
    # return [token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)]

lemmatize('hallo hoe gaat et met jouw? vriendelijk, deuren, boten, huiz')

df['processed_text'] = df.text.swifter.apply(lemmatize)

# df.to_excel(os.path.join(path, 'rli-paragraphs-processed.xlsx'), index=False)

df.processed_text.explode().value_counts().head(50)

stop_words = ['kernenergie', 'nucleaire energie', 'nucleaire stroom', 'nucleaire elektriciteit',
    'atoomenergie', 'atoomstroom', 'nucleair', 'kerncentrale', 'atoomcentrale', 'kernreactor',
    'nucleaire centrale', 'atoomreactor', 'fossiel', 'schaliegas', 'waterstof', 'kolen', 'gas', 'aardgas',
    'steenkool', 'aardolie', 'kolencentrale', 'windmolen', 'windenergie', 'zonnestrom', 'zonnepanelen', 'zon',
    'zonnepanel', 'zonne', 'zonnecel', 'biomassa', 'bruinkool', 'windpark', 'windturbine', 'molen']
stop_words += top_n_list('nl', 2000)
stop_words = [item for item in stop_words if item not in ['miljoen', 'veiligheid', 'cost', 'europa', 'euro',
    'leeftijd', 'organisatie', 'nationale','inwoners', 'president', 'politiek', 'project', 'staten',
    'europese', 'omgeving', 'internationale','politieke', 'partijen', 'universiteit', 'kiezen', 'ziekenhuis',
    'economie', 'verkiezingen', 'land', 'groen', 'overheid', 'geld', 'partij', 'generatie',
    'belgië', 'Verenigd Koninkrijk', 'Groot Brittannië', 'Frankrijk', 'Duitsland', 'Spanje', 'Italië',
    'Verenigde Staten']]
# stop_words = [lemmatize(x) for x in stop_words]
# stop_words = set([item for sublist in stop_words for item in sublist])

# def share_overlap(list_1, list_2):
#     set_1, set_2 = set(list_1), set(list_2)
#     len_overlap = len(set_1 & set_2)
#     len_total = (len(set_1) + len(set_2))/2
#
#     return len_overlap/len_total

df['processed_text'] = df.processed_text.apply(set).apply(lambda x: ', '.join(x))
df = df.groupby('processed_text').source.apply(sum).reset_index().merge(
    df.drop(columns='source').drop_duplicates(subset=['processed_text']), on='processed_text')

vec = CountVectorizer(min_df=10, ngram_range=(1,2), stop_words=stop_words, max_features=10000)
X = vec.fit_transform(df.processed_text)
vocab = vec.get_feature_names()

corpus = matutils.Sparse2Corpus(X.T)
# id2word = dict([(i, s) for i, s in enumerate(vocab)])
id2word = {v: k for k, v in vec.vocabulary_.items()}
d = corpora.Dictionary()
d.id2token = id2word
d.token2id = {v: k for k, v in id2word.items()}

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=d,
                                           num_topics=15,
                                           random_state=2,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           eta=0.9,
                                           per_word_topics=True)

lda_model.print_topics()

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

vis = gensimvis.prepare(lda_model, corpus, d)
pyLDAvis.save_html(vis, os.path.join(path, 'Plots', 'lda_vis.html'))

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
#
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
# Ruimtelijke impact: lokatiekeuze
# Financiering kosten bouw
# Business case
# Besluitvormingsproces
# Technologie
# Draagvlak
# Continuering Borssele

