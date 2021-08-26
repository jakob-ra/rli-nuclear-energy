import pandas as pd
import os
import swifter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_pickle(os.path.join(path, 'rli-sentencs-plus-translation.pkl'))

analyser = SentimentIntensityAnalyzer()

def sentiment_analysis_vader(input_text, vader_analyser):
    vs = vader_analyser.polarity_scores(input_text)

    return vs

sentiment_analysis_vader('Fukushima I still think that is one of the most shocking amoral events of recent years.', analyser)

df['vader_compound_sent'] = df.translated_text.apply(str).swifter.apply(lambda x: sentiment_analysis_vader(x, analyser)).apply(pd.Series)['compound']

df['sentiment'] = df.vader_compound_sent

# export
df.to_pickle(os.path.join(path, 'rli-sentence-translation-sentiment.pkl'))


# Sentiment analysis using TextBlob
# import spacy
# def sentiment_analysis_spacy(input_text, spacy_model):
#     doc = spacy_model(input_text)
#     polarity = doc._.polarity
#     # subjectivity = doc._.subjectivity
#
#     return polarity
#
# nlp = spacy.load("en_core_web_trf")
# nlp.add_pipe('spacytextblob')
#
# df['sentiment'] = df.translated_text.astype(str).swifter.apply(lambda x: sentiment_analysis_spacy(x, nlp))

# Sentiment analysis using BERTje
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
#
# classifier = pipeline('sentiment-analysis', model="wietsedv/bert-base-dutch-cased-finetuned-sentiment")
#
# def get_sentiment(text: str):
#     res = classifier(text)
#
#     return res[0]['score']
#
# df_sentences_predictions.sample(10).sentence.apply(get_sentiment)
#
# df_sentences_predictions['sentiment'] = df_sentences_predictions.sentence.apply(get_sentiment)

# sentiment analysis using VADER
