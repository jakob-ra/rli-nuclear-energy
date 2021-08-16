import pandas as pd
import os
import swifter
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_pickle(os.path.join(path, 'rli-sentence-translation-sentiment.pkl'))

# load Dutch tagger from Flair
tagger = SequenceTagger.load("flair/ner-dutch")

def extract_entities(text: str, tagger):
    if len(text) < 30:
        return []
    try:
        sentence = Sentence(text)
        tagger.predict(sentence)
        results = []
        for entity in sentence.get_spans('ner'):
            label = str(entity.to_dict()['labels'][0])[:3]
            entity = entity.to_plain_string()
            results.append((label, entity))

        return results

    except:
        return []


df['sentence'] = df.sentence.astype(str)
df['ner_result'] = df.sentence.swifter.apply(lambda x: extract_entities(x, tagger))

def extract_entity_for_type(ner_result: list, desired_type):
    entity_names = []
    for label in ner_result:
        entity_type, entity_name = label
        if entity_type == desired_type:
            entity_names.append(entity_name)

    return entity_names

df['persons'] = df.ner_result.apply(lambda x: extract_entity_for_type(x, 'PER'))
df['organizations'] = df.ner_result.apply(lambda x: extract_entity_for_type(x, 'ORG'))
df['locations'] = df.ner_result.apply(lambda x: extract_entity_for_type(x, 'LOC'))

# export
df.to_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner.pkl'))

