from google.cloud import translate
import pandas as pd
import os
from nltk import tokenize

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_pickle(os.path.join(path, 'rli-sentences.pkl'))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'C:/Users/Jakob/Downloads/rli-nuclear-energy-417965bfaf9a.json'

# prepare input file for Google Cloud translate, upload this in the Google Cloud Bucket 'rli-nuclear-energy'
df['sentence'].to_csv(os.path.join(path, 'rli-articles-to-translate.tsv'), sep="\t",
                  header=False, encoding='utf-8', index=False)

def batch_translate_text(
    input_uri="gs://rli-nuclear-energy/rli-articles-to-translate.tsv",
    output_uri="gs://rli-nuclear-energy/results/",
    project_id="rli-nuclear-energy",
    timeout=6000,
):
    """Translates a batch of texts on GCS and stores the result in a GCS location."""

    client = translate.TranslationServiceClient()

    location = "us-central1"
    # Supported file types: https://cloud.google.com/translate/docs/supported-formats
    gcs_source = {"input_uri": input_uri}

    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    }
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/language
    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "nl",
            "target_language_codes": ["en"],  # Up to 10 language codes here.
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout)

    print("Total Characters: {}".format(response.total_characters))
    print("Translated Characters: {}".format(response.translated_characters))

    return response

batch_translate_text()

# download translations and put them in path as 'results_rli-nuclear-energy_en_translations_sentence_level.tsv'
translations = pd.read_csv(os.path.join(path, 'results_rli-nuclear-energy_en_translations_sentence_level.tsv'), sep='\t', header=None,
                           names=['index', 'original_text', 'translated_text'])

df = df.merge(translations, left_index=True, right_on='index')

df.drop(columns=['index', 'index_x', 'index_y', 'original_text'], inplace=True)

# export
df.to_pickle(os.path.join(path, 'rli-sentencs-plus-translation.pkl'))