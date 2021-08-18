import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True
mpl.style.use('ggplot')

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

# import sentence-level results
df_sentences_predictions = pd.read_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner-topics.pkl'))

topic_names = ['Climate impact', 'Waste and storage', 'Geopolitics', 'Safety', 'Cost',
       'Technology', 'Ethics', 'Politics', 'Choice of site']

# plot topic prominence in number of sentences
# topic_prominence = df_sentences_predictions[['date'] + topic_names].copy()
# # topic_prominence.set_index('date', inplace=True)
# # topic_prominence.sort_index(inplace=True)
# # topic_prominence = topic_prominence.rolling(window='1000D', closed='both', min_periods=4000).mean()
# # topic_prominence = topic_prominence.rolling(window='365D', closed='both').mean()
# # # topic_prominence = topic_prominence.groupby(topic_prominence.date.dt.to_period('Y')).mean()
# # topic_prominence.plot(colormap='Set1', figsize=(7,7))
# # plt.xlabel('Year')
# # plt.ylabel('Topic prominence')
# # plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# # plt.xticks(rotation=45, ha="right")
# # # plt.savefig(os.path.join(path, 'Plots', 'topic-prominence-over-time-sentences'))
# # plt.show()

df_sentences_predictions[topic_names].sum()

df_sentences_predictions.columns
# plot average sentiment per topic
topic_sent = df_sentences_predictions[topic_names].multiply(df_sentences_predictions['sentiment'], axis=0)
topic_sent = topic_sent.sum()/(df_sentences_predictions[topic_names].sum())
topic_sent.sort_values().plot(kind='barh')
plt.xlabel('Average sentiment')
plt.ylabel('Topic')
plt.savefig(os.path.join(path, 'Plots', 'average-topic-sentiment'))
plt.show()

# plot average sentiment per topic over time
topic_sent_time = df_sentences_predictions[['date', 'sentiment'] + topic_names].copy()
topic_sent_time[topic_names] = topic_sent_time[topic_names].replace(False, np.nan)
topic_sent_time[topic_names] = topic_sent_time[topic_names].multiply(topic_sent_time['sentiment'], axis=0)
topic_sent_time[topic_names] = topic_sent_time[topic_names].apply(pd.to_numeric)
# topic_sent_time = topic_sent_time.groupby(topic_sent_time.date.dt.to_period('Q'))[topic_names].mean()
topic_sent_time.set_index('date', inplace=True)
topic_sent_time.sort_index(inplace=True)
# topic_sent_time = topic_sent_time.rolling(window='365D', closed='both')[topic_names].mean()
# topic_sent_time = topic_sent_time.groupby(topic_sent_time.index.to_period('M')).mean()
topic_sent_time = topic_sent_time.ewm(halflife='365D', times=topic_sent_time.index)[topic_names].mean()
topic_sent_time = topic_sent_time.reset_index().drop_duplicates(subset=['date'], keep='first').set_index('date')
topic_sent_time = topic_sent_time.rolling(window='365D', closed='both')[topic_names].mean()
topic_sent_time.iloc[40:].plot(cmap='Set1', figsize=(7,7))
plt.ylabel('Average topic sentiment')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()


# average sentiment per source
source_sent = df_sentences_predictions.groupby('source_agg').sentiment.mean().sort_values()
source_sent.plot(kind='barh')
plt.xlabel('Average sentiment')
plt.ylabel('Source')
plt.savefig(os.path.join(path, 'Plots', 'average-sentiment-across-sources'))
plt.show()

# average topic sentiment per source
source_topic_sent = df_sentences_predictions[['source_agg', 'sentiment'] + topic_names].copy()
source_topic_sent[topic_names] = source_topic_sent[topic_names].replace(False, np.nan)
source_topic_sent[topic_names] = source_topic_sent[topic_names].multiply(source_topic_sent.sentiment, axis=0)
source_topic_sent = source_topic_sent.explode('source_agg')
source_topic_sent[topic_names] = source_topic_sent[topic_names].apply(pd.to_numeric)
source_topic_sent = source_topic_sent.groupby('source_agg')[topic_names].mean()

source_topic_sent.plot(figsize=(9,9), colormap='Set1', linestyle='None', marker='o', markersize=12, markeredgecolor='black')
plt.xlabel('Source')
plt.ylabel('Average sentiment for topic')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 14})
plt.xticks([i for i in range(len(source_topic_sent.index))], source_topic_sent.index, rotation=45, ha='right')
plt.savefig(os.path.join(path, 'Plots', 'topic-sentiment-across-sources'))
plt.show()

## top NER
# top organizations
def fix_orgs(found_entities: list):
    entities = []
    for entity in found_entities:
        if 'Forum' in entity:
            entities.append('Forum voor Democratie')
            continue
        if entity == 'EU':
            entities.append('Europese Unie')
            continue
        if entity == 'CU':
            entities.append('ChristenUnie')
            continue
        if entity == 'Commissie':
            entities.append('Europese Commissie')
            continue
        if entity not in ['Klimaat', 'Kamer', 'kernenergie', 'Rijk', 'België', 'Belgi', 'Milieu']:
            entities.append(entity)

    return entities
df_sentences_predictions['organizations'] = df_sentences_predictions.organizations.apply(fix_orgs)
df_sentences_predictions.organizations.explode().value_counts(ascending=True).tail(50).plot(kind='barh', figsize=(6,10))
plt.ylabel('Organisation')
plt.xlabel('Number of mentions')
plt.savefig(os.path.join(path, 'Plots', 'top-50-organizations'))
plt.show()

# top persons
def fix_persons(found_entities: list):
    entities = []
    for entity in found_entities:
        if entity == 'Franois Hollande':
            entities.append('François Hollande')
            continue
        if entity == 'Bie':
            entities.append('Eric de Bie')
            continue
        if entity == 'De Bie':
            entities.append('Eric de Bie')
            continue
        if entity == 'JanLeen Kloosterman':
            entities.append('Jan Leen Kloosterman')
            continue
        if entity == 'Kloosterman':
            entities.append('Jan Leen Kloosterman')
            continue
        if entity == 'Leen':
            entities.append('Jan Leen Kloosterman')
            continue
        if entity not in ['Volt', 'Franciscus', 'Isral', 'God', 'Wubbo', 'Wise']:
            entities.append(entity)

    return entities

entities[entities.index.str.contains('Kloosterman')]
df_sentences_predictions['persons'] = df_sentences_predictions.persons.apply(fix_persons)

def first_last_name_deduplication(all_persons: list):
    first_and_last_names = []
    for entity in all_persons:
        if len(entity.split()) == 2:
            first_and_last_names.append(entity)

    name_dict = {}
    for entity in all_persons:
        if len(entity.split()) == 1:
            for name in first_and_last_names:
                if entity == name.split()[1] and name_dict.get(entity)==None:
                        name_dict.update({entity: name})

    return name_dict


choices = df_sentences_predictions.persons.explode().dropna().unique()
name_dict = first_last_name_deduplication(choices)

df_sentences_predictions['persons'] = df_sentences_predictions.persons.explode().replace(name_dict).reset_index().groupby('index')['persons'].apply(list)
df_sentences_predictions.persons.explode().replace(name_dict).value_counts(ascending=True).tail(50).plot(kind='barh', figsize=(6,10))
plt.ylabel('Person')
plt.xlabel('Number of mentions')
plt.savefig(os.path.join(path, 'Plots', 'top-50-persons'))
plt.show()


## NER network
df_sentences_predictions['entities'] = df_sentences_predictions.persons + df_sentences_predictions.organizations # take both people and organizations

entities = df_sentences_predictions.entities.explode().value_counts() # rank entities by # of mentions
entities = entities[entities > 5] # take entities which appear more than 5 times
newspaper_names = list(df_sentences_predictions.source_agg.unique())
newspaper_names += ['De Volkskrant', 'Volkskrant', 'Telegraaf', 'FD', 'Financieele Dagblad']
entities = entities[entities.index.map(lambda x: x not in newspaper_names)] # remove newspapers from entities

entity_names = list(entities.index)

res = df_sentences_predictions.entities.explode().dropna()
res = res[res.apply(lambda x: x in entity_names)]

res = res.groupby(res.index).apply(set)
res = res[res.apply(len) > 1]

entities_co_occ = pd.DataFrame(0, index=entity_names, columns=entity_names)
for index, item in res.iteritems():
    pair = list(item)
    entities_co_occ.loc[pair[0], pair[1]] += 1
    entities_co_occ.loc[pair[1], pair[0]] += 1

# find topic loading for each entity
entity_topic_loadings = df_sentences_predictions.explode('entities').groupby('entities')[topic_names].mean()
# entity_topic_loadings = entity_topic_loadings.iloc[entity_topic_loadings.sum(axis=1).to_numpy().nonzero()[0].tolist()]
entity_dom_topic = entity_topic_loadings.T.idxmax()
# entity_dom_topic = pd.DataFrame(index=entities.index).merge(pd.DataFrame(entity_dom_topic), left_index=True, right_index=True, how='left').squeeze()

import networkx as nx
G = nx.from_pandas_adjacency(entities_co_occ)

# remove all isolates from G
G.remove_nodes_from(list(nx.isolates(G)))

# remove components smaller than 5 from G
for component in list(nx.connected_components(G)):
    if len(component)<5:
        for node in component:
            G.remove_node(node)

G.nodes(data=True)
for n, data in G.nodes(data=True):
    data['size'] = entities[entities.index == n].values[0]
    data['topic'] = str(entity_dom_topic.loc[n])
    data['topic_loadings'] = str(entity_topic_loadings.loc[n].sort_values(ascending=False))

# export to Gephi
nx.readwrite.graphml.write_graphml(G, path=os.path.join(path, 'rli-entity-network.graphml'))


# from netwulf import visualize
# visualize(G)



# import nx_altair as nxa
# import altair_viewer
#
# # Draw the graph using Altair
# viz = nxa.draw_networkx(G)
#
# # Show it as an interactive plot!
# viz.interactive()
#
# altair_viewer.show(viz.interactive())