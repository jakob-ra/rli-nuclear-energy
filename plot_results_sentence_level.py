import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True
mpl.style.use('ggplot')

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

# import sentence-level results
df = pd.read_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner-topics.pkl'))

df.drop_duplicates(['sentence', 'text', 'source_agg'], inplace=True) # remove duplicates

topic_names = ['Climate impact', 'Waste and storage', 'Geopolitics', 'Safety', 'Cost',
       'Technology', 'Ethics', 'Politics', 'Choice of site']

# plot topic prominence in number of sentences
# topic_prominence = df[['date'] + topic_names].copy()
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


# plot average sentiment per topic
topic_sent = df[topic_names].multiply(df['sentiment'], axis=0)
topic_sent = topic_sent.sum()/(df[topic_names].sum())
topic_sent.sort_values().plot(kind='barh')
plt.xlabel('Average sentiment')
plt.ylabel('Topic')
plt.savefig(os.path.join(path, 'Plots', 'average-topic-sentiment'))
plt.show()

# plot average sentiment over time
# nuclear_keywords = ['kern', 'nucl', 'atoom']
# sent_time = df[df.sentence.str.lower().str.contains('|'.join(nuclear_keywords))]
sent_time = df[df.sentiment != 0].copy()
# sent_time = sent_time.groupby(df.date.dt.to_period('M')).sentiment.mean()
sent_time.set_index('date', inplace=True)
sent_time.sort_index(inplace=True)
sent_time = sent_time.rolling(window='700D', closed='both', min_periods=1000).sentiment.mean()
sent_time.plot()
plt.ylabel('Article sentiment (moving average)')
plt.xlabel('Year')
plt.savefig(os.path.join(path, 'Plots', 'average-sentiment-over-time'))
plt.show()

# df[(df.date.dt.to_period('Y').apply(str).isin(['2015', '2014', '2016'])) & (df.sentiment == -1)].translated_text
#
# df['translated_text'] = df.translated_text.apply(str)
# df = df.merge(df.groupby('text', as_index=False).agg({'sentiment': 'mean', 'translated_text': 'sum'}).sort_values(by='sentiment'), on='text', suffixes=['', '_article'])
# df.drop_duplicates('text').sort_values(by='sentiment_article').head(20).translated_text_article.values
#
#
# nuclear = df[df.sentence.str.lower().str.contains('|'.join(nuclear_keywords))]
# nuclear[nuclear.sentiment == -1].translated_text.values

# plot average sentiment per topic over time
topic_sent_time = df[['date', 'sentiment'] + topic_names].copy()
topic_sent_time[topic_names] = topic_sent_time[topic_names].replace(False, np.nan)
topic_sent_time[topic_names] = topic_sent_time[topic_names].multiply(topic_sent_time['sentiment'], axis=0)
topic_sent_time[topic_names] = topic_sent_time[topic_names].apply(pd.to_numeric)
# topic_sent_time = topic_sent_time.groupby(topic_sent_time.date.dt.to_period('Q'))[topic_names].mean()
topic_sent_time.set_index('date', inplace=True)
topic_sent_time.sort_index(inplace=True)
# topic_sent_time = topic_sent_time.rolling(window='365D', closed='both')[topic_names].mean()
# topic_sent_time = topic_sent_time.groupby(topic_sent_time.index.to_period('M')).mean()
# topic_sent_time = topic_sent_time.ewm(halflife='365D', times=topic_sent_time.index)[topic_names].mean()
# topic_sent_time = topic_sent_time.reset_index().drop_duplicates(subset=['date'], keep='first').set_index('date')
topic_sent_time = topic_sent_time.rolling(window='1000D', closed='both', min_periods=10)[topic_names].mean()
topic_sent_time.iloc[40:].plot(cmap='Set1', figsize=(7,7))
plt.ylabel('Average topic sentiment')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()



# average sentiment per source
source_sent = df.groupby('source_agg').sentiment.mean().sort_values()
source_sent.plot(kind='barh')
plt.xlabel('Average sentiment')
plt.ylabel('Source')
plt.savefig(os.path.join(path, 'Plots', 'average-sentiment-across-sources'))
plt.show()

# average topic sentiment per source
source_topic_sent = df[['source_agg', 'sentiment'] + topic_names].copy()
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
        if entity == 'Europese Unie EU':
            entities.append('Europese Unie')
            continue
        if entity == 'CU':
            entities.append('ChristenUnie')
            continue
        if entity == 'Commissie':
            entities.append('Europese Commissie')
            continue
        if entity == 'CDUCSU':
            entities.append('CDU')
            continue
        if entity == 'CDU CSU':
            entities.append('CDU')
            continue
        if entity not in ['milieu', 'CO2', 'CO', 'kernenergie', 'Klimaat', 'Kamer', 'kernenergie', 'Rijk', 'België', 'Belgi', 'Milieu']:
            entities.append(entity)

    return entities
df['organizations'] = df.organizations.apply(fix_orgs)
df.organizations.explode().value_counts(ascending=True).tail(50).plot(kind='barh', figsize=(6,10))
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
        if entity == 'Franois Mitterrand':
            entities.append('François Mitterrand')
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
        if entity == 'Arjan Lubach':
            entities.append('Arjen Lubach')
            continue
        if entity not in ['Volt', 'Franciscus', 'Isral', 'God', 'Wubbo', 'Wise']:
            entities.append(entity)

    return entities

df['persons'] = df.persons.apply(fix_persons)

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


choices = df.persons.explode().dropna().unique()
name_dict = first_last_name_deduplication(choices)

df['persons'] = df.persons.explode().replace(name_dict).reset_index().groupby('index')['persons'].apply(list)
df.persons.explode().replace(name_dict).value_counts(ascending=True).tail(50).plot(kind='barh', figsize=(6,10))
plt.ylabel('Person')
plt.xlabel('Number of mentions')
plt.savefig(os.path.join(path, 'Plots', 'top-50-persons'))
plt.show()


## NER network
df['entities'] = df.persons + df.organizations # take both people and organizations

entities = df.entities.explode().value_counts() # rank entities by # of mentions
entities = entities[entities > 5] # take entities which appear more than 5 times
newspaper_names = list(df.source_agg.unique())
newspaper_names += ['De Volkskrant', 'Volkskrant', 'Telegraaf', 'FD', 'Financieele Dagblad']
entities = entities[entities.index.map(lambda x: x not in newspaper_names)] # remove newspapers from entities

entity_names = list(entities.index)

#### do NER network on article level COMMENT OUT IF NOT WANTED
# df = df.explode('entities').dropna(subset=['entities'])
# df.groupby(['text', 'source_agg']).entities.apply(list).reset_index(drop=True)

res = df.entities.explode().dropna()
res = res[res.apply(lambda x: x in entity_names)]

res = res.groupby(res.index).apply(set)
res = res[res.apply(len) > 1]

entities_co_occ = pd.DataFrame(0, index=entity_names, columns=entity_names)
for index, item in res.iteritems():
    pair = list(item)
    entities_co_occ.loc[pair[0], pair[1]] += 1
    entities_co_occ.loc[pair[1], pair[0]] += 1

# find topic loading for each entity
entity_topic_loadings = df.explode('entities').groupby('entities')[topic_names].mean()
entity_dom_topic = entity_topic_loadings.T.idxmax() # find dominant topic

# find average sentiment per entity
entity_sentiment = df.explode('entities').dropna(subset=['entities']).groupby('entities').sentiment.mean()

# find mentions by source per entity
entity_source_mentions = pd.get_dummies(df.explode('entities').dropna(subset=['entities']).set_index('entities').source_agg)
entity_source_mentions = entity_source_mentions.groupby('entities').sum()

# load into NetworkX
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
    data['Number of mentions'] = entities[entities.index == n].values[0]
    data['Dominant topic'] = str(entity_dom_topic.loc[n])
    entity_topic_loading = entity_topic_loadings.loc[n]
    entity_topic_loading = entity_topic_loading.sort_values(ascending=False).replace(0, np.nan).dropna() # sort and drop 0s
    data['Topic loadings'] = ', '.join([topic + ': ' + '{:.3f}'.format(loading) for topic, loading in entity_topic_loading.iteritems()])
    data['Average sentiment'] = entity_sentiment.loc[n]
    mentions_in_source = entity_source_mentions.loc['Tesla']
    mentions_in_source = mentions_in_source.sort_values(ascending=False).replace(0, np.nan).dropna() # sort and drop 0s
    data['Mentions in source'] = ', '.join([source + ': ' + str(int(mentions)) for source, mentions in mentions_in_source.iteritems()])

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