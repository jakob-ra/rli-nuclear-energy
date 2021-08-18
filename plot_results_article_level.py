import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True
mpl.style.use('ggplot')

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

# import article level predictions
df_predictions = pd.read_pickle(os.path.join(path, 'rli-articles-with-topic-predictions.pkl'))

topic_names = ['Climate impact', 'Waste and storage', 'Geopolitics', 'Safety', 'Cost',
       'Technology', 'Ethics', 'Politics', 'Choice of site']

# plot overall topic prevalence
df_predictions[topic_names].sum().sort_values().plot(kind='barh')
plt.xlabel('Number of articles')
plt.ylabel('Topic')
plt.show()

# plot topic prevalence over time
topic_predictions_plot = df_predictions[['date'] + topic_names].copy()  # groupby time period and sum
topic_predictions_plot.set_index('date', inplace=True)
topic_predictions_plot.sort_index(inplace=True)
# topic_predictions_plot = topic_predictions_plot.rolling(window='365D', closed='both', min_periods=100).mean()
topic_predictions_plot = topic_predictions_plot.ewm(halflife='365D', times=topic_predictions_plot.index, min_periods=100).mean()
topic_predictions_plot = topic_predictions_plot.ewm(halflife='100D', times=topic_predictions_plot.index).mean()
topic_predictions_plot.plot(colormap='Set1', figsize=(7,7))
plt.xlabel('Year')
plt.ylabel('Share of articles mentioning topic')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(rotation=45, ha="right")
plt.savefig(os.path.join(path, 'Plots', 'topic-prominence-over-time'))
plt.show()

# plot topics per source
source_topics = df_predictions[['source'] + topic_names]
source_topics = source_topics.explode('source')
source_topics = source_topics.groupby('source').sum().div(source_topics.groupby('source').size(), axis=0)

source_topics.plot(figsize=(9,9), colormap='Set1', linestyle='None', marker='o', markersize=12, markeredgecolor='black')
plt.xlabel('Source')
plt.ylabel('Share of articles mentioning topic')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 14})
plt.xticks([i for i in range(len(source_topics.index))], source_topics.index, rotation=45, ha='right')
plt.savefig(os.path.join(path, 'Plots', 'topics-across-sources'))
plt.show()

