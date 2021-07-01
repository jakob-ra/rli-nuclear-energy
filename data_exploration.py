import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import spacy
from wordcloud import WordCloud

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True
mpl.style.use('ggplot')

# plt.set_cmap(sns.diverging_palette(220, 20, as_cmap=True))

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_excel(os.path.join(path, 'rli-articles-clean.xlsx'))

# Number of articles by source
df.groupby('source_agg')['text'].count().sort_values().plot(kind='barh')
plt.ylabel('Source')
plt.xlabel('Number of articles')
# plt.xticks(rotation=60, ha="right")
plt.savefig(os.path.join(path, 'Plots', 'number-of-articles-by-source'))
plt.show()

# Number of articles over time
df.groupby(df.date.dt.to_period('Y'))['text'].count().plot.bar()
plt.xlabel('Year')
plt.ylabel('Number of articles')
plt.xticks(rotation=45, ha="right")
plt.savefig(os.path.join(path, 'Plots', 'number-of-articles-over-time'))
plt.show()

# Length of articles
cutoff, tick_interval = 3000, 500
df['plot_length'] = df.length
df.loc[df.plot_length > cutoff, 'plot_length'] = cutoff
df.plot_length.hist(bins=50)
ticks = [i*tick_interval for i in range(cutoff//tick_interval + 1)]
plt.xticks(ticks=ticks, labels=ticks[:-1] + ['>' + str(cutoff)])
plt.ylabel('Number of articles')
plt.xlabel('Article length in words')
plt.savefig(os.path.join(path, 'Plots', 'articles-by-length'))
plt.show()

# word cloud
nlp = spacy.load("nl_core_news_sm")
stopwords = nlp.Defaults.stop_words
wordcloud = WordCloud(
    #background_color='white',
    colormap='Accent',
    max_words=100,
    max_font_size=80,
    width=1000, height=700,
    stopwords=stopwords).generate(' '.join(df.text))

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig(os.path.join(path, 'Plots', 'word-cloud-overall.png'), bbox_inches='tight')
plt.show()