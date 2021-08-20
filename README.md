This is the code for the Dutch media analysis on nuclear energy for RLI.

Download the data folder from Dropbox, specify the correct path and run the code in this order to replicate results:
1) read_articles.py - reads and consolidates the raw articles, can also skip and continue with rli-articles-clean.xlsx instead
2) data_exploration.py - produces explorative plots of the data
3) google_batch_translate.py - needs google cloud translation credentials, can also skip and continue with rli-sentencs-plus-translation.pkl instead
4) sentiment_analysis.py - Spacy sentiment analysis on translated sentences
5) named-entity-recognition.py - Finds organizations, people, locations, and miscellaneous entities using Dutch model on original (Dutch) models
6) topic_analysis.py - Use anchored topic model to predict topics both on sentence and article level
7) plot_results_article_level.py - plots article level results
8) plot_results_sentence_level.py - plots sentence level results and produces an entity network that was next processed in Gephi. 
The resulting network can be found under 'rli-entity-network.gephi'

