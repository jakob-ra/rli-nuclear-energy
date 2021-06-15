import matplotlib.pyplot as plt

df.groupby(df.date.dt.to_period('M'))['text'].count().plot.bar()
plt.show()