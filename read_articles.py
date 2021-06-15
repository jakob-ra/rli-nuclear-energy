import pandas as pd
import os
from striprtf.striprtf import rtf_to_text
from tqdm import tqdm
import dateparser

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

# read in raw data
data_folders = os.listdir(os.path.join(path, 'Data'))

docs = []
for folder in data_folders:
    files = os.listdir(os.path.join(path, 'Data', folder))
    for file in tqdm(files):
        doc = rtf_to_text(open(os.path.join(path, 'Data', folder, file)).read())
        docs.append(doc)

df = pd.DataFrame(docs, columns=['raw'])

df['raw'] = df.raw.str.replace(u'\xa0', u' ')

# process raw documents piece by piece
df['raw'] = df.raw.str.strip('\n\r ')
df['source'] = df.raw.apply(lambda x: x.split('\n')[0])
df['date'] = df.raw.apply(lambda x: x.split('\n')[1])
df['date'] = df.date.apply(lambda x: ' '.join(x.split()[:3])).apply(dateparser.parse)
df['copyright'] = df.raw.apply(lambda x: x.split('\n\n\n')[1])
df['text'] = df.raw.apply(lambda x: x.split('\n\n\n')[3])
df['text'] = df.text.apply(lambda x: x.split('Load-Date: ')[0])

df.to_excel(os.path.join(path, 'rli-articles-clean.xlsx'), index=False)


# df['byline'] = '' # byline does not always exist
# df.loc[df.raw.str.contains('Byline:'), 'byline'] = df.raw[df.raw.str.contains('Byline:')].apply(lambda x: x.split('Byline: ')[1])
# df.loc[df.byline != '', 'byline'] = df[df.byline != ''].byline.apply(lambda x: x.split('\n')[0])

# df['section'] = df.raw.apply(lambda x: x.split(':')[1])
# df['section'] = df.section.apply(lambda x: x.split(';')[0])
#
# df['words'] = df.raw.apply(lambda x: x.split(':')[2])
# df['words'] = df.words.apply(lambda x: x.split(' words')[0])