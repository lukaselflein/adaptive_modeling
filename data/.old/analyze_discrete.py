import collections

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

df = pd.read_csv('dortmund-data-r.csv', sep=';')
print(list(df))
#print(df.head())

df = df[['id', 'subject_group', 'token', 'answer_texts', 'answer_values', 'hidden_figure', 'hidden_inference']]

df['answer_values'] = df['answer_values'].apply(lambda x: x if x in ['yes', 'no', 'unknown'] else np.nan)
ids = df[['id', 'answer_values']]

ids = ids.dropna()
df = df.loc[df['id'].isin(ids['id'])]

def transform(x):
    if x == 'yes':
        return 1
    return 0

df['correctness'] = df['answer_values'].apply(transform)

# Aggregated patterns
res = df.groupby(['hidden_figure', 'hidden_inference'], as_index=False)['correctness'].agg('mean')
data = []
for info, inf_df in res.groupby('hidden_figure'):
    data.append({
        'figure': info,
        'deduction': inf_df.loc[inf_df['hidden_inference'] == 'Deduction']['correctness'].values[0],
        'induction': inf_df.loc[inf_df['hidden_inference'] == 'Induction']['correctness'].values[0],
        'abduction': inf_df.loc[inf_df['hidden_inference'] == 'Abduction']['correctness'].values[0]})

data_df = pd.DataFrame(data)
print()
print('Aggregated patterns:')
print(data_df[['figure', 'deduction', 'induction', 'abduction']])

# Individual Patterns
res = df.groupby(['token', 'hidden_figure', 'hidden_inference'], as_index=False)['correctness'].agg('mean')
res['correctness'] = res['correctness'].apply(lambda x: x > 0.6)
print(res.head())
inference_data = []
for info, inf_df in res.groupby(['token', 'hidden_inference']):
    if len(inf_df['hidden_figure'].unique()) != 4:
        print('ERROR in token', info[0])
        continue
    inference_data.append({
        'token': info[0],
        'inference': info[1],
        'MP': inf_df.loc[inf_df['hidden_figure'] == 'MP']['correctness'].values[0],
        'MT': inf_df.loc[inf_df['hidden_figure'] == 'MT']['correctness'].values[0],
        'AC': inf_df.loc[inf_df['hidden_figure'] == 'AC']['correctness'].values[0],
        'DA': inf_df.loc[inf_df['hidden_figure'] == 'DA']['correctness'].values[0]
        })
    # print(inference_data)
    # exit()

def convert_number(x):
    return ''.join([str(int(y)) for y in x])

inference_df = pd.DataFrame(inference_data)
inference_df['cnt'] = inference_df[['MP', 'MT', 'AC', 'DA']].apply(convert_number, axis=1)
print(inference_df.head())

inf_count = []
for info, info_df in inference_df.groupby('inference'):
    dat = dict(collections.Counter(info_df['cnt']))
    for key, value in dat.items():
        inf_count.append({'inference': info, 'pattern': key, 'count': value})
    # dat['inference'] = info
    # inf_count.append(dat)

inf_count_df = pd.DataFrame(inf_count)
inf_count_df = inf_count_df.replace(np.nan, 0)

print()
print('Individual patterns:')
print(inf_count_df.head())

# Enumerate pattern labels
pattern_labels = []
for idx in range(17):
    pattern_labels.append('{:04b}'.format(idx))

inf_count_df['pattern_idx'] = inf_count_df['pattern'].apply(lambda x: pattern_labels.index(x))
sns.barplot(x='pattern_idx', y='count', hue='inference', data=inf_count_df)
# plt.xticks(range(16))
# plt.xticks(range(17), pattern_labels, rotation=45)
plt.show()
exit()

for a in ['Deduction', 'Abduction', 'Induction']:
    sns.barplot(x='pattern', y='count', data=inf_count_df.loc[inf_count_df['inference'] == a])
    plt.title(a)
    plt.show()