import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dortmund-data-r.csv', sep=';')
df = df[[
    'id', 'subject_group', 'token', 'answer_values', 'element_id',
    'hidden_figure', 'hidden_inference', 'hidden_maximum', 'hidden_minimum'
]]

# Extract all answer tasks
df['task'] = df['element_id'].apply(
    lambda x: True if x.startswith('agree') else False)
task_df = df.loc[df['task']].copy()

# Extract all yes-no-unknown tasks
def infer_binary(x):
    if x in ['yes', 'no', 'unknown']:
        return True
    return False
task_df['binary'] = task_df['answer_values'].apply(infer_binary)

binary_df = task_df.loc[task_df['binary']].drop([
    'hidden_maximum', 'hidden_minimum', 'task'
], axis=1)
interval_df = task_df.loc[task_df['binary'] == False]

# Sanity checks
print('----- Sanity -----')
print('No. binary tokens:', len(binary_df['token'].unique()))
print('No. interval tokens:', len(interval_df['token'].unique()))
print('------------------')
print()

# Binary data evaluation
def binary_correctness(x):
    if x.lower() == 'yes':
        return 1
    return 0
binary_df['correctness'] = binary_df['answer_values'].apply(binary_correctness)

# Create aggregated table
binary_agg = binary_df.groupby(
    ['hidden_inference', 'hidden_figure'], as_index=False)['correctness'].agg('mean')
table_data = []
for info, info_df in binary_agg.groupby('hidden_inference'):
    table_data.append({
        'type': info,
        'MP': info_df.loc[info_df['hidden_figure'] == 'MP']['correctness'].values[0],
        'MT': info_df.loc[info_df['hidden_figure'] == 'MT']['correctness'].values[0],
        'AC': info_df.loc[info_df['hidden_figure'] == 'AC']['correctness'].values[0],
        'DA': info_df.loc[info_df['hidden_figure'] == 'DA']['correctness'].values[0]
    })
table_df = pd.DataFrame(table_data)[['type', 'MP', 'MT', 'DA', 'AC']]
print('Aggregated answer rates (binary tasks):')
print(table_df)
print()

# Create the individual patterns
binary_agg_df = binary_df.groupby(
    ['token', 'hidden_inference', 'hidden_figure'], as_index=False).agg('mean')
binary_agg_df['correctness'] = binary_agg_df['correctness'].apply(lambda x: x > 0.6)

pattern_data = []
for info, info_df in binary_agg_df.groupby(['token', 'hidden_inference']):
    if len(info_df) != 4:
        print('ERROR: Invalid number of responses for {}-{} ({})'.format(
            info[0], info[1], len(info_df)))
        continue

    pattern_data.append({
        'token': info[0],
        'inference': info[1],
        'MP': info_df.loc[info_df['hidden_figure'] == 'MP']['correctness'].values[0],
        'MT': info_df.loc[info_df['hidden_figure'] == 'MT']['correctness'].values[0],
        'AC': info_df.loc[info_df['hidden_figure'] == 'AC']['correctness'].values[0],
        'DA': info_df.loc[info_df['hidden_figure'] == 'DA']['correctness'].values[0]
    })

pattern_df = pd.DataFrame(pattern_data)
pattern_df['rep'] = pattern_df[['MP', 'MT', 'AC', 'DA']].apply(
    lambda x: ''.join([str(int(y)) for y in x]), axis=1)

# Generate list of patterns
pattern_list = []
for idx in range(16):
    pattern_list.append('{:04b}'.format(idx))

inf_count = []
for info, info_df in pattern_df.groupby('inference'):
    dat = dict(collections.Counter(info_df['rep']))
    for key in pattern_list:
        value = 0
        if key in dat:
            value = dat[key]
        inf_count.append({'Inference': info, 'Pattern': key, 'Count': value})
inf_count_df = pd.DataFrame(inf_count)
inf_count_df = inf_count_df.replace(np.nan, 0)

def transform_code(x):
    print(x)
    result = "("
    for idx, value in enumerate(x):
        prefix = '$\\neg$'
        if value == '1':
            prefix = ''

        text = ['MP,', 'MT,', 'AC,', 'DA'][idx]
        result += prefix + text
    return result + ")"
inf_count_df['Pattern'] = inf_count_df['Pattern'].apply(transform_code)

sns.set()
sns.barplot(x='Pattern', y='Count', color='black', data=inf_count_df.loc[inf_count_df['Inference'] == 'Deduction'])
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Number of Occurence')
plt.subplots_adjust(bottom=0.35)
plt.savefig('image.pdf', dpi=300)
plt.show()
