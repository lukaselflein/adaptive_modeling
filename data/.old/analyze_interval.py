import pandas as pd
import numpy as np

df = pd.read_csv('dortmund-data-r.csv', sep=';')
print(list(df))
# print(df.head())

df = df[[
    'id', 'subject_group', 'token', 'answer_texts', 'answer_values',
    'correctness', 'hidden_figure', 'hidden_inference', 'hidden_interval',
    'hidden_prediction_logic'
]]

# Drop all answers containing yes/no/unknown
id_df = df[['id', 'answer_values']]
id_df = id_df.replace('yes', np.nan)
id_df = id_df.replace('no', np.nan)
id_df = id_df.replace('unknown', np.nan)
id_df = id_df.dropna()
print(id_df['answer_values'].unique())
