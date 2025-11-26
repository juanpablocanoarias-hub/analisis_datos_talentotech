import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


pd.set_option('display.max_rows', 368)

path = 'C:\\Users\\juan.cano-a\\Desktop\\Generacion_(kWh).xlsx'
data = pd.read_excel(path, skiprows=2, sheet_name="Generacion_(kWh)")

print(data.columns.tolist())

cols = ["Recurso"]

print(data['Recurso'].value_counts())
print('\n')
print(data.isna().sum())
print('\n')
print(data.duplicated().sum())
print('\n')

data['produccion_diaria'] = data['0'] + data['1'] + data['2'] + data['3'] + data['4'] + data['5'] + data['6'] + data['7'] + data['8'] + data['9'] + data['10'] + data['11'] + data['12'] + data['13'] + data['14'] + data['15'] + data['16'] + data['17'] + data['18'] + data['19'] + data['20'] + data['21'] + data['22'] + data['23']

data.to_csv('dataset_limpia.csv', index = False)
