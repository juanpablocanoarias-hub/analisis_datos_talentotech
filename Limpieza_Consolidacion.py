import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = 'CSV\Generacion_(kWh).xlsx'
data_frame_raw = pd.read_excel(path, skiprows=2, sheet_name="Generacion_(kWh)")

print(data_frame_raw.columns.tolist())

cols = ["Recurso"]

print(data_frame_raw['Recurso'].value_counts())
print('\n')
print(data_frame_raw.isna().sum())
print('\n')
print(data_frame_raw.duplicated().sum())
print('\n')

data_frame_raw['produccion_diaria'] = data_frame_raw['0'] + data_frame_raw['1'] + data_frame_raw['2'] + data_frame_raw['3'] + data_frame_raw['4'] + data_frame_raw['5'] + data_frame_raw['6'] + data_frame_raw['7'] + data_frame_raw['8'] + data_frame_raw['9'] + data_frame_raw['10'] + data_frame_raw['11'] + data_frame_raw['12'] + data_frame_raw['13'] + data_frame_raw['14'] + data_frame_raw['15'] + data_frame_raw['16'] + data_frame_raw['17'] + data_frame_raw['18'] + data_frame_raw['19'] + data_frame_raw['20'] + data_frame_raw['21'] + data_frame_raw['22'] + data_frame_raw['23']

data_frame_raw.to_csv('data_frame_rawset_limpia.csv', index = False)

#Inicio 3.1.py


path2 = 'CSV\Listado_Recursos_Generacion.xlsx'
location = pd.read_excel(path2, skiprows=3, sheet_name="Listado_Recursos_Generacion")



data_location = data_frame_raw.merge(
    location[["Código SIC", "Municipio", "Departamento"]],
    left_on='Código Recurso',
    right_on='Código SIC',
    how='left'
)

data_location.to_csv('dataset_location.csv', index = False)

#Inicio 3.2.py

