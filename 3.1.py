import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


pd.set_option('display.max_rows', 368)

path = 'C:\\Users\\juan.cano-a\\Documents\\Coding\\dataset_limpia.csv'
data = pd.read_csv(path)

path2 = 'C:\\Users\\juan.cano-a\\Desktop\\Listado_Recursos_Generacion.xlsx'
location = pd.read_excel(path2, skiprows=3, sheet_name="Listado_Recursos_Generacion")

#print(location.columns.tolist())

data_loc = data.merge(
    location[["Código SIC", "Municipio", "Departamento"]],
    left_on='Código Recurso',
    right_on='Código SIC',
    how='left'
)

data_loc.to_csv('dataset_loc.csv', index = False)