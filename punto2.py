
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("dataset_final.csv")#Cargando los datos como df(dataframe) yolanda

# se cuenta de manera unica cada recurso por tipo de combustible
combustibles_count = df.groupby('Combustible')['Recurso'].nunique().sort_values(ascending=False)

#El grafico de barras con matplotlib y seaborn
fig, ax = plt.subplots(figsize=(12, 6))#Tamaño del grafico
palette = sns.color_palette("deep")#Paleta de colores para el grafico

sns.barplot(x=combustibles_count.index, y=combustibles_count.values, ax=ax, palette=palette)#es cogemos el tipo de grafico, Grafico de barras
ax.set_title('Cantidad de Generadoras por Tipo de Combustible', fontsize=14, fontweight='bold')#Titulo del grafico
ax.set_xlabel('Tipo de Combustible', fontsize=12)#Etiqueta del eje x
ax.set_ylabel('Número de Generadoras', fontsize=12)#Etiqueta del eje y
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')#Rotacion de las etiquetas del eje x para mejor visibilidad

plt.tight_layout()#Ajuste del diseño para evitar recortes
plt.show()#Mostramos el griafco