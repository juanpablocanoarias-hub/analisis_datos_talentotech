
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv("dataset_final.csv")

# Contar cuántas generadoras (recursos) hay por cada combustible
combustibles_count = df.groupby('Combustible')['Recurso'].nunique().sort_values(ascending=False)

# Crear gráfico de barras
fig, ax = plt.subplots(figsize=(12, 6))
palette = sns.color_palette("deep")

sns.barplot(x=combustibles_count.index, y=combustibles_count.values, ax=ax, palette=palette)
ax.set_title('Cantidad de Generadoras por Tipo de Combustible', fontsize=14, fontweight='bold')
ax.set_xlabel('Tipo de Combustible', fontsize=12)
ax.set_ylabel('Número de Generadoras', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()