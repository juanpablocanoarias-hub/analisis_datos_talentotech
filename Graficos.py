import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("dataset_final.csv")
columnas_horarias = ['0','1','2','3','4','5','6','7','8','9','10','11','12',
                    '13','14','15','16','17','18','19','20','21','22','23','produccion_diaria']
for columna in columnas_horarias:
    if columna in df.columns:
        df[f'{columna}_GWh'] = df[columna] / 1e6

df['Fecha'] = pd.to_datetime(df['Fecha'])

df['mes'] = df['Fecha'].dt.month
df['dia_semana'] = df['Fecha'].dt.day_name()
df['semana_año'] = df['Fecha'].dt.isocalendar().week
df['trimestre'] = df['Fecha'].dt.quarter
df['dia_del_mes'] = df['Fecha'].dt.day


i=2
if i == 0:
    plt.figure(figsize=(15, 6))
    df.groupby('Fecha')['produccion_diaria'].sum().plot()
    plt.title('Producción Eléctrica Diaria - Año Completo')
    plt.ylabel('Producción (kWh)')
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    df['mes'] = df['Fecha'].dt.month
    df.groupby('mes')['produccion_diaria'].sum().plot(kind='bar')
    plt.title('Producción por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Producción Total (kWh)')
    plt.show()

    # Producción por día de la semana
    df['dia_semana'] = df['Fecha'].dt.day_name()
    produccion_diaria = df.groupby('dia_semana')['produccion_diaria'].mean()
    orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    produccion_diaria.loc[orden_dias].plot(kind='bar')
    plt.title('Producción Promedio por Día de la Semana')
    plt.show()

    plt.figure(figsize=(12, 6))
    df.groupby('0')['produccion_diaria'].sum().plot(kind='hist')
    plt.title('Patrón Horario de Producción (Promedio)')
    plt.xlabel('Hora del Día')
    plt.ylabel('Producción Promedio (kWh)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    produccion_mensual = df.groupby('mes')['produccion_diaria'].sum()
    produccion_mensual.plot(kind='bar')
    plt.title('Producción Eléctrica Mensual')
    plt.ylabel('Producción Total (MWh)')
    plt.xlabel('Mes')
    plt.xticks(rotation=0)
    plt.show()

plt.figure(figsize=(12, 6))
produccion_tipo = df.groupby('Tipo Generación')['produccion_diaria_GWh'].sum().sort_values(ascending=False)
bars = plt.bar(produccion_tipo.index, produccion_tipo.values, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Producción Total por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Total (GWh)')
plt.xlabel('Tipo de Generación')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
produccion_diaria_total = df.groupby('Fecha')['produccion_diaria_GWh'].sum()
plt.plot(produccion_diaria_total.index, produccion_diaria_total.values, 
         linewidth=1, color='steelblue')
plt.title('Evolución Diaria de la Producción Eléctrica', fontsize=14, fontweight='bold')
plt.ylabel('Producción Diaria (GWh)')
plt.xlabel('Fecha')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()