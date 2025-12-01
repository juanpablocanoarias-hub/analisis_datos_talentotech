import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

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

horas_columns = [f'{j}_GWh' for j in range(24)]
produccion_promedio_hora = df[horas_columns].mean()

plt.figure(figsize=(14, 8))
plt.imshow([produccion_promedio_hora.values], cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Producción Promedio (GWh)')
plt.title('Producción Promedio por Hora del Día', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día')
plt.yticks([])
plt.xticks(range(24))
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
tipos_generacion = df['Tipo Generación'].unique()
for i, tipo in enumerate(tipos_generacion):
    df_tipo = df[df['Tipo Generación'] == tipo]
    perfil_horario = df_tipo[horas_columns].sum()
    plt.plot(range(24), perfil_horario.values, marker='o', label=tipo, linewidth=2)
plt.title('Perfil Horario Promedio por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Hora del Día')
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
tipos_generacion = df['Tipo Generación'].unique()
for i, tipo in enumerate(tipos_generacion):
    df_tipo = df[df['Tipo Generación'] == tipo]
    perfil_horario = df_tipo[horas_columns].mean()
    plt.plot(range(24), perfil_horario.values, marker='o', label=tipo, linewidth=2)
plt.title('Perfil Horario Promedio por Tipo de Generación', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Hora del Día')
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
produccion_dia_semana = df.groupby('dia_semana')['produccion_diaria_GWh'].mean()
dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dias_esp = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
produccion_dia_semana = produccion_dia_semana.reindex(dias_orden)
plt.bar(dias_esp, produccion_dia_semana.values, color='lightcoral', alpha=0.8)
plt.title('Producción Promedio por Día de la Semana', fontsize=14, fontweight='bold')
plt.ylabel('Producción Promedio (GWh)')
plt.xlabel('Día de la Semana')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

produccion_mes_combustible = df.groupby(['mes', 'Combustible'])['produccion_diaria_GWh'].sum().unstack().fillna(0)
top_combustibles = df.groupby('Combustible')['produccion_diaria_GWh'].sum().nlargest(13).index
produccion_mes_combustible = produccion_mes_combustible[top_combustibles]
produccion_mes_combustible.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('Producción Mensual por Combustible', fontsize=14, fontweight='bold')
plt.ylabel('Producción (GWh)')
plt.xlabel('Mes')
plt.xticks(range(12), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=0)
plt.legend(title='Combustible', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

produccion_departamento = df.groupby('Departamento')['produccion_diaria_GWh'].sum().sort_values(ascending=False)
produccion_municipio = df.groupby(['Departamento', 'Municipio'])['produccion_diaria_GWh'].sum().reset_index()
top_departamentos = produccion_departamento.head(15)
top_municipios = produccion_municipio.nlargest(15, 'produccion_diaria_GWh')

plt.figure(figsize=(14, 8))
bars = plt.barh(top_departamentos.index, top_departamentos.values, color='teal')
plt.title('Top 15 Departamentos por Producción Eléctrica', fontsize=16, fontweight='bold')
plt.xlabel('Producción Total (GWh)')
plt.ylabel('Departamento')
plt.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:,.0f}', ha='left', va='center', fontweight='bold')
plt.tight_layout()
plt.show()

df_plot = df.groupby(['Fecha', 'Tipo Generación'])['produccion_diaria_GWh'].sum().unstack()
plt.figure(figsize=(14, 8))
for tipo in df_plot.columns:
    plt.plot(df_plot.index, df_plot[tipo], label=tipo, linewidth=2)
plt.title('Evolución de Producción por Tipo')
plt.ylabel('Producción (GWh)')
plt.xlabel('Fecha')
plt.legend(title='Tipo Generación', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig = px.line(df.groupby(['Fecha', 'Tipo Generación'])['produccion_diaria_GWh'].sum().reset_index(),
              x='Fecha', y='produccion_diaria_GWh', color='Tipo Generación',
              title='Evolución de Producción por Tipo (Interactivo)')
plot(fig, filename='produccion_por_tipo.html', auto_open=True)

#fig = px.line(df.groupby(['Fecha', 'Tipo Generación'])['produccion_diaria_GWh'].sum().reset_index(),
#              x='Fecha', y='produccion_diaria_GWh', color='Tipo Generación',
#              title='Evolución de Producción por Tipo (Interactivo)')
#fig.show()

"""Genera un resumen del dataset con la estructura actual"""
print("RESUMEN - DATOS DIARIOS")
print("=" * 50)
print(f"Período: {df['Fecha'].min().strftime('%d/%m/%Y')} a {df['Fecha'].max().strftime('%d/%m/%Y')}")
print(f"Días analizados: {df['Fecha'].nunique()}")
print(f"Producción total: {df['produccion_diaria_GWh'].sum():,.1f} GWh")
print(f"Producción promedio diaria: {df['produccion_diaria_GWh'].mean():.1f} GWh")
print(f"Centrales eléctricas: {df['Recurso'].nunique()}")

print("\n Tipos de generación:")
for tipo, prod in df.groupby('Tipo Generación')['produccion_diaria_GWh'].sum().items():
    porcentaje = (prod / df['produccion_diaria_GWh'].sum()) * 100
    print(f"  - {tipo}: {prod:,.1f} GWh ({porcentaje:.1f}%)")

