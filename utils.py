import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataframe(data, columns=None):
    """Carga un DataFrame desde una URL, archivo CSV, diccionario o lista de listas."""
    if isinstance(data, str):
        return pd.read_csv(data)
    return pd.DataFrame(data, columns=columns)

def analyze_dataframe(df):
    """Muestra información básica del DataFrame y valores únicos de variables categóricas."""
    print("\nResumen de información del DataFrame:")
    print(df.info())
    print("\nPrimeras filas del DataFrame:")
    print(df.head())
    print("\nDescripción estadística de las columnas numéricas:")
    print(df.describe())

    print("\nValores únicos por columna categórica:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\nValores únicos en '{col}':")
        print(df[col].unique())


def visualize_distribution(df, column):
    """Grafica la distribución de una columna numérica si existe."""
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribución de {column}')
        plt.show()
    else:
        print(f"La columna '{column}' no existe en el DataFrame.")

def correlation_matrix(df):
    """Muestra la matriz de correlación de un DataFrame numérico."""
    numeric_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all')
    if numeric_df.empty:
        print("No hay columnas numéricas en el DataFrame para calcular la correlación.")
        return
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.show()

# Crea fn para iterar por columnas categorícas vía loop
def unique_values(df):
    """Muestra los valores únicos de las columnas categóricas de un DataFrame."""
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\nValores únicos en '{col}':")
        print(df[col].unique())

# The marketing team wants to know the top 5 less common customer locations. 
# Create a pandas Series object that contains the customer locations and their frequencies, and then retrieve the top 5 less common locations in ascending order.
def less_common_locations(df):
    """Retorna las 5 ubicaciones menos comunes de los clientes."""
    locations = df['Customer Location'].value_counts()
    return locations.sort_values().head(5)