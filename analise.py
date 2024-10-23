# Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
file_path = './br_ibge_censo_2022_indigenas_populacao_alfabetizada_grupo_idade_municipio.csv'

# Tente carregar o arquivo CSV
try:
    data = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    print("Arquivo CSV carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {file_path} não foi encontrado.")
except pd.errors.ParserError:
    print(f"Erro ao analisar o arquivo CSV. Verifique o delimitador e a formatação.")
    raise
except Exception as e:
    print(f"Erro inesperado: {e}")
    raise

# Limpando os dados (separando corretamente as colunas)
try:
    # Dividindo as colunas misturadas por vírgula
    data[['id_municipio', 'sexo', 'grupo_idade', 'alfabetizacao', 'populacao_indigena']] = data['id_municipio,sexo,grupo_idade,alfabetizacao,populacao_indigena'].str.split(',', expand=True)
    data.drop('id_municipio,sexo,grupo_idade,alfabetizacao,populacao_indigena', axis=1, inplace=True)
    
    # Convertendo a coluna 'populacao_indigena' para numérica
    data['populacao_indigena'] = pd.to_numeric(data['populacao_indigena'], errors='coerce')
    print("Dados limpos e prontos para análise!")
except KeyError as e:
    print(f"Erro ao acessar colunas: {e}")
    print("Verifique se os nomes das colunas no CSV estão corretos.")
    raise
except Exception as e:
    print(f"Erro inesperado durante a limpeza dos dados: {e}")
    raise

# Funções adicionais para análise e pré-processamento
def explore_data(data):
    """Função para explorar o conjunto de dados e verificar valores ausentes, tipos de dados e estatísticas básicas."""
    print("\nResumo dos dados:")
    print(data.info())
    print("\nValores ausentes por coluna:")
    print(data.isnull().sum())
    print("\nEstatísticas descritivas:")
    print(data.describe(include='all'))

def treat_missing_values(data):
    """Tratamento de valores ausentes, incluindo possíveis preenchimentos ou remoção."""
    # Exemplo: preenchendo valores ausentes com a mediana
    data['populacao_indigena'] = data['populacao_indigena'].fillna(data['populacao_indigena'].median())
    print("Valores ausentes tratados!")

def identify_outliers(data):
    """Identificar outliers na coluna de população indígena."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data['populacao_indigena'], color='skyblue', flierprops={'markerfacecolor':'red', 'markersize':6})
    plt.title("Boxplot para Identificação de Outliers na População Indígena\n", fontsize=14)
    plt.xlabel("População Indígena", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("boxplot_outliers_melhorado.png")
    plt.show()

def transform_variables(data):
    """Transformações adicionais, se necessário."""
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
    data['faixa_etaria'] = pd.cut(pd.to_numeric(data['grupo_idade'].str.extract(r'(\d+)')[0], errors='coerce'), bins=age_bins, labels=age_labels)
    print("Transformações aplicadas!")

def plot_distribution(data):
    """Plotar a distribuição da população indígena."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data['populacao_indigena'], kde=True, bins=30, color='purple')
    plt.title("Distribuição da População Indígena Alfabetizada", fontsize=14)
    plt.xlabel("População Indígena Alfabetizada", fontsize=12)
    plt.ylabel("Frequência", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("distribuicao_populacao_indigena.png")
    plt.show()

def plot_grouped_bars(data):
    """Gráfico de barras agrupadas por faixa etária e sexo."""
    grouped_data = data.groupby(['faixa_etaria', 'sexo'])['populacao_indigena'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped_data, x='faixa_etaria', y='populacao_indigena', hue='sexo', palette='Set2')
    plt.title("Média da População Indígena Alfabetizada por Faixa Etária e Sexo", fontsize=14)
    plt.xlabel("Faixa Etária", fontsize=12)
    plt.ylabel("Média de População Alfabetizada", fontsize=12)
    plt.legend(title="Sexo")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("barras_agrupadas_faixa_etaria_sexo.png")
    plt.show()

# Criando o modelo, controle e visão
class CensusModel:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        """Retorna os dados brutos"""
        return self.data

class CensusController:
    def __init__(self, model):
        self.model = model

    def calculate_statistics(self):
        """Calcula estatísticas descritivas, como média, mediana, moda e variância."""
        stats = {
            "media": self.model.get_data()['populacao_indigena'].mean(),
            "mediana": self.model.get_data()['populacao_indigena'].median(),
            "moda": self.model.get_data()['populacao_indigena'].mode()[0],
            "variancia": self.model.get_data()['populacao_indigena'].var(),
        }
        return stats

    def calculate_correlation(self):
        """Calcula a correlação entre variáveis (apenas numéricas)."""
        numeric_data = self.model.get_data().select_dtypes(include=['float64', 'int64'])
        return numeric_data.corr()

class CensusView:
    def __init__(self, controller):
        self.controller = controller

    def display_statistics(self):
        """Exibe estatísticas descritivas básicas."""
        stats = self.controller.calculate_statistics()
        print("Estatísticas Descritivas:")
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value}")

    def display_correlation(self):
        """Exibe uma matriz de correlação."""
        plt.figure(figsize=(10, 8))
        correlation = self.controller.calculate_correlation()
        sns.heatmap(correlation, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={"shrink": 0.8}, linewidths=.5)
        plt.title("Matriz de Correlação entre Variáveis Numéricas\n", fontsize=14)
        plt.savefig("matriz_correlacao_melhorada.png")
        plt.show()

# Aplicando novas funções
explore_data(data)
treat_missing_values(data)
identify_outliers(data)
transform_variables(data)
plot_distribution(data)
plot_grouped_bars(data)

# Criando o modelo, controle e visão para novas análises
model = CensusModel(data)
controller = CensusController(model)
view = CensusView(controller)

view.display_statistics()
view.display_correlation()
