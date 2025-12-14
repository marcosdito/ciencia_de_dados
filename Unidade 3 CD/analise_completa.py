import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuração de estilo
sns.set(style="whitegrid")

# 1. Carregar os dados
file_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
print(f"Carregando arquivo: {file_path}")
df = pd.read_csv(file_path)

# 2. Pré-processamento básico
# Converter 'Attrition' para numérico para ver correlação (Yes=1, No=0)
df['Attrition_Numeric'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Selecionar apenas colunas numéricas para análise
df_numeric = df.select_dtypes(include=[np.number])

# Remover colunas constantes ou irrelevantes (EmployeeCount, StandardHours geralmente são fixos)
df_numeric = df_numeric.drop(['EmployeeCount', 'StandardHours'], axis=1, errors='ignore')

print("Dados pré-processados.")

# ==========================================
# ANÁLISE DE CORRELAÇÃO
# ==========================================
print("Gerando Análise de Correlação...")
plt.figure(figsize=(16, 12))
corr_matrix = df_numeric.corr()

# Gerar Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Mascara para triangular superior
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Mapa de Calor de Correlação (Variáveis Numéricas)', fontsize=16)
plt.tight_layout()
plt.savefig('resultado_correlacao.png')
print("Gráfico salvo: resultado_correlacao.png")
plt.close()

# Mostrando as correlações mais fortes com Attrition
print("\nTop 5 correlações com Attrition (Rotatividade):")
print(corr_matrix['Attrition_Numeric'].sort_values(ascending=False).head(6))
print("-" * 30)


# ==========================================
# CLUSTERING (K-MEANS)
# ==========================================
print("Executando Clustering (K-Means)...")

# Padronizar os dados (essencial para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric.dropna())

# Aplicar K-Means (vamos assumir 3 grupos de perfis)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar clusters ao dataframe original (nas linhas correspondentes)
df_clustered = df_numeric.dropna().copy()
df_clustered['Cluster'] = clusters

# Visualização dos Clusters usando PCA (reduzindo para 2D para plotar)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('Visualização dos Clusters de Funcionários (PCA 2D)', fontsize=16)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter, label='Cluster ID')
plt.savefig('resultado_clustering.png')
print("Gráfico salvo: resultado_clustering.png")
plt.close()

# Analisar características médias de cada cluster
print("\nPerfil Médio dos Clusters (Principais Variáveis):")
cluster_summary = df_clustered.groupby('Cluster')[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'Attrition_Numeric']].mean()
print(cluster_summary)

print("\nConcluído. Verifique os arquivos PNG gerados.")
