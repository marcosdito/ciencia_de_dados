import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

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

# ==========================================
# NÚMERO DE GRUPOS
# ==========================================

print("Calculando Método do Cotovelo e Silhouette Score...")

inertia = []
silhouette_scores = []
K_range = range(2, 10) # Testando de 2 a 9 clusters

for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertia.append(kmeans_test.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_test.labels_))

# Plotar os Gráficos de Decisão
fig, ax1 = plt.subplots(figsize=(12, 6))

# Gráfico do Cotovelo (Inertia)
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inertia (Soma dos erros ao quadrado)', color='tab:blue')
ax1.plot(K_range, inertia, 'o-', color='tab:blue', label='Inertia (Cotovelo)')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Gráfico do Silhouette (Eixo secundário)
ax2 = ax1.twinx() 
ax2.set_ylabel('Silhouette Score', color='tab:orange')
ax2.plot(K_range, silhouette_scores, 's--', color='tab:orange', label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('Definição do K Ideal: Cotovelo vs Silhouette')
plt.grid(True)
plt.savefig('analise_k_ideal.png')
plt.close()
print("Gráfico salvo: analise_k_ideal.png")


# ==========================================
# 2. SELEÇÃO E PADRONIZAÇÃO DE DADOS
# ==========================================
# Vamos focar nas variáveis que mais definem perfil profissional
cols_cluster = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 
                'YearsAtCompany', 'YearsInCurrentRole', 
                'DistanceFromHome', 'NumCompaniesWorked']

# Filtrar apenas essas colunas para o cluster (remove ruído)
df_cluster_data = df[cols_cluster].dropna()

# Padronizar (StandardScaler é obrigatório para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster_data)

# ==========================================
# 4. APLICAÇÃO DO CLUSTER FINAL (Exemplo com K=3)
# ==========================================
# NOTA: Após olhar o gráfico acima, altere k_final se necessário.
k_final = 3 
print(f"Executando K-Means final com K={k_final}...")

kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Adicionar cluster ao dataframe original (usando o índice para alinhar)
df.loc[df_cluster_data.index, 'Cluster'] = clusters

# ==========================================
# 5. ANÁLISE DOS AGRUPAMENTOS E VARIÁVEIS
# ==========================================

# Boxplots para entender a influência de cada variável
plt.figure(figsize=(18, 10))
for i, col in enumerate(cols_cluster):
    plt.subplot(2, 4, i+1)
    sns.boxplot(x='Cluster', y=col, data=df, palette='viridis')
    plt.title(col)
plt.tight_layout()
plt.savefig('influencia_variaveis.png')
plt.close()
print("Gráfico salvo: influencia_variaveis.png")

# Tabela Resumo dos Perfis (Média das variáveis por cluster)
summary = df.groupby('Cluster')[cols_cluster + ['Attrition_Numeric']].mean()
# Adiciona contagem de pessoas por cluster
summary['Contagem'] = df['Cluster'].value_counts()
print("\n=== PERFIL MÉDIO DOS CLUSTERS ===")
print(summary.T) # Transposto para facilitar leitura