"""
Módulo para el modelo de clustering de jugadores por estilo de juego.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza las características para clustering.
    
    Args:
        X: DataFrame con características
        
    Returns:
        DataFrame con características escaladas
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info(f"Características escaladas: media ≈ 0, desviación estándar ≈ 1")
    return X_scaled

def reduce_dimensions_pca(X: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, PCA]:
    """
    Reduce la dimensionalidad de las características usando PCA.
    
    Args:
        X: DataFrame con características
        n_components: Número de componentes a mantener
        
    Returns:
        Tupla con (DataFrame reducido, modelo PCA)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Crear DataFrame con componentes
    columns = [f'PC{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=columns, index=X.index)
    
    # Calcular varianza explicada
    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA: {n_components} componentes explican {explained_variance:.2%} de la varianza")
    
    return X_pca_df, pca

def reduce_dimensions_tsne(X: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> pd.DataFrame:
    """
    Reduce la dimensionalidad de las características usando t-SNE.
    
    Args:
        X: DataFrame con características
        n_components: Número de componentes a mantener
        perplexity: Parámetro de equilibrio entre estructura local y global
        
    Returns:
        DataFrame reducido
    """
    tsne = TSNE(
        n_components=n_components, 
        random_state=42, 
        perplexity=perplexity,
        n_jobs=-1  # Usar todos los núcleos disponibles
    )
    X_tsne = tsne.fit_transform(X)
    
    # Crear DataFrame con componentes
    columns = [f'tSNE{i+1}' for i in range(n_components)]
    X_tsne_df = pd.DataFrame(X_tsne, columns=columns, index=X.index)
    
    logger.info(f"t-SNE: Datos reducidos a {n_components} dimensiones con perplexity={perplexity}")
    return X_tsne_df

def reduce_dimensions_umap(X: pd.DataFrame, n_components: int = 2, 
                          n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    """
    Reduce la dimensionalidad de las características usando UMAP.
    
    Args:
        X: DataFrame con características
        n_components: Número de componentes a mantener
        n_neighbors: Número de vecinos para equilibrar estructura local vs global
        min_dist: Distancia mínima entre puntos en espacio de baja dimensión
        
    Returns:
        DataFrame reducido
    """
    reducer = umap.UMAP(
        n_components=n_components, 
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    X_umap = reducer.fit_transform(X)
    
    # Crear DataFrame con componentes
    columns = [f'UMAP{i+1}' for i in range(n_components)]
    X_umap_df = pd.DataFrame(X_umap, columns=columns, index=X.index)
    
    logger.info(f"UMAP: Datos reducidos a {n_components} dimensiones")
    return X_umap_df

def find_optimal_k(X: pd.DataFrame, max_k: int = 10) -> Tuple[int, Dict]:
    """
    Encuentra el número óptimo de clusters usando el método del codo y silueta.
    
    Args:
        X: DataFrame con características
        max_k: Número máximo de clusters a probar
        
    Returns:
        Tupla con (k óptimo, métricas por k)
    """
    results = {}
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calcular métricas
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0
        
        results[k] = {
            'inertia': inertia,
            'silhouette': silhouette,
            'calinski_harabasz': calinski
        }
        
        logger.info(f"k={k}: Inercia={inertia:.2f}, Silueta={silhouette:.4f}, CH={calinski:.2f}")
    
    # Determinar k óptimo según silueta (mayor valor)
    silhouette_values = [results[k]['silhouette'] for k in k_range]
    optimal_k = k_range[np.argmax(silhouette_values)]
    
    logger.info(f"K óptimo según silueta: {optimal_k}")
    return optimal_k, results

def apply_kmeans(X: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.Series, KMeans]:
    """
    Aplica el algoritmo K-Means para agrupar los datos.
    
    Args:
        X: DataFrame con características
        n_clusters: Número de clusters
        
    Returns:
        Tupla con (etiquetas de cluster, modelo K-Means)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Convertir a Series para mantener el índice
    cluster_series = pd.Series(cluster_labels, index=X.index, name='cluster')
    
    # Calcular métricas
    silhouette = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
    
    logger.info(f"K-Means aplicado con {n_clusters} clusters. Silhouette score: {silhouette:.4f}")
    
    # Calcular tamaños de los clusters
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        logger.info(f"Cluster {cluster}: {size} elementos ({size/len(X):.1%})")
    
    return cluster_series, kmeans

def analyze_clusters(X: pd.DataFrame, cluster_labels: pd.Series) -> pd.DataFrame:
    """
    Analiza las características de cada cluster.
    
    Args:
        X: DataFrame con características originales
        cluster_labels: Serie con etiquetas de cluster
        
    Returns:
        DataFrame con estadísticas por cluster
    """
    # Combinar características con etiquetas
    df_combined = X.copy()
    df_combined['cluster'] = cluster_labels
    
    # Calcular estadísticas por cluster
    cluster_stats = df_combined.groupby('cluster').agg(
        ['mean', 'std', 'min', 'max']
    )
    
    # Calcular características más distintivas por cluster
    cluster_means = df_combined.groupby('cluster').mean()
    global_means = X.mean()
    
    # Calcular cuánto se desvía cada cluster de la media global
    distinctive_features = pd.DataFrame()
    for cluster in cluster_means.index:
        # Normalizar por desviación estándar global para comparabilidad
        std_dev = X.std()
        # Evitar división por cero
        std_dev = std_dev.replace(0, 1)  
        deviation = (cluster_means.loc[cluster] - global_means) / std_dev
        distinctive_features[f'cluster_{cluster}'] = deviation
    
    logger.info("Análisis de clusters completado")
    
    return cluster_stats, distinctive_features

def plot_clusters_2d(X_reduced: pd.DataFrame, cluster_labels: pd.Series, 
                    title: str = 'Visualización de Clusters',
                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualiza los clusters en 2D.
    
    Args:
        X_reduced: DataFrame con datos reducidos a 2 dimensiones
        cluster_labels: Serie con etiquetas de cluster
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    if X_reduced.shape[1] != 2:
        logger.warning("Esta función espera datos reducidos a 2 dimensiones")
        return
    
    plt.figure(figsize=figsize)
    
    # Obtener nombres de columnas para ejes
    x_col, y_col = X_reduced.columns
    
    # Crear scatter plot con colores por cluster
    sns.scatterplot(
        x=x_col, 
        y=y_col,
        hue=cluster_labels,
        palette='viridis',
        data=pd.concat([X_reduced, cluster_labels], axis=1),
        legend='full',
        alpha=0.8
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance_by_cluster(distinctive_features: pd.DataFrame, 
                                     top_n: int = 10,
                                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza las características más importantes para cada cluster.
    
    Args:
        distinctive_features: DataFrame con desviaciones normalizadas
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
    """
    n_clusters = distinctive_features.shape[1]
    
    plt.figure(figsize=figsize)
    
    # Crear subplots para cada cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize, sharex=True)
    if n_clusters == 1:
        axes = [axes]
    
    for i, cluster_col in enumerate(distinctive_features.columns):
        # Ordenar características por importancia absoluta
        top_features = distinctive_features[cluster_col].abs().nlargest(top_n)
        # Obtener valores originales para estas características
        top_values = distinctive_features[cluster_col][top_features.index]
        
        # Crear barplot
        ax = axes[i]
        colors = ['red' if x < 0 else 'green' for x in top_values]
        top_values.sort_values().plot(kind='barh', ax=ax, color=colors)
        
        ax.set_title(f'Características distintivas para {cluster_col}')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_cluster_sizes(cluster_labels: pd.Series, 
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualiza el tamaño de cada cluster.
    
    Args:
        cluster_labels: Serie con etiquetas de cluster
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Contar elementos por cluster
    cluster_counts = cluster_labels.value_counts().sort_index()
    
    # Crear gráfico de barras
    ax = cluster_counts.plot(kind='bar', color='skyblue')
    
    # Añadir etiquetas con el número exacto y porcentaje
    total = len(cluster_labels)
    for i, count in enumerate(cluster_counts):
        percentage = count / total * 100
        ax.text(i, count + 0.1, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom')
    
    plt.title('Distribución de elementos por cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Número de elementos')
    plt.tight_layout()
    plt.show()

def run_clustering_analysis(X: pd.DataFrame, 
                           n_clusters: Optional[int] = None,
                           max_k: int = 10,
                           reduction_method: str = 'pca',
                           analyze_features: bool = True) -> Dict:
    """
    Ejecuta el análisis completo de clustering.
    
    Args:
        X: DataFrame con características
        n_clusters: Número de clusters (si es None, se determina automáticamente)
        max_k: Máximo número de clusters a probar para encontrar el óptimo
        reduction_method: Método de reducción de dimensionalidad ('pca', 'tsne', 'umap')
        analyze_features: Si se analizan las características distintivas de cada cluster
        
    Returns:
        Diccionario con resultados del clustering
    """
    results = {}
    
    # 1. Escalar características
    logger.info("Paso 1: Escalando características...")
    X_scaled = scale_features(X)
    results['X_scaled'] = X_scaled
    
    # 2. Encontrar número óptimo de clusters si no se especifica
    if n_clusters is None:
        logger.info("Paso 2: Encontrando número óptimo de clusters...")
        optimal_k, k_metrics = find_optimal_k(X_scaled, max_k=max_k)
        n_clusters = optimal_k
        results['k_metrics'] = k_metrics
    
    logger.info(f"Utilizando {n_clusters} clusters para el análisis")
    
    # 3. Aplicar K-Means
    logger.info("Paso 3: Aplicando K-Means...")
    cluster_labels, kmeans_model = apply_kmeans(X_scaled, n_clusters=n_clusters)
    results['cluster_labels'] = cluster_labels
    results['kmeans_model'] = kmeans_model
    
    # 4. Reducir dimensionalidad para visualización
    logger.info(f"Paso 4: Reduciendo dimensionalidad con {reduction_method}...")
    if reduction_method == 'pca':
        X_reduced, pca_model = reduce_dimensions_pca(X_scaled)
        results['X_reduced'] = X_reduced
        results['reduction_model'] = pca_model
    elif reduction_method == 'tsne':
        X_reduced = reduce_dimensions_tsne(X_scaled)
        results['X_reduced'] = X_reduced
    elif reduction_method == 'umap':
        X_reduced = reduce_dimensions_umap(X_scaled)
        results['X_reduced'] = X_reduced
    else:
        logger.warning(f"Método de reducción {reduction_method} no reconocido. Usando PCA.")
        X_reduced, pca_model = reduce_dimensions_pca(X_scaled)
        results['X_reduced'] = X_reduced
        results['reduction_model'] = pca_model
    
    # 5. Analizar características de los clusters
    if analyze_features:
        logger.info("Paso 5: Analizando características de los clusters...")
        cluster_stats, distinctive_features = analyze_clusters(X, cluster_labels)
        results['cluster_stats'] = cluster_stats
        results['distinctive_features'] = distinctive_features
    
    logger.info("Análisis de clustering completado!")
    return results