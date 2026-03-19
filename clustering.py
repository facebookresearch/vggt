import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class GeometricClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        
    def fit_predict(self, csv_path, output_csv_path):
        df = pd.parse_csv(csv_path)
        
        # Características geométricas a utilizar
        features = ['camera_height', 'pitch_angle', 'mean_depth']
        X = df[features]
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Agrupar
        df['cluster_id'] = self.kmeans.fit_predict(X_scaled)
        df.to_csv(output_csv_path, index=False)
        
        print(f"Agrupamiento completado. Centroides:\n{self.kmeans.cluster_centers_}")
        return df