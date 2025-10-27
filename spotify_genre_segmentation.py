# spotify_genre_segmentation.py
# Complete pipeline: load -> preprocess -> visualize -> KMeans clustering -> PCA -> recommender -> save results
# Works with a typical Spotify-style dataset (audio features + metadata)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "dataset.csv"            # path to your uploaded file
OUT_CLUSTERED = "clustered_songs.csv"
RANDOM_STATE = 42

SAMPLE_FOR_SELECTION = True
SAMPLE_SIZE = 5000   # used for silhouette/elbow & fit speed

# ---------------------------
# 1) Load dataset
# ---------------------------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

# ---------------------------
# 2) Identify features
# ---------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
common_audio = ['danceability','energy','loudness','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo','duration_ms','key','mode']
audio_features = [c for c in common_audio if c in numeric_cols]

if len(audio_features) >= 4:
    features_for_model = audio_features
else:
    features_for_model = [c for c in numeric_cols if c.lower() not in ('id','track_id','playlist_id')]

if len(features_for_model) < 2:
    raise ValueError("Need at least 2 numeric features for clustering.")

# ---------------------------
# 3) Preprocessing: impute + scale
# ---------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, features_for_model)
], remainder='drop')

X_full = preprocessor.fit_transform(df)
df_num_imputed = df[features_for_model].fillna(df[features_for_model].median())

# ---------------------------
# 4) Choose K (elbow + silhouette)
# ---------------------------
if SAMPLE_FOR_SELECTION and X_full.shape[0] > SAMPLE_SIZE:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(np.arange(X_full.shape[0]), size=SAMPLE_SIZE, replace=False)
    X_sample = X_full[sample_idx]
else:
    X_sample = X_full

min_k = 2
max_k = min(10, max(3, X_sample.shape[0]//50))
K_range = list(range(min_k, max_k+1))

sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labs = km.fit_predict(X_sample)
    try:
        sil_scores.append(silhouette_score(X_sample, labs))
    except Exception:
        sil_scores.append(np.nan)

best_k = K_range[int(np.nanargmax(sil_scores))]
print(f"Chosen K (best silhouette): {best_k}")

# ---------------------------
# 5) Fit KMeans full data
# ---------------------------
kmeans_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
kmeans_model.fit(X_sample)
labels_full = kmeans_model.predict(X_full)
df['cluster'] = labels_full

# ---------------------------
# 6) PCA 2D Visualization
# ---------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_sample)
X_pca = pca.transform(X_full)

# ---------------------------
# 7) Cluster means
# ---------------------------
cluster_means = pd.concat([df[features_for_model], df['cluster']], axis=1).groupby('cluster').mean().round(3)

# ---------------------------
# 8) Nearest Neighbour Recommender
# ---------------------------
nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto')
nbrs.fit(X_full)

def recommend_by_index(idx, n=5):
    """
    idx can be integer index or string (partial song name)
    returns top-n similar songs
    """
    if isinstance(idx, str):
        name_cols = [c for c in df.columns if c.lower() in ('track_name','name','title','song')]
        if not name_cols:
            raise ValueError("No track name columns found.")
        matches = df[df[name_cols[0]].str.contains(idx, case=False, na=False)]
        if matches.empty:
            raise ValueError("No matches found for query string.")
        idx = matches.index[0]

    if not (0 <= idx < len(df)):
        raise IndexError("Index out of range.")

    distances, indices = nbrs.kneighbors(X_full[idx:idx+1], n_neighbors=n+1)
    inds = indices[0].tolist()
    if idx in inds:
        inds.remove(idx)
    res = df.iloc[inds].copy()
    res['distance'] = distances[0][1:len(inds)+1]
    return res

# ---------------------------
# 9) Save results
# ---------------------------
df.to_csv(OUT_CLUSTERED, index=False)
print("Saved clustered dataset to:", OUT_CLUSTERED)