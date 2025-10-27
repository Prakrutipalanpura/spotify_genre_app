Python 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import streamlit as st
... import pandas as pd
... import matplotlib.pyplot as plt
... import numpy as np
... from spotify_genre_segmentation import recommend_by_index, df, labels_full, cluster_means
... 
... # ---------------------------
... # Streamlit UI
... # ---------------------------
... 
... st.set_page_config(page_title="Spotify Genre Segmentation", layout="wide")
... 
... st.title("🎵 Spotify Genre Segmentation & Song Recommender")
... st.markdown("""
... This interactive web app clusters Spotify tracks based on their *audio features* 
... and recommends similar songs using a *nearest-neighbour recommender*.
... 
... Upload your Spotify dataset or use the default one to explore music patterns!
... """)
... 
... # ---------------------------
... # Show dataset preview
... # ---------------------------
... with st.expander("📂 View Dataset Preview"):
...     st.dataframe(df.head(10))
... 
... # ---------------------------
... # Show cluster summary
... # ---------------------------
... st.subheader("📊 Cluster Summary (Mean Feature Values)")
... st.dataframe(cluster_means)
... 
... # ---------------------------
... # Visualize Cluster Distribution
... # ---------------------------
... st.subheader("🎨 Cluster Distribution")
... cluster_counts = df['cluster'].value_counts().sort_index()
... fig, ax = plt.subplots()
... ax.bar(cluster_counts.index, cluster_counts.values)
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Songs")
ax.set_title("Cluster Distribution")
st.pyplot(fig)

# ---------------------------
# Song Recommendation Section
# ---------------------------
st.subheader("🎧 Find Similar Songs")

song_name = st.text_input("Enter a song name (or part of it) to find similar tracks:")

if song_name:
    try:
        recs = recommend_by_index(song_name, n=5)
        cols = [c for c in df.columns if c.lower() in ('track_name','artists','album_name')]
        cols_to_show = cols + ['cluster','distance'] if 'distance' in recs.columns else cols
        st.success(f"Here are songs similar to *{song_name}* 🎶")
        st.dataframe(recs[cols_to_show])
    except Exception as e:
        st.error(f"❌ {str(e)}")

st.markdown("---")
