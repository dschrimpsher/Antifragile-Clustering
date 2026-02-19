import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # python-louvain

# Load the expanded file
df = pd.read_csv("data/definitions_expanded.csv")

# This is what we will cluster
definitions = df["definition"].astype(str).tolist()

# Keep the metadata (title/year/etc.) around
titles = df["title"].tolist()
years = df["year"].tolist()
definition_numbers = df["definition_number"].tolist()

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(definitions)

# Similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Build similarity graph
G = nx.Graph()
threshold = 0.45  # tweak to adjust granularity



for i in range(len(definitions)):
    G.add_node(i)

    for j in range(i + 1, len(definitions)):
        sim = sim_matrix[i][j]
        if sim > threshold:
            G.add_edge(i, j, weight=sim)

# Louvain clustering
partition = community_louvain.best_partition(G, weight="weight")

# Attach cluster ids back to the dataframe
df["louvain_cluster"] = df.index.map(partition)

# Optional: save with cluster IDs
df.to_csv("data/definitions_expanded_with_clusters.csv", index=False)

# Example: peek at clusters grouped with paper/year
for cid, group in df.groupby("louvain_cluster"):
    print(f"\n=== Cluster {cid} ===")
    for _, row in group.iterrows():
        print(f"[{int(row['year'])}] {row['title']} (def {row['definition_number']}):")
        print(" ", row["definition"])
