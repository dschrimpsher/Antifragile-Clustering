import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from common import MECH_COLS, MAIN_SIX
import networkx as nx
import community as community_louvain  # python-louvain



df = pd.read_csv("data/encoded_uuid_v2_normalized.csv")
df["definition"] = df["definition"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Make sure they're clean 0/1 ints
df[MECH_COLS] = (
    df[MECH_COLS]
        .fillna(0)
        .astype(np.int8)
)

# Create the vector column
def_vector = df[MECH_COLS].to_numpy(dtype=np.int8)

S_cos = cosine_similarity(def_vector)
def_vectory_bool = def_vector.astype(bool)

D_jacc = pairwise_distances(def_vectory_bool, metric="jaccard")
S_jacc = 1.0 - D_jacc
vals = S_jacc[np.triu_indices_from(S_jacc, k=1)]
print("Min:", vals.min())
print("Mean:", vals.mean())
print("Max:", vals.max())
print("Percentiles [25,50,75,90,95]:", np.percentile(vals, [25,50,75,90,95]))


threshold = 0.4
n = S_jacc.shape[0]

G = nx.Graph()

# Use definition_id if you have it
ids = df["definition_id"].tolist()
G.add_nodes_from(ids)

rows, cols = np.where(np.triu(S_jacc, k=1) >= threshold)

for i, j in zip(rows, cols):
    G.add_edge(ids[i], ids[j], weight=float(S_jacc[i, j]))

print("Edges:", len(rows))
print("Possible pairs:", len(S_jacc)*(len(S_jacc)-1)//2)
edge_ratio  = len(rows) / (len(S_jacc)*(len(S_jacc)-1)//2)
print("Edge ratio:", edge_ratio)
if edge_ratio > 0.05:
    print("Moderately Dense Graph")
else:
    print("Sparse Graph, consider changing the threshold")



# Run Louvain
partition = community_louvain.best_partition(G, weight="weight", random_state=42, resolution=1.0)

# Attach to dataframe
df["community"] = df["definition_id"].map(partition)

print("Number of communities:", len(set(partition.values())))
print(df["community"].value_counts().sort_index())

# partition is: {uuid: community_id}
cluster_profiles = (
    df.groupby("community")[MECH_COLS]
      .mean()
      .round(2)
)

print("Cluster Profiles per Mechanism:", cluster_profiles)

cluster_var = (
    df.groupby("community")[MECH_COLS]
      .std()
      .fillna(0)
)
print("Cluster Variance:", cluster_var)

#
# Strong dominance:
#
# mechanism mean ≥ 0.70
#
# Moderate dominance:
#
# ≥ 0.50
#
# Hybrid:
#
# two mechanisms ≥ 0.50
#
# Sparse:
#
# no mechanism ≥ 0.40

cluster_profiles = df.groupby("community")[MECH_COLS].mean()


def label_cluster(row):
    strong = row[row >= 0.6].index.tolist()

    if len(strong) == 1:
        return f"{strong[0]}-dominant"
    elif len(strong) >= 2:
        return "hybrid-" + "-".join(strong)
    else:
        return "mixed/weak"


cluster_labels = cluster_profiles.apply(label_cluster, axis=1)
cluster_labels.name = "label"   # give the Series a column name

print(cluster_labels)


output_labels = "results/cluster_labels.csv"
cluster_labels.to_csv(output_labels, index=True)
print(f"Saved cluster labels to {output_labels}")

def classify_definition(row):
    main_sum = row[MAIN_SIX].sum()

    if main_sum == 6:
        return "all_mechanisms"

    if main_sum > 0:
        return "partial_mechanism"

    if row["rore_extended"] == 1:
        return "rore_extended_only"

    if row["adaptability_framed"] == 1:
        return "adaptability_framed_only"

    return "no_mechanism"


df["mechanism_class"] = df.apply(classify_definition, axis=1)

cluster_class_distribution = (
    df.groupby("community")["mechanism_class"]
      .value_counts(normalize=True)
      .unstack()
      .fillna(0)
      .round(2)
)

print(cluster_class_distribution)

output_definitions = "results/definitions_with_clusters.csv"
df.to_csv(output_definitions, index=False)
print(f"Saved row-level results to {output_definitions}")

cluster_profiles_out = cluster_profiles.round(3)
cluster_profiles_out["size"] = df["community"].value_counts()

cluster_profiles_out = cluster_profiles_out.sort_index()

output_profiles = "results/cluster_profiles.csv"
cluster_profiles_out.to_csv(output_profiles)
print(f"Saved cluster profiles to {output_profiles}")

output_class_dist = "results/cluster_class_distribution.csv"
cluster_class_distribution.to_csv(output_class_dist)
print(f"Saved class distribution to {output_class_dist}")

edge_df = nx.to_pandas_edgelist(G)

output_edges = "results/mechanism_graph_edges.csv"
edge_df.to_csv(output_edges, index=False)
print(f"Saved graph edges to {output_edges}")

