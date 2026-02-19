import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # python-louvain

mechanism_names = [
    "convexity",
    "asymmetry",
    "protective_robustness",
    "iterative_learning",
    "adaptive_restructuring",
    "growth_through_disorder",
    "adaptability-oriented",
    "resilience-oriented",
]

bounds = mechanism_indexes + [len(centers)]  # IMPORTANT

def run_cluster_precomputed(emb_defs, definitions, center, threshold=0.45):
    center_emb = model.encode([center], convert_to_numpy=True)[0]

    # similarity of each definition to center
    sims = cosine_similarity(emb_defs, center_emb.reshape(1, -1)).ravel()

    g = nx.Graph()
    n = len(definitions)

    center_node = n
    none_node = n + 1

    g.add_nodes_from(range(n + 2))

    for i, sim in enumerate(sims):
        if sim > threshold:
            g.add_edge(i, center_node, weight=float(sim))
        else:
            g.add_edge(i, none_node, weight=1.0)

    return community_louvain.best_partition(g, weight="weight")



df = pd.read_csv("data/definitions_expanded.csv")
df["definition"] = df["definition"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# ---- Round 1 ----
defs = df["definition"].tolist()
embedded_definitions = model.encode(defs, convert_to_numpy=True)

mechanism_index = 0
center_clusters = []
new_cols = {}

for c in centers:
    partition = run_cluster_precomputed(embedded_definitions, defs, c, threshold=0.45)

    new_cols[f"louvain_cluster_{c}"] = df.index.to_series().map(partition)
    center_clusters.append(partition[len(defs)])  # <-- YOU FORGOT THIS
    # print(c)


# Add all columns at once
df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)


# --- after your loop that fills df[f"louvain_cluster_{c}"] and center_clusters ---
# center_clusters[i] corresponds to centers[i]


# bounds = mechanism_indexes + [len(centers)]  # you already have this
# mechanism_names = [...]                      # you already have this

center_counts = []

for k, mech in enumerate(mechanism_names):
    start, end = bounds[k], bounds[k+1]

    # collect per-center boolean masks (each is length=len(df))
    masks = []
    for i in range(start, end):
        c = centers[i]
        center_cid = center_clusters[i]
        masks.append(df[f"louvain_cluster_{c}"].eq(center_cid))

    # Combine across centers for this mechanism
    mask_df = pd.concat(masks, axis=1)           # shape: (n_rows, n_centers_for_mech)
    matched_any = mask_df.any(axis=1)            # bool: matched at least one center
    score = mask_df.sum(axis=1).astype(int)      # int: number of centers matched

    # Persist columns
    df[f"match_{mech}"] = matched_any
    df[f"score_{mech}"] = score

    # Count definitions matching at least one center
    count = int(matched_any.sum())
    center_counts.append(count)

    print(f"{mech}: {count} definitions matched ≥1 keyword")


summary_rows = []

for mech, count in zip(mechanism_names, center_counts):
    mask = df[f"match_{mech}"]

    if mask.any():
        y_min = int(df.loc[mask, "year"].min())
        y_max = int(df.loc[mask, "year"].max())
        year_range = f"{y_min}-{y_max}"
    else:
        year_range = "N/A"

    summary_rows.append({
        "mechanism": mech,
        "count": count,
        "year_range": year_range
    })

    print(f"{mech}: {count} definitions, {year_range}")



df.to_csv("data/definitions_expanded_with_clusters.csv", index=False)
# print(df[["louvain_cluster", "louvain_subcluster"]].value_counts().sort_index())

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    "data/definitions_expanded_with_clusters_summary.csv",
    index=False
)

