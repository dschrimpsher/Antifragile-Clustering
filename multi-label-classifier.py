import networkx as nx
import pandas as pd
import numpy as np
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from common import ScoringStrategy, mechanism_names
from dataclasses import dataclass
from typing import List
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict

# model = SentenceTransformer("all-MiniLM-L6-v2")  # load ONCE
model = SentenceTransformer("all-mpnet-base-v2")

convexity_anchor_texts = [
    "Convex or nonlinear response under stress, where significant or large disturbances produce unusually large performance gains.",
    ## Secondary
    "Nonlinear performance function in which large stressors result in disproportionately enhanced outcomes.",
    "Convex response especially when stressed, generating unusually large gains from significant disturbances."
    "Significant shocks lead to enhanced performance through a nonlinear (convex) function.",

    # GTP
    "Convexity is an antifragile mechanism in which a system experiences limited losses from negative shocks while realizing disproportionately large gains from positive shocks as variability increases.",
    "An antifragile mechanism exhibits convexity when its payoff curve bends upward, meaning that increased stress or volatility produces asymmetric benefits rather than balanced risks.",
    "In antifragile systems, convexity refers to a structural property where exposure to uncertainty results in nonlinear upside that outweighs and dominates potential downside.",
    "The system’s performance increases nonlinearly as volatility increases.",
    "Larger shocks produce disproportionately larger gains.",
    "The response curve is convex with increasing marginal benefit under stress."
]

asymmetry_anchor_texts = [
    "Asymmetric response under stress characterized by bounded downside and upside gain from increased volatility.",

    # Secondary Anchors
    "Asymmetric gains under stress, where positive response to increased volatility exceeds potential loss.",
    "Bounded performance function with upside gain when exposed to uncertainty or shocks.",
    "Positive, asymmetric response to volatility producing greater upside than downside.",

    # GTP
    "Asymmetry is an antifragile mechanism in which a system is structured so that potential gains from favorable events substantially exceed potential losses from unfavorable events.",
    "An antifragile system exhibits asymmetry when its exposure to uncertainty is skewed toward limited downside and open-ended or amplified upside.",
    "In the context of antifragility, asymmetry refers to a risk structure where negative shocks produce bounded harm while positive shocks create disproportionate or scalable benefits."
]

protective_rore_anchor_texts = [
    "Manages or withstands disturbances and absorbs shocks in a way that leads to improvement or enhanced performance.",

    # Alternate
    "Manages or withstand disturbances through robustness, absorbing shocks and emerging more robust while achieving improvement.",

    # Secondary Anchors
    "Endures extreme events and emerges more robust, transforming risk into opportunity.",
    "Establishes robustness and capitalizes on disruption, achieving gains beyond recovery.",
    "Absorbs shocks and gets better, doing more than merely recovering.",

    # GTP
    "Protective robustness and resilience refer to an antifragility-related mechanism in which a system is deliberately designed to absorb shocks, resist failure, and maintain core functionality under stress.",
    "A system exhibits protective robustness and resilience when it preserves structural integrity and operational continuity despite disruptions, variability, or adverse conditions.",
    "In the context of antifragility research, protective robustness and resilience describe defensive capabilities that limit damage and enable recovery from shocks without necessarily generating additional gains."
]

iterative_learning_anchor_texts = [
    "Learns from stress or events, with knowledge accumulation enabling improved performance under uncertainty.",
    # Secondary Anchors
    "Improves through learning from disruption or randomness.",
    "Fosters learning and uses accumulated knowledge to capitalize on uncertainty.",
    "Knowledge develops or gets better through exposure to stress.",

    # GTP
    "Iterative learning is an antifragile mechanism in which a system improves its performance through repeated cycles of feedback, adjustment, and refinement in response to experience.",
    "An antifragile system exhibits iterative learning when exposure to errors, variation, or small failures leads to incremental modifications that cumulatively enhance capability over time.",
    "In the context of antifragility, iterative learning refers to a structured process of trial, feedback, and adaptation through which performance is progressively optimized across successive iterations."
]

adaptive_restructuring_anchor_texts = [
    "Adapts or restructures in response to stress, removing weaknesses or shifting system design to improve functionality and performance.",
    # Secondary Anchors
    "Stress triggers evolutionary change or innovation that strengthens the system.",
    "Regenerates or modifies structure to capitalize on uncertainty and exploit opportunities.",
    "Improves by redesigning or strengthening system components following disruption.",

    "Adaptive restructuring is an antifragile mechanism in which a system responds to stress or disruption by altering its internal structure, configuration, or organizational design to better function under new conditions.",
    "An antifragile system exhibits adaptive restructuring when significant shocks trigger deliberate reorganization, reconfiguration, or redistribution of resources that improve long-term viability.",
    "In the context of antifragility, adaptive restructuring refers to structural transformation initiated in response to environmental change, resulting in a revised system architecture better suited to prevailing conditions."
]

growth_through_disorder_anchor_texts = [
    "Growth or performance improvement because of disorder, uncertainty, volatility, or shocks.",

    # Secondary Anchors
    "Benefits from volatility or disruption, resulting in gains.",
    "Performance gains due to exposure to shocks or uncertainty.",
    "Improves as a result of disorder or environmental disturbance.",

    # GTP
    "Growth through disorder is an antifragile mechanism in which exposure to volatility, disruption, or randomness directly stimulates expansion, innovation, or enhanced capability beyond pre-shock levels.",
    "An antifragile system exhibits growth through disorder when instability or stress serves as a catalyst for new development, resulting in measurable improvement because of, not merely despite, the disturbance.",
    "In the context of antifragility, growth through disorder refers to a process where crises, variability, or shocks generate transformative gains that would not have emerged under stable conditions."
]


def aggregate_similarity(sim_row: np.ndarray, strategy: ScoringStrategy, k: int = 3) -> float:
    sim_row = np.asarray(sim_row, dtype=float)

    if strategy == ScoringStrategy.MAX_SIMILARITY:
        return float(sim_row.max())

    if strategy == ScoringStrategy.TOP_K_AVERAGE:
        k = max(1, min(int(k), sim_row.size))
        topk = np.sort(sim_row)[-k:]
        return float(topk.mean())

    raise ValueError(f"Unknown strategy: {strategy}")


def score_one_mechanism(
        emb_defs: np.ndarray,
        anchor_texts: list[str],
        *,
        model,
        strategy: ScoringStrategy = ScoringStrategy.TOP_K_AVERAGE,
        top_k: int = 3
) -> np.ndarray:
    """
    Returns: scores shape (N,) for this one mechanism.
    """
    emb_anchors = model.encode(anchor_texts, convert_to_numpy=True)  # (M, D)
    sims = cosine_similarity(emb_defs, emb_anchors)  # (N, M)

    scores = np.array(
        [aggregate_similarity(row, strategy=strategy, k=top_k) for row in sims],
        dtype=float
    )
    return scores


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns: boolean mask shape (N,) indicating which defs match this mechanism.
    """
    return np.asarray(scores) >= float(threshold)


df = pd.read_csv("data/encoded_uuid_v3_normalized.csv")
df["definition"] = df["definition"].astype(str)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# ---- Round 1 ----
defs = df["definition"].tolist()
embedded_definitions = model.encode(defs, convert_to_numpy=True)


@dataclass
class MechanismConfig:
    anchors: List[str]
    threshold: float
    target: int


mechanisms = {
    "convexity": MechanismConfig(convexity_anchor_texts, 0.58, 7),
    "asymmetry": MechanismConfig(asymmetry_anchor_texts, 0.515, 5),
    "protective_rore": MechanismConfig(protective_rore_anchor_texts, 0.55, 11),
    "iterative_learning": MechanismConfig(iterative_learning_anchor_texts, 0.45625, 8),
    "adaptive_restructuring": MechanismConfig(adaptive_restructuring_anchor_texts, 0.509, 10),
    "growth_through_disorder": MechanismConfig(growth_through_disorder_anchor_texts, 0.39, 43),
}

results_scores = {}
results_hits = {}
for name, config in mechanisms.items():
    done = False
    print(name, config.threshold)
    to_many = False
    delta = 0.01
    while not done:
        scores = score_one_mechanism(
            embedded_definitions,
            config.anchors[:1],
            model=model,
            strategy=ScoringStrategy.TOP_K_AVERAGE,
            top_k=3,
        )
        hits = apply_threshold(scores, threshold=config.threshold)

        results_scores[name] = scores
        results_hits[name] = hits
        if results_hits[name].sum() < config.target and config.threshold > 0.0:
            # print(results_hits[name].sum(), config.threshold, "To Few")
            config.threshold -= delta
            if to_many:
                delta /= 2
            to_many = False
        elif results_hits[name].sum() > config.target and config.threshold <= 1.0:
            # print(results_hits[name].sum(), config.threshold, "To Many")
            config.threshold += delta
            if not to_many:
                delta /= 2
            to_many = True
        else:
            done = True
            print(results_hits[name].sum(), config.threshold, "Final")

df_out = df.copy()

for name in mechanisms.keys():
    df_out[f"score_{name}"] = results_scores[name]
    df_out[f"is_{name}"] = results_hits[name]

mechanism_cols = [
    "is_convexity",
    "is_asymmetry",
    "is_protective_rore",
    "is_iterative_learning",
    "is_adaptive_restructuring",
    "is_growth_through_disorder",
]

df_out["amf_any"] = df_out[mechanism_cols].any(axis=1)
df_out["non_amf"] = (~df_out["amf_any"])

paper_profile = (
    df_out
    .groupby("title")[mechanism_cols]
    .max()  # max works as OR for booleans
    .reset_index()
)

paper_profile["has_transformative_mechanism"] = (
        paper_profile["is_convexity"]
        | paper_profile["is_asymmetry"]
        | paper_profile["is_protective_rore"]
        | paper_profile["is_growth_through_disorder"]
        | paper_profile["is_iterative_learning"]
        | paper_profile["is_adaptive_restructuring"]
)

# Step 10

non_amf_defs = df_out[df_out["non_amf"]].copy()
non_amf_defs_list = non_amf_defs["definition"].tolist()

# Cluster non-amf defs
non_amf_indices = non_amf_defs.index.to_numpy()
emb_non_amf = embedded_definitions[non_amf_indices]


# Louvain Cluster Option

def run_cluster_center_focus(emb_defs, definitions, center, threshold=0.45):
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
            g.add_edge(i, none_node, weight=float(sim))

    return community_louvain.best_partition(g, weight="weight")


def run_basic_louvain_cluster(emb_defs, definitions, threshold=0.45):
    sim_matrix = cosine_similarity(emb_defs)

    # Build similarity graph
    g = nx.Graph()

    for i in range(len(definitions)):
        g.add_node(i)

        for j in range(i + 1, len(definitions)):
            sim = sim_matrix[i][j]
            if sim > threshold:
                g.add_edge(i, j, weight=sim)

    # Louvain clustering
    return community_louvain.best_partition(g, weight="weight")


def agg_max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size else 0.0

def topk_mean(x: np.ndarray, k: int = 3) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    k = max(1, min(k, x.size))
    return float(np.sort(x)[-k:].mean())

rore_extended_anchors = [
    "Manages or copes with disruption, restores function under uncertainty, and may become more robust or resilient after stress.",
    "Recovers from shocks and subsequently improves or becomes stronger.",
    "Absorbs disturbances and achieves higher performance following stress.",
    "Deals with uncertainty and goes beyond resilience through recovery accompanied by improvement."
]

adaptability_extended_anchors = [
    "Ensures adaptability under uncertainty and expands performance following disruption.",
    "Maintains adaptability in response to uncertainty or disturbance.",
    "Adapts to change and achieves performance expansion after an event.",
    "Demonstrates adaptability when exposed to uncertainty."
]

viability_anchors = [
    "Survives disruption, volatility, or uncertainty."
    "Maintains existence under disturbance or stress.",
    "Continues operating despite uncertainty or shock.",
    "Remains viable when exposed to disruption.",
]

default_threshold = 0.68

non_mechanisms = {
    "rore_extended": MechanismConfig(rore_extended_anchors, default_threshold, 100),
    "adaptability_extended": MechanismConfig(adaptability_extended_anchors, default_threshold, 100),
    "viability": MechanismConfig(viability_anchors, default_threshold, 100),
}

base_partition = run_basic_louvain_cluster(emb_non_amf, non_amf_defs_list, threshold=default_threshold)
non_amf_defs["louvain_cluster"] = [base_partition[i] for i in range(len(non_amf_defs))]

cluster_counts = Counter(base_partition.values())

print("Cluster counts:", cluster_counts)
print("Number of clusters:", len(cluster_counts))

cluster_ids = sorted(non_amf_defs["louvain_cluster"].unique())

cluster_anchor_sims = {}  # cluster_id -> np.array shape (A,)

anchor_texts = []
anchor_owner = []  # same length as anchor_texts, stores which non-amf label each anchor came from

for name, config in non_mechanisms.items():
    for a in config.anchors:
        anchor_texts.append(a)
        anchor_owner.append(name)

emb_anchors = model.encode(anchor_texts, convert_to_numpy=True)  # (A, D)
cluster_best_label = {}
cluster_best_score = {}

for cid in cluster_ids:
    members = (non_amf_defs["louvain_cluster"] == cid).to_numpy()
    emb_cluster = emb_non_amf[members]  # (Nc, D)

    # centroid = emb_cluster.mean(axis=0, keepdims=True)  # (1, D)
    # sims = cosine_similarity(centroid, emb_anchors).ravel()  # (A,)

    sim_mat = cosine_similarity(emb_cluster, emb_anchors)  # (Nc, A)

    by_label = defaultdict(list)

    for anchor_idx, owner in enumerate(anchor_owner):
        by_label[owner].extend(sim_mat[:, anchor_idx])

    label_scores = {label: agg_max(np.array(vals)) for label, vals in by_label.items()}

    # label_scores = {
    #     label: topk_mean(np.array(vals), k=5)
    #     for label, vals in by_label.items()
    # }

    best_label = max(label_scores, key=label_scores.get)

    print(f"\nCluster {cid}  best_match={best_label}  scores={label_scores}")
    cluster_best_label[cid] = best_label
    cluster_best_score[cid] = label_scores[best_label]

non_amf_defs["non_amf_cluster_label"] = (
    non_amf_defs["louvain_cluster"]
        .map(cluster_best_label)
)

non_amf_defs["non_amf_cluster_score"] = (
    non_amf_defs["louvain_cluster"]
        .map(cluster_best_score)
)

non_amf_defs.to_csv(
    "results/non_amf_louvain_clustering_results.csv",
    index=False
)

# HDBSCAN hdbscan_cluster Option

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric="euclidean",
    cluster_selection_method="eom"
)


non_amf_hdbscan_defs = df_out[df_out["non_amf"]].copy()
non_amf_hdbscan_defs_list = non_amf_hdbscan_defs["definition"].tolist()

# hdbscan_cluster non-amf defs
non_amf_hdbscan_indices = non_amf_hdbscan_defs.index.to_numpy()
emb_non_amf_hdbscan = embedded_definitions[non_amf_hdbscan_indices]

cluster_labels = clusterer.fit_predict(emb_non_amf_hdbscan)
non_amf_hdbscan_defs["hdbscan_cluster"] = cluster_labels

for c in sorted(non_amf_hdbscan_defs["hdbscan_cluster"].unique()):
    print(f"\nhdbscan_cluster {c}")
    examples = non_amf_hdbscan_defs[non_amf_hdbscan_defs["hdbscan_cluster"] == c]["definition"].head(5)
    for e in examples:
        print("-", e)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(non_amf_hdbscan_defs["definition"])

terms = np.array(vectorizer.get_feature_names_out())

for c in sorted(non_amf_hdbscan_defs["hdbscan_cluster"].unique()):
    indices = np.where(non_amf_hdbscan_defs["hdbscan_cluster"] == c)[0]
    mean_tfidf = X[indices].mean(axis=0).A1
    top_terms = terms[np.argsort(mean_tfidf)[-10:]]
    print(f"\nhdbscan_cluster {c + 2} top terms:", top_terms)


for cid in sorted(non_amf_hdbscan_defs["hdbscan_cluster"].unique()):
    members = (non_amf_hdbscan_defs["hdbscan_cluster"] == cid).to_numpy()
    emb_cluster = emb_non_amf_hdbscan[members]  # (Nc, D)

    # centroid = emb_cluster.mean(axis=0, keepdims=True)  # (1, D)
    # sims = cosine_similarity(centroid, emb_anchors).ravel()  # (A,)

    sim_mat = cosine_similarity(emb_cluster, emb_anchors)  # (Nc, A)

    by_label = defaultdict(list)

    for anchor_idx, owner in enumerate(anchor_owner):
        by_label[owner].extend(sim_mat[:, anchor_idx])

    label_scores = {label: agg_max(np.array(vals)) for label, vals in by_label.items()}

    # label_scores = {
    #     label: topk_mean(np.array(vals), k=5)
    #     for label, vals in by_label.items()
    # }

    best_label = max(label_scores, key=label_scores.get)

    print(f"\nhdbscan_cluster {cid}  best_match={best_label}  scores={label_scores}")
    cluster_best_label[cid] = best_label
    cluster_best_score[cid] = label_scores[best_label]

non_amf_hdbscan_defs["cluster_label"] = (
    non_amf_hdbscan_defs["hdbscan_cluster"]
        .map(cluster_best_label)
)

non_amf_hdbscan_defs["cluster_score"] = (
    non_amf_hdbscan_defs["hdbscan_cluster"]
        .map(cluster_best_score)
)


df_out.loc[non_amf_hdbscan_defs.index, "hdbscan_cluster"] = cluster_labels

# Finally Write data

df_out.to_csv("results/multi-label-classified_definitions.csv", index=False)

paper_profile.to_csv("results/multi-label-classified_papers.csv", index=False)

cluster_export = df_out[df_out["non_amf"]].copy()

cluster_export.to_csv(
    "results/non_amf_hdbscan_hdbscan_clustered_definitions.csv",
    index=False
)

cluster_summary = (
    cluster_export
    .groupby("hdbscan_cluster")
    .agg(
        count=("definition", "count"),
        papers=("title", "nunique")
    )
    .reset_index()
)

cluster_top_terms = {}

for c in sorted(cluster_export["hdbscan_cluster"].dropna().unique()):
    mask = (cluster_export["hdbscan_cluster"] == c).to_numpy()
    mean_tfidf = X[mask].mean(axis=0).A1
    top_terms = terms[np.argsort(mean_tfidf)[-8:]]
    cluster_top_terms[c] = ", ".join(top_terms)

cluster_summary["top_terms"] = cluster_summary["hdbscan_cluster"].map(cluster_top_terms)

cluster_summary.to_csv(
    "results/non_amf_hdbscan_hdbscan_cluster_summary_with_terms.csv",
    index=False
)
