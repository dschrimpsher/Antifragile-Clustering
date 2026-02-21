import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from common import ScoringStrategy, mechanism_names
from dataclasses import dataclass
from typing import List
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer


# model = SentenceTransformer("all-MiniLM-L6-v2")  # load ONCE
model = SentenceTransformer("all-mpnet-base-v2")

convexity_anchor_texts = [
    "Convexity is an antifragile mechanism in which a system experiences limited losses from negative shocks while realizing disproportionately large gains from positive shocks as variability increases.",
    "An antifragile mechanism exhibits convexity when its payoff curve bends upward, meaning that increased stress or volatility produces asymmetric benefits rather than balanced risks.",
    "In antifragile systems, convexity refers to a structural property where exposure to uncertainty results in nonlinear upside that outweighs and dominates potential downside.",
    "The system’s performance increases nonlinearly as volatility increases.",
    "Larger shocks produce disproportionately larger gains.",
    "The response curve is convex with increasing marginal benefit under stress."
]

asymmetry_anchor_texts = [
    "Asymmetry is an antifragile mechanism in which a system is structured so that potential gains from favorable events substantially exceed potential losses from unfavorable events.",
    "An antifragile system exhibits asymmetry when its exposure to uncertainty is skewed toward limited downside and open-ended or amplified upside.",
    "In the context of antifragility, asymmetry refers to a risk structure where negative shocks produce bounded harm while positive shocks create disproportionate or scalable benefits."
]

protective_rore_anchor_texts = [
    "Protective robustness and resilience refer to an antifragility-related mechanism in which a system is deliberately designed to absorb shocks, resist failure, and maintain core functionality under stress.",
    "A system exhibits protective robustness and resilience when it preserves structural integrity and operational continuity despite disruptions, variability, or adverse conditions.",
    "In the context of antifragility research, protective robustness and resilience describe defensive capabilities that limit damage and enable recovery from shocks without necessarily generating additional gains."
]

iterative_learning_anchor_texts = [
    "Iterative learning is an antifragile mechanism in which a system improves its performance through repeated cycles of feedback, adjustment, and refinement in response to experience.",
    "An antifragile system exhibits iterative learning when exposure to errors, variation, or small failures leads to incremental modifications that cumulatively enhance capability over time.",
    "In the context of antifragility, iterative learning refers to a structured process of trial, feedback, and adaptation through which performance is progressively optimized across successive iterations."
]

adaptive_restructuring_anchor_texts = [
    "Adaptive restructuring is an antifragile mechanism in which a system responds to stress or disruption by altering its internal structure, configuration, or organizational design to better function under new conditions.",
    "An antifragile system exhibits adaptive restructuring when significant shocks trigger deliberate reorganization, reconfiguration, or redistribution of resources that improve long-term viability.",
    "In the context of antifragility, adaptive restructuring refers to structural transformation initiated in response to environmental change, resulting in a revised system architecture better suited to prevailing conditions."
]

growth_through_disorder_anchor_texts = [
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

mechanisms = {
    "convexity": MechanismConfig(convexity_anchor_texts, 0.62),
    "asymmetry": MechanismConfig(asymmetry_anchor_texts, 0.62),
    "protective_rore": MechanismConfig(protective_rore_anchor_texts, 0.55),
    "iterative_learning": MechanismConfig(iterative_learning_anchor_texts, 0.52),
    "adaptive_restructuring": MechanismConfig(adaptive_restructuring_anchor_texts, 0.52),
    "growth_through_disorder": MechanismConfig(growth_through_disorder_anchor_texts, 0.52),
}

results_scores = {}
results_hits = {}
for name, config in mechanisms.items():
    scores = score_one_mechanism(
        embedded_definitions,
        config.anchors,
        model=model,
        strategy=ScoringStrategy.TOP_K_AVERAGE,
        top_k=3,
    )
    hits = apply_threshold(scores, threshold=config.threshold)

    results_scores[name] = scores
    results_hits[name] = hits

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
    .max()   # max works as OR for booleans
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


# Cluster non-amf defs
non_amf_indices = non_amf_defs.index.to_numpy()
emb_non_amf = embedded_definitions[non_amf_indices]

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(emb_non_amf)
non_amf_defs["cluster"] = cluster_labels

for c in sorted(non_amf_defs["cluster"].unique()):
    print(f"\nCluster {c+2}")
    examples = non_amf_defs[non_amf_defs["cluster"] == c]["definition"].head(5)
    for e in examples:
        print("-", e)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(non_amf_defs["definition"])

terms = np.array(vectorizer.get_feature_names_out())

for c in sorted(non_amf_defs["cluster"].unique()):
    indices = np.where(non_amf_defs["cluster"] == c)[0]
    mean_tfidf = X[indices].mean(axis=0).A1
    top_terms = terms[np.argsort(mean_tfidf)[-10:]]
    print(f"\nCluster {c+2} top terms:", top_terms)

df_out.loc[non_amf_defs.index, "non_amf_cluster"] = cluster_labels


# Write data

df_out.to_csv("results/multi-label-classified_definitions.csv", index=False)

paper_profile.to_csv("results/multi-label-classified_papers.csv", index=False)

cluster_export = df_out[df_out["non_amf"]].copy()

cluster_export.to_csv(
    "results/non_amf_clustered_definitions.csv",
    index=False
)

cluster_summary = (
    cluster_export
    .groupby("non_amf_cluster")
    .agg(
        count=("definition", "count"),
        papers=("title", "nunique")
    )
    .reset_index()
)

cluster_top_terms = {}

for c in sorted(cluster_export["non_amf_cluster"].dropna().unique()):
    mask = (cluster_export["non_amf_cluster"] == c).to_numpy()
    mean_tfidf = X[mask].mean(axis=0).A1
    top_terms = terms[np.argsort(mean_tfidf)[-8:]]
    cluster_top_terms[c] = ", ".join(top_terms)

cluster_summary["top_terms"] = cluster_summary["non_amf_cluster"].map(cluster_top_terms)

cluster_summary.to_csv(
    "results/non_amf_cluster_summary_with_terms.csv",
    index=False
)