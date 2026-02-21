from enum import Enum, auto

class ScoringStrategy(Enum):
    MAX_SIMILARITY = auto()
    TOP_K_AVERAGE = auto()

mechanism_names = [
    "convexity",
    "asymmetry",
    "protective_robustness",
    "iterative_learning",
    "adaptive_restructuring",
    "growth_through_disorder",
    "adaptability_framed",
    "rore_extended",
]

MECH_COLS = [
    "convexity",
    "asymmetry",
    "protective_rore",
    "iterative_learning",
    "adaptive_restructuring",
    "growth_through_disorder",
    "adaptability_framed",
    "rore_extended",
]

MAIN_SIX = [
    "convexity",
    "asymmetry",
    "protective_rore",
    "iterative_learning",
    "adaptive_restructuring",
    "growth_through_disorder",
]

ANNS_COLUMNS = [
    "definition"
]

ANNS_MECH_COLS = [
    "Convex",
    "Asym",
    "ProRO",
    "learn",
    "restructure",
    "GTD",
    "AD",
    "RO/RE"
]

ENCODED_COLS = [
    "title",
    "definition",
]
