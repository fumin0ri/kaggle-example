"""Project-wide constants used by every notebook."""


class Baseline:
    """Single source of truth for the validation setup."""

    TARGET = "Churn"
    ID_COLUMN = "id"
    FOLD_COLUMN = "fold"
    SEED = 42
    N_SPLITS = 5
    MIN_NUMERIC_UNIQUE = 5
