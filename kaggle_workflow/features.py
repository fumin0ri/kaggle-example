"""Competition-specific, target-free feature engineering.

Every transformation is deterministic and is fitted jointly on train and test only when it
uses counts. Those counts never use ``Transported``.  Feature blocks make ablation and greedy
selection cheap: the exact same code serves raw, basic, selected, and full experiments.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
FEATURE_BLOCKS = (
    "identity",
    "cabin",
    "group",
    "name",
    "age",
    "spend_total",
    "spend_log",
    "spend_ratio",
    "missing",
    "cross",
    "rules",
)

FEATURE_SETS = {
    "raw": (),
    "basic": ("identity", "cabin", "group", "age", "spend_total", "missing"),
    "full": FEATURE_BLOCKS,
}


def resolve_blocks(feature_set: str, blocks: Iterable[str] | None = None) -> tuple[str, ...]:
    """Resolve a named feature set or an explicit ordered block list."""
    if blocks is not None:
        resolved = tuple(dict.fromkeys(blocks))
    else:
        try:
            resolved = FEATURE_SETS[feature_set]
        except KeyError as exc:
            raise ValueError(f"Unknown feature_set={feature_set!r}") from exc
    unknown = set(resolved) - set(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")
    return resolved


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def create_features(
    df: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    feature_set: str = "full",
    blocks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create target-free candidates and return model-ready columns.

    ``reference`` should be the concatenated train/test feature frame. It is used only for
    unsupervised frequency features, so no label information can cross a validation fold.
    Exact passenger identifiers and names are removed after extracting useful components.
    """
    selected = set(resolve_blocks(feature_set, blocks))
    out = df.drop(columns=["Transported"], errors="ignore").copy()
    ref = (reference if reference is not None else df).drop(
        columns=["Transported"], errors="ignore"
    )

    passenger_parts = out["PassengerId"].astype("string").str.split("_", n=1, expand=True)
    ref_passenger_parts = ref["PassengerId"].astype("string").str.split("_", n=1, expand=True)
    group_id = passenger_parts[0]
    ref_group_id = ref_passenger_parts[0]

    cabin_parts = out["Cabin"].astype("string").str.split("/", n=2, expand=True)
    ref_cabin_parts = ref["Cabin"].astype("string").str.split("/", n=2, expand=True)
    surname = out["Name"].astype("string").str.split().str[-1]
    ref_surname = ref["Name"].astype("string").str.split().str[-1]

    if "identity" in selected:
        out["PassengerGroupNumber"] = pd.to_numeric(group_id, errors="coerce")
        out["PassengerNumberInGroup"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    if "cabin" in selected:
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNumber"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out["CabinDeckSide"] = cabin_parts[0].fillna("?") + "_" + cabin_parts[2].fillna("?")
        out["CabinNumberBand"] = (out["CabinNumber"] // 100).astype("Int64").astype("string")
        cabin_key = cabin_parts.fillna("?").agg("/".join, axis=1)
        ref_cabin_key = ref_cabin_parts.fillna("?").agg("/".join, axis=1)
        out["CabinOccupancy"] = cabin_key.map(ref_cabin_key.value_counts()).astype(float)

    if "group" in selected:
        out["GroupSize"] = group_id.map(ref_group_id.value_counts()).astype(float)
        out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
        out["IsGroupLeader"] = (passenger_parts[1] == "01").astype(int)

    if "name" in selected:
        out["NameLength"] = out["Name"].astype("string").str.len().astype(float)
        out["SurnameSize"] = surname.map(ref_surname.value_counts()).astype(float)
        out["SharedSurname"] = (out["SurnameSize"] > 1).astype(int)

    age = pd.to_numeric(out["Age"], errors="coerce")
    if "age" in selected:
        out["AgeSquared"] = age**2
        out["AgeBand"] = pd.cut(
            age,
            bins=[-np.inf, 5, 12, 17, 25, 35, 50, 65, np.inf],
            labels=["baby", "child", "teen", "young", "adult", "middle", "senior", "elder"],
        ).astype("string")
        out["IsChild"] = (age < 13).astype(int)
        out["IsMinor"] = (age < 18).astype(int)
        out["IsSenior"] = (age >= 60).astype(int)

    spend = out[SPEND_COLUMNS].apply(pd.to_numeric, errors="coerce")
    total_spend = spend.sum(axis=1, min_count=1)
    if "spend_total" in selected:
        out["TotalSpend"] = total_spend
        out["EssentialSpend"] = spend["RoomService"] + spend["FoodCourt"]
        out["LuxurySpend"] = spend["ShoppingMall"] + spend["Spa"] + spend["VRDeck"]
        out["SpendMean"] = spend.mean(axis=1)
        out["SpendStd"] = spend.std(axis=1)
        out["SpendMax"] = spend.max(axis=1)
        out["SpendServicesUsed"] = (spend.fillna(0) > 0).sum(axis=1)
        out["NoSpend"] = (spend.fillna(0).sum(axis=1) == 0).astype(int)

    if "spend_log" in selected:
        for column in SPEND_COLUMNS:
            out[f"Log1p{column}"] = np.log1p(spend[column].clip(lower=0))
        out["Log1pTotalSpend"] = np.log1p(total_spend.clip(lower=0))

    if "spend_ratio" in selected:
        for column in SPEND_COLUMNS:
            out[f"{column}Share"] = _safe_divide(spend[column], total_spend)
        out["SpendPerAge"] = _safe_divide(total_spend, age + 1)
        group_size = group_id.map(ref_group_id.value_counts()).astype(float)
        out["SpendPerGroupMember"] = _safe_divide(total_spend, group_size)

    if "missing" in selected:
        source_columns = [column for column in ref.columns if column != "Transported"]
        out["MissingCount"] = out[source_columns].isna().sum(axis=1)
        out["AnyMissing"] = (out["MissingCount"] > 0).astype(int)
        for column in ["Age", "Cabin", *SPEND_COLUMNS]:
            out[f"{column}Missing"] = out[column].isna().astype(int)

    if "cross" in selected:
        deck = cabin_parts[0].fillna("?")
        side = cabin_parts[2].fillna("?")
        home = out["HomePlanet"].astype("string").fillna("?")
        destination = out["Destination"].astype("string").fillna("?")
        out["Route"] = home + "_to_" + destination
        out["HomeDeck"] = home + "_" + deck
        out["DestinationDeck"] = destination + "_" + deck
        out["RouteSide"] = home + "_" + destination + "_" + side

    if "rules" in selected:
        cryo = (out["CryoSleep"].astype("string") == "True").fillna(False)
        vip = (out["VIP"].astype("string") == "True").fillna(False)
        no_spend = spend.fillna(0).sum(axis=1) == 0
        out["CryoNoSpendConsistent"] = (cryo & no_spend).astype(int)
        out["CryoHasSpendConflict"] = (cryo & ~no_spend).astype(int)
        out["VIPHasSpend"] = (vip & ~no_spend).astype(int)
        out["ChildInCryo"] = ((age < 13) & cryo).astype(int)

    drop_columns = ["PassengerId", "Name"]
    if "cabin" in selected:
        drop_columns.append("Cabin")
    return out.drop(columns=drop_columns, errors="ignore")
