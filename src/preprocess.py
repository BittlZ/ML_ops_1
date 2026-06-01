from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from . import config

# The feature builder inserts ~160 columns one at a time; that is intentional
# and the resulting frame is defragmented before return. Silence the noisy
# (and here irrelevant) fragmentation warning.
warnings.simplefilter("ignore", PerformanceWarning)

KW_PREMIUM = ["times square", "near", "manhattan", "queens", "subway",
              "central", "midtown", "village"]
KW_BUDGET = ["sunny", "spacious", "brooklyn", "bedroom", "cozy",
             "small", "shared", "basic"]
KW_EXTRA = ["private", "room", "apt", "studio", "entire", "luxury",
            "modern", "beautiful", "quiet", "near"]

PAIR_COLS = [
    ("location_cluster", "type_house"),
    ("location", "type_house"),
    ("geo_k50", "type_house"),
    ("host_name", "type_house"),
]

NYC_LAT, NYC_LON = 40.7128, -74.0060
TSQ_LAT, TSQ_LON = 40.7580, -73.9855


def _pair_key(df: pd.DataFrame, c1: str, c2: str) -> np.ndarray:
    return (df[c1].astype(str) + "__" + df[c2].astype(str)).values


def _pct_rank_in_group(values: np.ndarray, keys: np.ndarray,
                       ref: dict) -> np.ndarray:
    """Percentile rank of each value within the train distribution of its group."""
    out = np.full(len(values), 0.5, dtype=float)
    keys = pd.Series(keys)
    for key, idx in keys.groupby(keys).groups.items():
        arr = ref.get(key)
        if arr is None or len(arr) == 0:
            continue
        pos = np.searchsorted(arr, values[idx.values], side="right")
        out[idx.values] = pos / len(arr)
    return out


class FeaturePipeline:
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        df = train.copy()
        df["host_name"] = df["host_name"].fillna("__nan__")
        df["name"] = df["name"].fillna("")
        y = df[config.TARGET_COL].values.astype(float)
        self.global_mean = float(y.mean())

        self.sum_clip = float(df["sum"].quantile(0.99))
        self.ref_date = pd.to_datetime(df["last_dt"], errors="coerce").max()

        df["host_uid"] = df["host_name"] + "_" + df["location_cluster"].astype(str)
        self.freq = {c: df[c].value_counts().to_dict()
                     for c in ["location", "type_house", "location_cluster", "host_uid"]}

        self.id_listing_count = df.groupby("_id")["host_name"].count().to_dict()
        self.id_freq = df["_id"].value_counts().to_dict()
        agg = df.groupby("_id").agg(
            id_mean_sum=("sum", "mean"), id_std_sum=("sum", "std"),
            id_mean_reviews=("amt_reviews", "mean"),
            id_mean_min_days=("min_days", "mean"),
            id_n_locations=("location", "nunique"),
            id_n_types=("type_house", "nunique"),
        )
        agg["id_std_sum"] = agg["id_std_sum"].fillna(0)
        self.agg_id = {c: agg[c].to_dict() for c in agg.columns}
        self.agg_id_global = {
            "id_mean_sum": float(df["sum"].mean()), "id_std_sum": 0.0,
            "id_mean_reviews": float(df["amt_reviews"].mean()),
            "id_mean_min_days": float(df["min_days"].mean()),
            "id_n_locations": 1.0, "id_n_types": 1.0,
        }

        self.group_sum_stats = {}
        for col in ["location_cluster", "location", "type_house"]:
            g = df.groupby(col)["sum"].agg(["mean", "std"])
            self.group_sum_stats[col] = (g["mean"].to_dict(), g["std"].to_dict())
        self.location_median = df.groupby("location")["sum"].median().to_dict()
        self.global_sum_mean = float(df["sum"].mean())
        self.global_sum_std = float(df["sum"].std())

        self.id_rank_ref = np.sort(df["_id"].values)
        self.rank_sum_loc = {k: np.sort(v.values)
                             for k, v in df.groupby("location")["sum"]}
        self.rank_sum_type = {k: np.sort(v.values)
                              for k, v in df.groupby("type_house")["sum"]}
        self.rank_rev_loc = {k: np.sort(v.values)
                             for k, v in df.groupby("location")["amt_reviews"]}

        sub = df[["_id", "sum", "type_house"]].copy()
        type_map = {"Entire home/apt": 3, "Private room": 2,
                    "Shared room": 1, "Hotel room": 2.5}
        sub["type_encoded"] = sub["type_house"].map(type_map).fillna(2)
        multi = sub.groupby("_id").filter(lambda x: len(x) >= 2)
        if len(multi) > 0:
            hp = multi.groupby("_id").agg(
                host_price_range=("sum", lambda x: x.max() - x.min()),
                host_type_diversity=("type_encoded", "nunique"))
            self.host_price_range = hp["host_price_range"].to_dict()
            self.host_type_diversity = hp["host_type_diversity"].to_dict()
            self.host_price_range_med = float(hp["host_price_range"].median())
            self.host_type_diversity_med = float(hp["host_type_diversity"].median())
        else:
            self.host_price_range, self.host_type_diversity = {}, {}
            self.host_price_range_med = self.host_type_diversity_med = 0.0

        coords = df[["lat", "lon"]].values
        self.kmeans = {}
        self.geo_centers = {}
        self.geo_freq = {}
        geo_labels = {}
        for k in config.GEO_KS:
            km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
            labels = km.fit_predict(coords)
            self.kmeans[k] = km
            self.geo_centers[k] = km.cluster_centers_
            self.geo_freq[k] = pd.Series(labels).value_counts().to_dict()
            geo_labels[k] = labels


        dslr = self._days_since(df)
        has_rev = df["last_dt"].notna().astype(int).values
        vitality = has_rev * np.clip(365 - np.clip(dslr, 0, None), 0, None) / 365
        vitality = np.where(dslr == -1, 0, vitality)
        self.id_mean_vitality = (
            pd.DataFrame({"_id": df["_id"].values, "v": vitality})
            .groupby("_id")["v"].mean().to_dict())
        self.global_vitality = float(np.mean(vitality))


        self.density_tree = BallTree(np.radians(coords), metric="haversine")

        tdf = pd.DataFrame({"_id": df["_id"].values, "target": y})
        self.id_full_mean = tdf.groupby("_id")["target"].mean().to_dict()
        g = tdf.groupby("_id")["target"]
        id_te = pd.DataFrame({
            "id_te_mean": g.mean(), "id_te_median": g.median(),
            "id_te_std": g.std().fillna(0), "id_te_min": g.min(),
            "id_te_max": g.max(), "id_te_count": g.count(),
            "id_te_zero_frac": g.apply(lambda x: (x == 0).mean()),
            "id_te_high_frac": g.apply(lambda x: (x >= 300).mean()),
            "id_te_q25": g.quantile(0.25), "id_te_q75": g.quantile(0.75),
        })
        id_te["id_te_range"] = id_te["id_te_max"] - id_te["id_te_min"]
        id_te["id_te_iqr"] = id_te["id_te_q75"] - id_te["id_te_q25"]
        self.id_te = {c: id_te[c].to_dict() for c in id_te.columns}
        self.id_te_cols = list(id_te.columns)

        self.geo_te = {}
        for k in config.GEO_KS:
            gt = pd.DataFrame({"g": geo_labels[k], "target": y})
            self.geo_te[k] = {
                "mean": gt.groupby("g")["target"].mean().to_dict(),
                "zero_frac": gt.groupby("g")["target"].apply(lambda x: (x == 0).mean()).to_dict(),
                "high_frac": gt.groupby("g")["target"].apply(lambda x: (x >= 300).mean()).to_dict(),
            }

        lt = pd.DataFrame({"l": df["location"].values, "target": y})
        self.loc_te = {
            "loc_te_mean": lt.groupby("l")["target"].mean().to_dict(),
            "loc_te_high_frac": lt.groupby("l")["target"].apply(lambda x: (x >= 300).mean()).to_dict(),
            "loc_te_zero_frac": lt.groupby("l")["target"].apply(lambda x: (x == 0).mean()).to_dict(),
        }

        df["geo_k50"] = geo_labels[50]
        self.cross_te = {}
        for c1, c2 in PAIR_COLS:
            keys = _pair_key(df, c1, c2)
            cdf = pd.DataFrame({"k": keys, "target": y})
            self.cross_te[f"crossTE_{c1}_x_{c2}"] = cdf.groupby("k")["target"].mean().to_dict()

        self.knn_tree = BallTree(np.radians(coords), metric="haversine")
        self.knn_y = y.copy()

        feats = self._assemble(df)
        feature_cols = [c for c in feats.columns
                        if c not in config.DROP_COLS and c != config.TARGET_COL
                        and c not in ("geo_k50",)]
        feature_cols = [c for c in feats.columns
                        if c not in config.DROP_COLS and c != config.TARGET_COL]
        self.feature_columns = feature_cols

        num_cols = [c for c in feature_cols if c not in config.CAT_FEATURES]
        self.medians = {c: float(pd.to_numeric(feats[c], errors="coerce").median())
                        for c in num_cols}

        self.label_maps = {}
        for c in config.CAT_FEATURES:
            cats = feats[c].astype(str).unique().tolist()
            self.label_maps[c] = {v: i for i, v in enumerate(sorted(cats))}

        self.fitted = True
        return self

    def _days_since(self, df: pd.DataFrame) -> np.ndarray:
        parsed = pd.to_datetime(df["last_dt"], errors="coerce")
        days = (self.ref_date - parsed).dt.days
        return days.fillna(-1).values

    def _map(self, series: pd.Series, mapping: dict, default: float) -> np.ndarray:
        return series.map(mapping).fillna(default).values

    def _assemble(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["host_name"] = df["host_name"].fillna("__nan__")
        df["name"] = df["name"].fillna("")

        df["has_reviews"] = df["last_dt"].notna().astype(int)
        df["avg_reviews"] = df["avg_reviews"].fillna(0)
        dslr = self._days_since(df)
        df["days_since_last_review"] = dslr
        parsed = pd.to_datetime(df["last_dt"], errors="coerce")
        df["review_year"] = parsed.dt.year.fillna(0).astype(int)
        df["review_month"] = parsed.dt.month.fillna(0).astype(int)

        df["sum"] = df["sum"].clip(upper=self.sum_clip)
        df["min_days"] = df["min_days"].clip(upper=365)
        for col in ["sum", "min_days", "amt_reviews", "avg_reviews", "total_host"]:
            df[f"log_{col}"] = np.log1p(df[col])

        df["dist_to_center"] = np.sqrt((df["lat"] - NYC_LAT) ** 2 + (df["lon"] - NYC_LON) ** 2)
        df["dist_to_times_sq"] = np.sqrt((df["lat"] - TSQ_LAT) ** 2 + (df["lon"] - TSQ_LON) ** 2)
        df["sum_per_min_day"] = df["sum"] / df["min_days"].clip(lower=1)
        df["reviews_per_host_listing"] = df["amt_reviews"] / df["total_host"].clip(lower=1)
        df["is_professional_host"] = (df["total_host"] > 5).astype(int)
        df["id_rank"] = np.searchsorted(self.id_rank_ref, df["_id"].values, side="right") / len(self.id_rank_ref)
        df["log_id"] = np.log1p(df["_id"])

        rb = np.zeros(len(df), dtype=float)
        rb[(dslr >= 0) & (dslr <= 30)] = 4
        rb[(dslr > 30) & (dslr <= 180)] = 3
        rb[(dslr > 180) & (dslr <= 365)] = 2
        rb[dslr > 365] = 1
        df["recency_bucket"] = rb

        df["name_len"] = df["name"].str.len()
        df["name_word_count"] = df["name"].str.split().str.len().fillna(0)
        df["name_upper_ratio"] = df["name"].apply(
            lambda x: sum(c.isupper() for c in x) / max(len(x), 1))

        df["host_uid"] = df["host_name"] + "_" + df["location_cluster"].astype(str)
        for col in ["location", "type_house", "location_cluster", "host_uid"]:
            df[f"{col}_freq"] = self._map(df[col], self.freq[col], 0)

        df["id_listing_count"] = self._map(df["_id"], self.id_listing_count, 0)
        df["id_freq"] = self._map(df["_id"], self.id_freq, 0)
        for c, mapping in self.agg_id.items():
            df[c] = self._map(df["_id"], mapping, self.agg_id_global[c])

        vitality = df["has_reviews"].values * np.clip(365 - np.clip(dslr, 0, None), 0, None) / 365
        vitality = np.where(dslr == -1, 0, vitality)
        df["listing_vitality"] = vitality
        df["host_activity_score"] = df["log_amt_reviews"] * df["recency_bucket"]
        df["id_mean_vitality"] = self._map(df["_id"], self.id_mean_vitality, self.global_vitality)

        for col in ["location_cluster", "location", "type_house"]:
            mean_map, std_map = self.group_sum_stats[col]
            df[f"{col}_sum_mean"] = self._map(df[col], mean_map, self.global_sum_mean)
            df[f"{col}_sum_std"] = self._map(df[col], std_map, self.global_sum_std)
        df["price_zscore_location"] = (df["sum"] - df["location_sum_mean"]) / df["location_sum_std"].clip(lower=1)

        df["listing_age_months"] = df["amt_reviews"] / df["avg_reviews"].clip(lower=0.01)
        df.loc[df["avg_reviews"] == 0, "listing_age_months"] = 0
        df["listing_age_months"] = df["listing_age_months"].clip(upper=240)
        df["first_review_days_ago"] = df["days_since_last_review"] + df["listing_age_months"] * 30
        df.loc[df["days_since_last_review"] == -1, "first_review_days_ago"] = -1
        df["min_booking_value"] = df["sum"] * df["min_days"]
        df["log_min_booking_value"] = np.log1p(df["min_booking_value"])
        df["annual_review_rate"] = df["avg_reviews"] * 12
        df["price_per_review"] = df["sum"] / (df["amt_reviews"] + 1)
        df["host_revenue_proxy"] = df["sum"] * df["total_host"]
        df["log_host_revenue"] = np.log1p(df["host_revenue_proxy"])
        df["min_days_x_total_host"] = df["min_days"] * df["total_host"]
        df["reviews_density"] = df["amt_reviews"] / df["listing_age_months"].clip(lower=1)

        coords = df[["lat", "lon"]].values
        for k in config.GEO_KS:
            labels = self.kmeans[k].predict(coords)
            df[f"geo_k{k}"] = labels
            df[f"geo_k{k}_freq"] = pd.Series(labels).map(self.geo_freq[k]).fillna(0).values
            centers = self.geo_centers[k]
            df[f"geo_k{k}_dist"] = np.sqrt(
                (df["lat"].values - centers[labels, 0]) ** 2 +
                (df["lon"].values - centers[labels, 1]) ** 2)

        name_lower = df["name"].str.lower()
        for kw in KW_PREMIUM:
            df[f"kw_p_{kw.replace(' ', '_')}"] = name_lower.str.contains(kw, na=False).astype(int)
        for kw in KW_BUDGET:
            df[f"kw_b_{kw}"] = name_lower.str.contains(kw, na=False).astype(int)
        df["kw_premium_count"] = df[[c for c in df.columns if c.startswith("kw_p_")]].sum(axis=1)
        df["kw_budget_count"] = df[[c for c in df.columns if c.startswith("kw_b_")]].sum(axis=1)
        df["kw_premium_ratio"] = df["kw_premium_count"] / (df["kw_premium_count"] + df["kw_budget_count"] + 1)
        for kw in KW_EXTRA:
            df[f"kw_{kw}"] = name_lower.str.contains(kw, na=False).astype(int)
        df["name_has_number"] = name_lower.str.contains(r"\d+\s*(?:br|bed|bath|room)", na=False).astype(int)
        df["name_exclamation"] = df["name"].str.contains("!", na=False).astype(int)
        df["name_dash_count"] = df["name"].str.count("-")
        df["name_capital_word_count"] = df["name"].apply(
            lambda x: sum(1 for w in x.split() if w and w[0].isupper()))

        type_map = {"Entire home/apt": 3, "Private room": 2, "Shared room": 1, "Hotel room": 2.5}
        df["type_encoded"] = df["type_house"].map(type_map).fillna(2)
        df["sum_x_type"] = df["sum"] * df["type_encoded"]
        df["min_days_x_type"] = df["min_days"] * df["type_encoded"]
        df["reviews_x_location_freq"] = df["amt_reviews"] * df["location_freq"]
        df["sum_rank_in_location"] = _pct_rank_in_group(
            df["sum"].values, df["location"].values, self.rank_sum_loc)
        df["price_vs_location_median"] = df["sum"] / pd.Series(
            self._map(df["location"], self.location_median, self.global_sum_mean)).clip(lower=1).values
        df["sum_rank_in_type"] = _pct_rank_in_group(
            df["sum"].values, df["type_house"].values, self.rank_sum_type)
        df["reviews_rank_in_location"] = _pct_rank_in_group(
            df["amt_reviews"].values, df["location"].values, self.rank_rev_loc)

        df["host_price_range"] = self._map(df["_id"], self.host_price_range, self.host_price_range_med)
        df["host_type_diversity"] = self._map(df["_id"], self.host_type_diversity, self.host_type_diversity_med)

        df["no_reviews"] = (df["amt_reviews"] == 0).astype(int)
        df["few_reviews"] = (df["amt_reviews"] <= 2).astype(int)
        df["high_price"] = (df["sum"] > 200).astype(int)
        df["long_min_days"] = (df["min_days"] > 14).astype(int)
        df["very_long_min_days"] = (df["min_days"] > 30).astype(int)
        df["old_or_no_review"] = ((df["days_since_last_review"] > 365) |
                                  (df["days_since_last_review"] == -1)).astype(int)
        df["is_shared"] = (df["type_house"] == "Shared room").astype(int)
        df["dead_listing_score"] = df["no_reviews"] + df["long_min_days"] + df["high_price"] + df["old_or_no_review"]
        df["dead_listing_v2"] = df["few_reviews"] + df["very_long_min_days"] + df["high_price"] + df["old_or_no_review"]
        df["pro_host_x_min_days"] = df["total_host"] * df["min_days"]
        df["pro_host_x_no_reviews"] = df["total_host"] * df["no_reviews"]
        df["pro_host_x_high_price"] = df["total_host"] * df["high_price"]
        df["log_total_host_x_log_min_days"] = df["log_total_host"] * df["log_min_days"]
        df["host_scale"] = pd.cut(df["total_host"], bins=[-1, 1, 5, 20, 50, 9999],
                                  labels=[0, 1, 2, 3, 4]).astype(float)
        df["review_staleness"] = np.where(
            df["listing_age_months"] > 0,
            df["days_since_last_review"].clip(lower=0) / (df["listing_age_months"] * 30).clip(lower=1),
            np.where(df["days_since_last_review"] == -1, 2.0, 0.0))
        df["review_staleness"] = df["review_staleness"].clip(upper=5)
        df["shared_low_price"] = (df["is_shared"] & (df["sum"] < 50)).astype(int)
        df["shared_x_price"] = df["is_shared"] * df["sum"]

        radii_coords = np.radians(df[["lat", "lon"]].values)
        for r in [0.5, 1.0, 2.0]:
            df[f"density_{r}km"] = self.density_tree.query_radius(
                radii_coords, r=r / 6371.0, count_only=True)

        df["min_days_bucket"] = pd.cut(df["min_days"], bins=[0, 1, 3, 7, 14, 30, 90, 365],
                                       labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(0)

        df["id_loo_te"] = self._map(df["_id"], self.id_full_mean, self.global_mean)
        for c in self.id_te_cols:
            default = self.global_mean if c == "id_te_mean" else (0.5 if "frac" in c else 0)
            df[c] = self._map(df["_id"], self.id_te[c], default)
        for k in config.GEO_KS:
            lbl = df[f"geo_k{k}"]
            df[f"geo_k{k}_te_mean"] = pd.Series(lbl).map(self.geo_te[k]["mean"]).fillna(self.global_mean).values
            df[f"geo_k{k}_te_zero_frac"] = pd.Series(lbl).map(self.geo_te[k]["zero_frac"]).fillna(0.5).values
            df[f"geo_k{k}_te_high_frac"] = pd.Series(lbl).map(self.geo_te[k]["high_frac"]).fillna(0.5).values
        df["loc_te_mean"] = self._map(df["location"], self.loc_te["loc_te_mean"], self.global_mean)
        df["loc_te_high_frac"] = self._map(df["location"], self.loc_te["loc_te_high_frac"], 0.5)
        df["loc_te_zero_frac"] = self._map(df["location"], self.loc_te["loc_te_zero_frac"], 0.5)

        knn_q = self.knn_tree.query(np.radians(df[["lat", "lon"]].values), k=max(config.K_LIST))
        neigh_idx = knn_q[1]
        for k in config.K_LIST:
            vals = self.knn_y[neigh_idx[:, :k]]
            df[f"knn_te_k{k}"] = vals.mean(axis=1)
            df[f"knn_te_k{k}_std"] = vals.std(axis=1)

        for c1, c2 in PAIR_COLS:
            name = f"crossTE_{c1}_x_{c2}"
            keys = _pair_key(df, c1, c2)
            df[name] = pd.Series(keys).map(self.cross_te[name]).fillna(self.global_mean).values

        # Defragment (many incremental column inserts above) to avoid pandas
        # PerformanceWarnings and keep downstream slicing fast.
        return df.copy()

    def transform(self, raw: pd.DataFrame) -> dict:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline должен быть обучен перед вызовом transform().")
        feats = self._assemble(raw)

        X = feats.reindex(columns=self.feature_columns).copy()
        for c in config.CAT_FEATURES:
            X[c] = X[c].astype(str).fillna("nan")
        for c in self.feature_columns:
            if c in config.CAT_FEATURES:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.medians.get(c, 0.0))

        X_le = X.copy()
        for c in config.CAT_FEATURES:
            mapping = self.label_maps[c]
            X_le[c] = X_le[c].map(mapping).fillna(-1).astype(int)

        X_rf = X_le.astype(float)

        return {"cat": X, "le": X_le, "rf": X_rf}

    def build_training_matrix(self, train: pd.DataFrame) -> dict:
        """Return model-ready views, target and host groups for training."""
        views = self.transform(train)
        y = np.clip(train[config.TARGET_COL].values.astype(float),
                    config.TARGET_MIN, config.TARGET_MAX)
        groups = train["host_name"].fillna("__nan__").values
        views["y"] = y
        views["groups"] = groups
        return views
