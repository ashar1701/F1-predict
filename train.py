from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("data/processed/podium_dataset.csv")
MODEL_DIR = Path("artifacts/models")
REPORT_DIR = Path("artifacts/reports")
TARGET_COLUMN = "PodiumClass"
GROUP_COLUMNS = ["Year", "RoundNumber"]
ORDERED_PODIUM_POSITIONS = [1, 2, 3]
NUMERIC_FEATURES = [
    "QualifyingPosition",
    "GridPosition",
    "Q1",
    "Q2",
    "Q3",
    "DriverAvgFinishBefore",
    "DriverPodiumRateBefore",
    "TeamAvgFinishBefore",
    "DriverTrackAvgFinishBefore",
]
CATEGORICAL_FEATURES = [
    "Driver",
    "TeamName",
    "EventName",
    "Country",
    "Location",
]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run model.py first to build the training data."
        )

    dataset = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in dataset.columns:
        if "FinishPosition" not in dataset.columns:
            raise ValueError(f"{TARGET_COLUMN} was not found and FinishPosition is unavailable to derive it.")
        dataset[TARGET_COLUMN] = dataset["FinishPosition"].where(dataset["FinishPosition"].isin(ORDERED_PODIUM_POSITIONS), 0).astype(int)
    dataset = dataset.sort_values(["Year", "RoundNumber", "QualifyingPosition", "Driver"]).reset_index(drop=True)
    return dataset


def choose_features(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [column for column in NUMERIC_FEATURES if column in dataset.columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in dataset.columns]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found in the dataset.")

    return numeric_features, categorical_features

#selecting training set and test set
def split_train_test(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_years = sorted(dataset["Year"].dropna().astype(int).unique())
    if len(unique_years) >= 2:       #checking if there are at least 2 seasons in the dataset to do a year-based split
        test_year = unique_years[-1]   #using the most recent season as the test set to better evaluate how the model performs on new data, while training on past seasons to learn historical patterns. This simulates a real-world scenario where we want to predict future
        train_frame = dataset[dataset["Year"] < test_year].copy()
        test_frame = dataset[dataset["Year"] == test_year].copy()
    else:
        split_index = max(int(len(dataset) * 0.8), 1)   #if there are not enough seasons, do a random split of 80% train and 20% test. The max with 1 ensures that we don't end up with an empty training set if the dataset is very small.
        train_frame = dataset.iloc[:split_index].copy()  #.iloc() is usedto select rows. The :split_index means we take all rows from the start up to (but not including) the split_index for training, and from split_index to the end for testing. This way we ensure that we have a proper train/test split even if the dataset is small or has limited seasons.
        test_frame = dataset.iloc[split_index:].copy()

    if train_frame.empty or test_frame.empty:
        raise ValueError("Train/test split failed. Add more races or seasons to the dataset.")

    return train_frame, test_frame


def ordered_podium_from_qualifying(test_frame: pd.DataFrame) -> pd.Series:
    if "QualifyingPosition" not in test_frame.columns:
        raise ValueError("QualifyingPosition is required for the baseline.")

    ranked = test_frame.groupby(GROUP_COLUMNS)["QualifyingPosition"].rank(method="first", ascending=True)
    predictions = ranked.where(ranked <= 3, 0).fillna(0).astype(int)
    return pd.Series(predictions, index=test_frame.index)


def ordered_podium_from_probabilities(test_frame: pd.DataFrame, probability_frame: pd.DataFrame) -> pd.Series:
    predictions = pd.Series(0, index=test_frame.index, dtype=int)

    for _, race_indices in test_frame.groupby(GROUP_COLUMNS).groups.items():
        remaining_indices = list(race_indices)

        for position in ORDERED_PODIUM_POSITIONS:
            if position not in probability_frame.columns or not remaining_indices:
                continue

            best_index = probability_frame.loc[remaining_indices, position].idxmax()
            predictions.loc[best_index] = position
            remaining_indices.remove(best_index)

    return predictions


def exact_podium_order_accuracy(frame: pd.DataFrame, predictions: pd.Series) -> float:
    race_matches = 0
    race_total = 0

    evaluation = frame[GROUP_COLUMNS + ["Driver", TARGET_COLUMN]].copy()
    evaluation["PredictedClass"] = predictions

    for _, race in evaluation.groupby(GROUP_COLUMNS, sort=False):
        actual_order = tuple(
            race.loc[race[TARGET_COLUMN] > 0]
            .sort_values(TARGET_COLUMN)["Driver"]
            .tolist()
        )
        predicted_order = tuple(
            race.loc[race["PredictedClass"] > 0]
            .sort_values("PredictedClass")["Driver"]
            .tolist()
        )
        if len(actual_order) == len(ORDERED_PODIUM_POSITIONS):
            race_total += 1
            race_matches += int(actual_order == predicted_order)

    if race_total == 0:
        return 0.0

    return round(race_matches / race_total, 4)


def metric_summary(
    y_true: pd.Series,
    predictions: pd.Series,
    probabilities: pd.DataFrame | None = None,
) -> dict[str, Any]:
    podium_true = (y_true > 0).astype(int)
    podium_predictions = (predictions > 0).astype(int)
    winner_true = (y_true == 1).astype(int)
    winner_predictions = (predictions == 1).astype(int)

    summary: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_precision": round(
            float(precision_score(y_true, predictions, average="macro", zero_division=0)),
            4,
        ),
        "macro_recall": round(
            float(recall_score(y_true, predictions, average="macro", zero_division=0)),
            4,
        ),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4),
        "podium_precision": round(float(precision_score(podium_true, podium_predictions, zero_division=0)), 4),
        "podium_recall": round(float(recall_score(podium_true, podium_predictions, zero_division=0)), 4),
        "podium_f1": round(float(f1_score(podium_true, podium_predictions, zero_division=0)), 4),
        "winner_precision": round(float(precision_score(winner_true, winner_predictions, zero_division=0)), 4),
        "winner_recall": round(float(recall_score(winner_true, winner_predictions, zero_division=0)), 4),
    }

    if probabilities is not None and y_true.nunique() > 1:
        probability_columns = sorted(int(column) for column in probabilities.columns)
        try:
            summary["roc_auc_ovr"] = round(
                float(
                    roc_auc_score(
                        y_true,
                        probabilities[probability_columns],
                        multi_class="ovr",
                        labels=probability_columns,
                    )
                ),
                4,
            )
        except ValueError:
            pass

    return summary


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return model


def save_outputs(model: Pipeline, metrics: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "logistic_regression_ordered_podium.pkl"
    report_path = REPORT_DIR / "ordered_podium_metrics.json"

    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(metrics, report_file, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {report_path}")


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    numeric_features, categorical_features = choose_features(dataset)
    train_frame, test_frame = split_train_test(dataset)

    feature_columns = numeric_features + categorical_features
    x_train = train_frame[feature_columns]
    y_train = train_frame[TARGET_COLUMN]
    x_test = test_frame[feature_columns]
    y_test = test_frame[TARGET_COLUMN]

    baseline_predictions = ordered_podium_from_qualifying(test_frame)
    baseline_metrics = metric_summary(y_test, baseline_predictions)
    baseline_metrics["exact_podium_order_accuracy"] = exact_podium_order_accuracy(test_frame, baseline_predictions)

    model = build_model(numeric_features, categorical_features)
    model.fit(x_train, y_train)

    probability_array = model.predict_proba(x_test)
    probability_columns = [int(label) for label in model.named_steps["classifier"].classes_]
    probability_frame = pd.DataFrame(probability_array, index=test_frame.index, columns=probability_columns)
    model_predictions = ordered_podium_from_probabilities(test_frame, probability_frame)
    model_metrics = metric_summary(y_test, model_predictions, probability_frame)
    model_metrics["exact_podium_order_accuracy"] = exact_podium_order_accuracy(test_frame, model_predictions)

    metrics = {
        "dataset_path": str(DATASET_PATH),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_years": sorted(train_frame["Year"].dropna().astype(int).unique().tolist()),
        "test_years": sorted(test_frame["Year"].dropna().astype(int).unique().tolist()),
        "target_column": TARGET_COLUMN,
        "features": feature_columns,
        "baseline_ordered_qualifying": baseline_metrics,
        "logistic_regression_ordered_podium": model_metrics,
    }

    print("Baseline metrics (qualifying order -> podium order):")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nModel metrics (ordered-podium logistic regression):")
    print(json.dumps(model_metrics, indent=2))

    save_outputs(model, metrics)


if __name__ == "__main__":
    main()
