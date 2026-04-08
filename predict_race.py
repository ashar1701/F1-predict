from __future__ import annotations

import pickle
from pathlib import Path

import fastf1
import pandas as pd

from train import CATEGORICAL_FEATURES, GROUP_COLUMNS, NUMERIC_FEATURES, ORDERED_PODIUM_POSITIONS


CACHE_DIR = Path("cache")
DATASET_PATH = Path("data/processed/podium_dataset.csv")
MODEL_PATH = Path("artifacts/models/logistic_regression_ordered_podium.pkl")
TARGET_YEAR = 2026
TARGET_ROUND = 1


def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def timedeltas_to_seconds(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].dt.total_seconds()
    return frame


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the ordered podium model first.")

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def load_history(dataset_path: Path, year: int, round_number: int) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Build the training dataset first.")

    dataset = pd.read_csv(dataset_path)
    history = dataset[(dataset["Year"] < year) | ((dataset["Year"] == year) & (dataset["RoundNumber"] < round_number))].copy()
    history = history.sort_values(["Year", "RoundNumber", "QualifyingPosition", "Driver"]).reset_index(drop=True)
    return history


def build_feature_frame(year: int, round_number: int, history: pd.DataFrame) -> pd.DataFrame:
    qualifying = fastf1.get_session(year, round_number, "Q")
    qualifying.load(laps=False, telemetry=False, weather=False, messages=False)

    qualifying_results = qualifying.results.copy().reset_index(drop=True)
    qualifying_results = timedeltas_to_seconds(qualifying_results, ["Q1", "Q2", "Q3"])
    qualifying_columns = [
        "DriverNumber",
        "Abbreviation",
        "FullName",
        "TeamName",
        "Position",
        "Q1",
        "Q2",
        "Q3",
    ]
    feature_frame = qualifying_results[
        [column for column in qualifying_columns if column in qualifying_results.columns]
    ].rename(columns={"Position": "QualifyingPosition"})

    feature_frame["Year"] = year
    feature_frame["RoundNumber"] = round_number
    feature_frame["EventName"] = qualifying.event["EventName"]
    feature_frame["Country"] = qualifying.event["Country"]
    feature_frame["Location"] = qualifying.event["Location"]
    feature_frame["Driver"] = feature_frame.get("Abbreviation", feature_frame.get("DriverNumber"))
    feature_frame["GridPosition"] = pd.to_numeric(feature_frame["QualifyingPosition"], errors="coerce")

    numeric_columns = ["QualifyingPosition", "GridPosition", "Q1", "Q2", "Q3"]
    for column in numeric_columns:
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    driver_avg_finish = history.groupby("Driver")["FinishPosition"].mean() if not history.empty else pd.Series(dtype=float)
    driver_podium_rate = history.groupby("Driver")["IsPodium"].mean() if not history.empty else pd.Series(dtype=float)
    team_avg_finish = history.groupby("TeamName")["FinishPosition"].mean() if not history.empty else pd.Series(dtype=float)
    driver_track_avg_finish = (
        history.groupby(["Driver", "Location"])["FinishPosition"].mean() if not history.empty else pd.Series(dtype=float)
    )

    feature_frame["DriverAvgFinishBefore"] = feature_frame["Driver"].map(driver_avg_finish)
    feature_frame["DriverPodiumRateBefore"] = feature_frame["Driver"].map(driver_podium_rate)
    feature_frame["TeamAvgFinishBefore"] = feature_frame["TeamName"].map(team_avg_finish)
    feature_frame["DriverTrackAvgFinishBefore"] = [
        driver_track_avg_finish.get((driver, location), pd.NA)
        for driver, location in zip(feature_frame["Driver"], feature_frame["Location"])
    ]

    return feature_frame


def ordered_podium_from_probabilities(frame: pd.DataFrame, probability_frame: pd.DataFrame) -> pd.Series:
    predictions = pd.Series(0, index=frame.index, dtype=int)

    for _, race_indices in frame.groupby(GROUP_COLUMNS).groups.items():
        remaining_indices = list(race_indices)
        for position in ORDERED_PODIUM_POSITIONS:
            if position not in probability_frame.columns or not remaining_indices:
                continue
            best_index = probability_frame.loc[remaining_indices, position].idxmax()
            predictions.loc[best_index] = position
            remaining_indices.remove(best_index)

    return predictions


def main() -> None:
    ensure_cache()
    model = load_model(MODEL_PATH)
    history = load_history(DATASET_PATH, TARGET_YEAR, TARGET_ROUND)
    feature_frame = build_feature_frame(TARGET_YEAR, TARGET_ROUND, history)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    probability_array = model.predict_proba(feature_frame[feature_columns])
    probability_columns = [int(label) for label in model.named_steps["classifier"].classes_]
    probability_frame = pd.DataFrame(probability_array, index=feature_frame.index, columns=probability_columns)
    predictions = ordered_podium_from_probabilities(feature_frame, probability_frame)

    output = feature_frame[["Driver", "TeamName", "QualifyingPosition", "GridPosition"]].copy()
    output["WinProbability"] = probability_frame.get(1, 0.0)
    output["P2Probability"] = probability_frame.get(2, 0.0)
    output["P3Probability"] = probability_frame.get(3, 0.0)
    output["PodiumProbability"] = output[["WinProbability", "P2Probability", "P3Probability"]].sum(axis=1)
    output["PredictedPodiumPosition"] = predictions

    podium_prediction = output.loc[output["PredictedPodiumPosition"] > 0].sort_values("PredictedPodiumPosition")

    print(f"Predicted podium for {TARGET_YEAR} round {TARGET_ROUND}:")
    if podium_prediction.empty:
        print("No ordered podium prediction was produced.")
    else:
        print(podium_prediction.to_string(index=False))

    print("\nTop podium probability candidates:")
    print(output.sort_values("PodiumProbability", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()