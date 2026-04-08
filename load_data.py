from __future__ import annotations

from pathlib import Path
import warnings
import fastf1
import pandas as pd


CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("data/processed")
DATASET_PATH = OUTPUT_DIR / "podium_dataset.csv"
START_YEAR = 2005
END_YEAR = 2007
ROLLING_FEATURE_COLUMNS = [
	"DriverAvgFinishBefore",
	"DriverPodiumRateBefore",
	"TeamAvgFinishBefore",
	"DriverTrackAvgFinishBefore",
]


def ensure_cache() -> None:
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	fastf1.Cache.enable_cache(str(CACHE_DIR))


def timedeltas_to_seconds(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	for column in columns:
		if column in frame.columns:
			frame[column] = frame[column].dt.total_seconds()
	return frame


def coerce_numeric_columns(
	frame: pd.DataFrame,
	columns: list[str],
	*,
	year: int,
	round_number: int,
	frame_name: str,
) -> pd.DataFrame:
	for column in columns:
		if column not in frame.columns:
			continue

		original = frame[column]
		converted = pd.to_numeric(original, errors="coerce")
		introduced_missing = int(converted.isna().sum() - original.isna().sum())
		if introduced_missing > 0:
			warnings.warn(
				f"{year} round {round_number}: {frame_name}.{column} had {introduced_missing} non-numeric value(s) coerced to NaN.",
				stacklevel=2,
			)
		frame[column] = converted

	return frame


def validate_merge_frame(
	frame: pd.DataFrame,
	merge_keys: list[str],
	*,
	year: int,
	round_number: int,
	frame_name: str,
) -> pd.DataFrame:
	missing_key_rows = frame[merge_keys].isna().any(axis=1)
	missing_count = int(missing_key_rows.sum())
	if missing_count > 0:
		warnings.warn(
			f"{year} round {round_number}: dropping {missing_count} {frame_name} row(s) with missing merge keys {merge_keys}.",
			stacklevel=2,
		)
		frame = frame.loc[~missing_key_rows].copy()

	if frame.empty:
		raise ValueError(f"{year} round {round_number}: no {frame_name} rows remain after merge-key validation.")

	duplicate_rows = frame.duplicated(subset=merge_keys, keep=False)
	duplicate_count = int(duplicate_rows.sum())
	if duplicate_count > 0:
		duplicate_examples = frame.loc[duplicate_rows, merge_keys].drop_duplicates().head(5).to_dict("records")
		raise ValueError(
			f"{year} round {round_number}: {frame_name} has duplicate merge keys for {duplicate_count} row(s). "
			f"Examples: {duplicate_examples}"
		)

	return frame


def drop_rows_missing_required_values(
	frame: pd.DataFrame,
	required_columns: list[str],
	*,
	year: int,
	round_number: int,
) -> pd.DataFrame:
	missing_columns = [column for column in required_columns if column not in frame.columns]
	if missing_columns:
		raise ValueError(
			f"{year} round {round_number}: missing required columns after merge: {missing_columns}"
		)

	missing_summary = {
		column: int(frame[column].isna().sum())
		for column in required_columns
		if int(frame[column].isna().sum()) > 0
	}
	if missing_summary:
		invalid_rows = frame[required_columns].isna().any(axis=1)
		warnings.warn(
			f"{year} round {round_number}: dropping {int(invalid_rows.sum())} merged row(s) with missing required values {missing_summary}.",
			stacklevel=2,
		)
		frame = frame.loc[~invalid_rows].copy()

	if frame.empty:
		raise ValueError(f"{year} round {round_number}: no rows remain after required-value validation.")

	return frame


def load_existing_base_dataset(dataset_path: Path) -> pd.DataFrame:
	if not dataset_path.exists():
		return pd.DataFrame()

	existing_dataset = pd.read_csv(dataset_path)
	columns_to_drop = [column for column in ROLLING_FEATURE_COLUMNS if column in existing_dataset.columns]
	if columns_to_drop:
		existing_dataset = existing_dataset.drop(columns=columns_to_drop)

	return existing_dataset


def deduplicate_weekends(dataset: pd.DataFrame) -> pd.DataFrame:
	deduplication_keys = [
		column
		for column in ["Year", "RoundNumber", "DriverNumber", "Abbreviation", "Driver"]
		if column in dataset.columns
	]
	if not deduplication_keys:
		raise ValueError("No deduplication keys were found in the combined dataset.")

	dataset = dataset.drop_duplicates(subset=deduplication_keys, keep="last")
	return dataset.reset_index(drop=True)


def combine_with_existing_dataset(batch_dataset: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
	existing_dataset = load_existing_base_dataset(dataset_path)
	if existing_dataset.empty:
		return deduplicate_weekends(batch_dataset.copy())

	combined_dataset = pd.concat([existing_dataset, batch_dataset], ignore_index=True)
	return deduplicate_weekends(combined_dataset)


def add_rolling_features(dataset: pd.DataFrame) -> pd.DataFrame:
#sort the dataset to ensure that the caclulations are done in correct order
	dataset = dataset.sort_values(["Year", "RoundNumber", "QualifyingPosition", "Driver"]).reset_index(drop=True)


	dataset["DriverAvgFinishBefore"] = (
		dataset.groupby("Driver")["FinishPosition"]
		.transform(lambda values: values.shift(1).expanding().mean()) # shifting the values to ensure that we are always 
		# calculating the average based on the previous races
	)
	dataset["DriverPodiumRateBefore"] = (
		dataset.groupby("Driver")["IsPodium"]
		.transform(lambda values: values.shift(1).expanding().mean())  # transform adds the lambda function to each group and returns a series  so you get a column back
	)
	dataset["TeamAvgFinishBefore"] = (
		dataset.groupby("TeamName")["FinishPosition"]   
		.transform(lambda values: values.shift(1).expanding().mean())
	)
	dataset["DriverTrackAvgFinishBefore"] = (
		dataset.groupby(["Driver", "Location"])["FinishPosition"]
		.transform(lambda values: values.shift(1).expanding().mean())
	)

	return dataset


def load_session(year: int, round_number: int, session_code: str):
	session = fastf1.get_session(year, round_number, session_code)
	session.load(laps=False, telemetry=False, weather=False, messages=False)
	return session


def build_race_weekend_dataset(year: int, round_number: int) -> pd.DataFrame:
	qualifying = load_session(year, round_number, "Q")
	race = load_session(year, round_number, "R")

	qualifying_results = qualifying.results.copy().reset_index(drop=True)
	race_results = race.results.copy().reset_index(drop=True)

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
	race_columns = [
		"DriverNumber",
		"Abbreviation",
		"GridPosition",
		"Position",
		"Points",
		"Status",
	]

	qualifying_results = qualifying_results[[column for column in qualifying_columns if column in qualifying_results.columns]].rename(
		columns={"Position": "QualifyingPosition"}
	)
	race_results = race_results[[column for column in race_columns if column in race_results.columns]].rename(
		columns={"Position": "FinishPosition"}
	)

	qualifying_results = coerce_numeric_columns(
		qualifying_results,
		["QualifyingPosition"],
		year=year,
		round_number=round_number,
		frame_name="qualifying_results",
	)
	race_results = coerce_numeric_columns(
		race_results,
		["GridPosition", "FinishPosition", "Points"],
		year=year,
		round_number=round_number,
		frame_name="race_results",
	)

	merge_keys = [column for column in ["DriverNumber", "Abbreviation"] if column in qualifying_results.columns and column in race_results.columns]
	if not merge_keys:
		raise ValueError(f"No merge keys found for {year} round {round_number}")

	qualifying_results = validate_merge_frame(
		qualifying_results,
		merge_keys,
		year=year,
		round_number=round_number,
		frame_name="qualifying_results",
	)
	race_results = validate_merge_frame(
		race_results,
		merge_keys,
		year=year,
		round_number=round_number,
		frame_name="race_results",
	)

	weekend = qualifying_results.merge(race_results, on=merge_keys, how="inner", validate="one_to_one")
	if len(weekend) < min(len(qualifying_results), len(race_results)):
		warnings.warn(
			f"{year} round {round_number}: merge kept {len(weekend)} rows from {len(qualifying_results)} qualifying rows and {len(race_results)} race rows.",
			stacklevel=2,
		)

	weekend["Year"] = year
	weekend["RoundNumber"] = round_number
	weekend["EventName"] = race.event["EventName"]
	weekend["Country"] = race.event["Country"]
	weekend["Location"] = race.event["Location"]
	weekend["Driver"] = weekend.get("Abbreviation", weekend.get("DriverNumber"))

	weekend = drop_rows_missing_required_values(
		weekend,
		["QualifyingPosition", "FinishPosition"],
		year=year,
		round_number=round_number,
	)

	weekend["IsPodium"] = (weekend["FinishPosition"] <= 3).astype(int)
	weekend["PodiumClass"] = weekend["FinishPosition"].where(weekend["FinishPosition"] <= 3, 0).astype(int)

	return weekend


def build_dataset(start_year: int, end_year: int) -> pd.DataFrame:
	all_rows: list[pd.DataFrame] = []

	for year in range(start_year, end_year + 1):
		schedule = fastf1.get_event_schedule(year)
		rounds = sorted(schedule["RoundNumber"].dropna().astype(int).unique())

		for round_number in rounds:
			try:
				print(f"Loading {year} round {round_number}...")
				all_rows.append(build_race_weekend_dataset(year, round_number))
			except Exception as error:
				print(f"Skipped {year} round {round_number}: {error}")

	if not all_rows:
		raise RuntimeError("No race data was collected.")

	dataset = pd.concat(all_rows, ignore_index=True)
	return dataset


def main() -> None:
	ensure_cache()
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	batch_dataset = build_dataset(START_YEAR, END_YEAR)
	combined_dataset = combine_with_existing_dataset(batch_dataset, DATASET_PATH)
	final_dataset = add_rolling_features(combined_dataset.copy())
	final_dataset.to_csv(DATASET_PATH, index=False)

	print("\nProcessed batch:", f"{START_YEAR}-{END_YEAR}")
	print("Dataset saved to:", DATASET_PATH)
	print("Batch rows:", len(batch_dataset))
	print("Total rows:", len(final_dataset))
	print("Columns:", list(final_dataset.columns))
	print("\nSample:")
	print(final_dataset.head(10).to_string(index=False))


if __name__ == "__main__":
	main()

