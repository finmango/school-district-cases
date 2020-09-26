#!/usr/bin/env python

import datetime
import json
import logging
import os
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import yaml
from pandas import DataFrame, Int32Dtype, concat, isna, read_csv

# ROOT directory
ROOT = Path(os.path.dirname(__file__))

# Used to fill unknown districts
UNKNOWN_DISTRICT_ID = "9999999"


def table_rename(data: DataFrame, column_adapter: Dict[str, str]) -> DataFrame:
    """Rename all columns of a dataframe and drop the columns not in the adapter."""
    data = data.rename(columns=column_adapter)
    data = data.drop(columns=[col for col in data.columns if col not in column_adapter.values()])
    return data


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file given its path."""
    with open(config_path, "r") as fh:
        config_yaml = yaml.safe_load(fh)
    return config_yaml


def nullable_method_call(func: Callable, *args, print_exc: bool = True, **kwargs) -> Any:
    """Return the output of calling the provided `func`, default to `None` in case of failure."""
    try:
        return func(*args, **kwargs)
    except:
        if print_exc:
            traceback.print_exc()
        return None


def convert_dtype(schema: Dict[str, str], data: DataFrame) -> DataFrame:
    """Convert all columns in `data` to the appropriate dtype according to `schema`."""
    df = DataFrame(index=data.index)
    for column_name, dtype in schema.items():
        if column_name not in data.columns:
            continue
        elif dtype == "str":
            df[column_name] = data[column_name]
        elif dtype == "float":
            apply_func = partial(nullable_method_call, float, print_exc=False)
            df[column_name] = data[column_name].apply(apply_func).astype(float)
        elif dtype == "int":
            apply_func = partial(nullable_method_call, int, print_exc=False)
            df[column_name] = data[column_name].apply(apply_func).astype(Int32Dtype())
        else:
            raise TypeError(f"Unknown dtype {dtype}")
    return df


def parse_district_id(district_id: str) -> str:
    """Ensure that `district_id` is a 7-digit string."""
    return f"{int(district_id):07d}"


def read_data(schema: Dict[str, str], state: str, url: str) -> DataFrame:
    """Read all CSV data from `url` into a dataframe and use `schema` to determine dtype."""
    data = read_csv(url, dtype=str, skiprows=1)
    data.columns = [col.strip() for col in data.columns]
    column_adapter = {
        "date": "date",
        "district_id": "district_id",
        "new": "new_student_cases",
        "cumulative": "cumulative_student_cases",
        "new.1": "new_staff_cases",
        "cumulative.1": "cumulative_staff_cases",
        "new.2": "new_unspecified_cases",
        "cumulative.2": "cumulative_unspecified_cases",
        "website": "source",
    }
    data = table_rename(data, column_adapter)

    # Make sure data has at least the core columns
    missing_columns = set(list(column_adapter.values())[:-1]) - set(data.columns)
    assert len(missing_columns) == 0, f"Missing columns {missing_columns} for {state}."

    # Keep only data with date and district ID
    data["district_id"] = data["district_id"].fillna(UNKNOWN_DISTRICT_ID).apply(parse_district_id)
    data = data.dropna(subset=["date", "district_id"])

    # Add a state column to all records
    data["state"] = state

    # Get rid of data without any known cases
    case_columns = list(column_adapter.values())[2:-1]
    data = data.dropna(subset=case_columns, how="all")

    # Warn if the data is empty
    if len(data) == 0:
        logging.warning(f"State {state} has zero records.")

    # Return data with the appropriate type
    return convert_dtype(schema, data)


def data_source_iterator(config: Dict[str, Any]) -> Iterable[DataFrame]:
    """Load all data tables defined by the provided config file."""
    for source in config["sources"]:
        try:
            yield read_data(config["schema"], source["state"], source["url"])
            print(f"Data successfully downloaded for {source['state']}: {source['url']}")
        except:
            log_msg = f"Failed to process data for {source['state']}: {source['url']}"
            logging.error(log_msg, exc_info=True)


def read_metadata(config: Dict[str, Any]) -> DataFrame:
    """Read the metadata file defined in the provided config file."""
    columns = [
        "district_id",
        "state",
        "district_name",
        "longitude",
        "latitude",
        "county_name",
        "street_address",
        "city",
        "zip_code",
        "phone",
        "student_count",
        "teacher_count",
        "school_count",
    ]
    metadata = read_csv(config["districts"], dtype=str, usecols=columns)
    metadata.district_id = metadata.district_id.apply(parse_district_id)

    return metadata


def convert_to_geojson(data: DataFrame) -> Dict[str, Any]:
    """Convert the dataframe into a set GeoJSON records."""
    # Our desired GeoJSON format only has the latest datapoint for each district
    data = data.sort_values("date")
    data["date"] = data["date"].fillna("9999-99-99")
    data = data.groupby(["state", "district_id", "date"]).last().reset_index()
    data.loc[data["date"] == "9999-99-99", "date"] = None

    # Iterate over each row and convert it to GeoJSON type
    records = []
    geo_props = ["latitude", "longitude"]
    for _, row in data.iterrows():
        row_props = {col: None if isna(val) else val for col, val in row.to_dict().items()}

        records.append(
            {
                "type": "Feature",
                "properties": {col: val for col, val in row_props.items() if col not in geo_props},
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["latitude"]), float(row["longitude"])],
                },
            }
        )

    return {"type": "FeatureCollection", "features": records}


def main():
    """Main entrypoint."""
    # Read local configuration file
    config = load_config(ROOT / "config.yaml")

    # Read all the case files and put them into a single table
    cases = concat(data_source_iterator(config))

    # Add metadata information to all records
    metadata = read_metadata(config["metadata"])
    data = cases.merge(metadata, how="left", on=["district_id", "state"])

    # Spit out errors for cases without matching metadata
    for _, record in data[data["district_name"].isna()].iterrows():
        logging.warning(f"Record without metadata: {record.to_dict()}")

    # Sort the data by state and district
    data = data.sort_values(["state", "district_id", "date"])

    # Convert data to GeoJSON format
    geojson = convert_to_geojson(data)

    # Write results to disk
    output_folder = ROOT / ".." / "output"
    output_folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_folder / "latest-school-cases.csv", index=False)
    with open(output_folder / "latest-school-cases.geojson", "w") as fh:
        json.dump(geojson, fh)

    # Write a dated copy to disk
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    data.to_csv(output_folder / f"{date_str}-school-cases.csv", index=False)
    with open(output_folder / f"{date_str}-school-cases.geojson", "w") as fh:
        json.dump(geojson, fh)


if __name__ == "__main__":
    main()
