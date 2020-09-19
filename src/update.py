#!/usr/bin/env python

import datetime
import json
import os
import traceback
import warnings
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


def read_data(schema: Dict[str, str], url: str) -> DataFrame:
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
    assert all(
        col in data.columns for col in list(column_adapter.values())[:6]
    ), f"Not all the expected columns we found for {url}"

    # Get rid of data without a district ID
    data["district_id"] = data["district_id"].fillna(UNKNOWN_DISTRICT_ID).apply(parse_district_id)
    data = data[data["district_id"].notna()]

    data = convert_dtype(schema, data)
    return data.dropna(subset=["date", "district_id"])


def data_source_iterator(config: Dict[str, Any]) -> Iterable[DataFrame]:
    """Load all data tables defined by the provided config file."""
    for source in config["sources"]:
        try:
            yield read_data(config["schema"], source["url"])
            print(f"Data successfully downloaded for {source['state']}: {source['url']}")
        except:
            print(f"Failed to process data for {source['state']}: {source['url']}")
            traceback.print_exc()


def read_metadata(config: Dict[str, Any]) -> DataFrame:
    """Read the metadata file defined in the provided config file."""
    column_adapter = {
        "STATECODE": "state",
        "GEOID": "district_id",
        "NAME": "district_name",
        "INTPTLAT": "latitude",
        "INTPTLON": "longitude",
    }
    geo = read_csv(config["geo"], dtype=str)
    geo = table_rename(geo, column_adapter)
    geo.latitude = geo.latitude.str.replace("+", "")
    geo["district_id"] = geo["district_id"].apply(parse_district_id)

    nces = read_csv(config["nces"], dtype=str)
    nces["district_id"] = nces["district_id"].apply(parse_district_id)
    nces = nces[
        [
            "district_id",
            "district_name",
            "state",
            "county_name",
            "student_count",
            "teacher_count",
            "school_count",
        ]
    ]

    # Put all sources of metadata into a single table
    metadata = geo.merge(nces, how="outer", on=["district_id"], suffixes=(".1", ".2"))

    # Use the district name / state from whichever table has it
    for shared_column in ("district_name", "state"):
        column_1 = metadata[f"{shared_column}.1"]
        column_2 = metadata[f"{shared_column}.2"]
        metadata[shared_column] = column_1.fillna(column_2)
        metadata.drop(columns=[f"{shared_column}.1", f"{shared_column}.2"], inplace=True)

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
    data = metadata.merge(cases, how="outer", on=["district_id"])

    # Spit out errors for cases without matching metadata
    for _, record in data[data["district_name"].isna()].iterrows():
        warnings.warn(f"Record without metadata: {record.to_dict()}")

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
