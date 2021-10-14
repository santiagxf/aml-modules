import argparse
import pandas as pd
from typing import Union, List

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName


BOOL_AGG = {
    'True for any': lambda x: any(x),
    'True for all': lambda x: all(x),
    'False for any': lambda x: not all(x),
    'False for all': lambda x: not any(x)
}

STRING_AGG = {
    'Concatenate (comma separated)': lambda x: ','.join(x),
    'First': lambda x: x.iloc[0],
    'Last': lambda x: x.iloc[-1],
    'Mode': lambda x:x.value_counts().index[0],
    'Count': lambda x:x.count()
}

NUMERIC_AGG = {
    'Maximum': 'max',
    'Minimum': 'min',
    'Mean': 'mean',
    'Mode': lambda x:x.value_counts().index[0],
    'Sum': 'sum',
    'Variance': 'var',
    'Concatenate (comma separated)': lambda x: ','.join(x)
}

DATETIME_AGG = {
    'Maximum': 'max',
    'Minimum': 'min',
    'Number of days': lambda x: max(x) - min(x)
}

AGG_SHORT = {
    'True for any': 'any',
    'True for all': 'all',
    'False for any': 'notall',
    'False for all': 'none',
    'Concatenate (comma separated)': 'concat',
    'First': 'first',
    'Last': 'last',
    'Mode': 'mode',
    'Count': 'count',
    'Maximum': 'max',
    'Minimum': 'min',
    'Mean': 'mean',
    'Sum': 'sum',
    'Variance': 'var',
    'Number of days': 'days'
}
    

def get_columns_by_type(data_folder, dtype: str, exclude: Union[List[str], None]=None):
    return [column['name'] for column in data_folder.schema['columnAttributes'] if column['type'] == dtype and column['name'] not in exclude]

def RunModule(input_dataset: str, group_by_columns: str, aggregate_booleans_by: str, 
              aggregate_numbers_by: str, aggregate_strings_by: str, aggregate_datetimes_by: str,
              output_dataset: str):

    data_folder = load_data_frame_from_directory(input_dataset)
    group_by = [x.strip() for x in group_by_columns.split(',')]
    
    if data_folder.schema:
        numeric_columns = get_columns_by_type(data_folder, dtype=ColumnTypeName.NUMERIC, exclude=group_by)
        boolean_columns = get_columns_by_type(data_folder, dtype=ColumnTypeName.BINARY, exclude=group_by)
        string_columns = get_columns_by_type(data_folder, dtype=ColumnTypeName.STRING, exclude=group_by)
        datetime_columns = get_columns_by_type(data_folder, dtype=ColumnTypeName.DATETIME, exclude=group_by)
    else:
        raise TypeError("Input data doesn't have an schema available")

    # Build aggregations
    aggregations = {}
    for col in numeric_columns:
        agg_name = f'{col}_{AGG_SHORT[aggregate_numbers_by]}'
        print(f"[DEBUG] Building numeric agg with name {agg_name} for {col} and function {aggregate_numbers_by}")
        aggregations[agg_name] = pd.NamedAgg(column=col, aggfunc=NUMERIC_AGG[aggregate_numbers_by])

    for col in boolean_columns:
        agg_name = f'{col}_{AGG_SHORT[aggregate_booleans_by]}'
        print(f"[DEBUG] Building boolean agg with name {agg_name} for {col} and function {aggregate_booleans_by}")
        aggregations[agg_name] = pd.NamedAgg(column=col, aggfunc=BOOL_AGG[aggregate_booleans_by])

    for col in string_columns:
        agg_name = f'{col}_{AGG_SHORT[aggregate_strings_by]}'
        print(f"[DEBUG] Building string agg with name {agg_name} for {col} and function {aggregate_strings_by}")
        aggregations[agg_name] = pd.NamedAgg(column=col, aggfunc=STRING_AGG[aggregate_strings_by])

    for col in datetime_columns:
        agg_name = f'{col}_{AGG_SHORT[aggregate_datetimes_by]}'
        print(f"[DEBUG] Building datetime agg with name {agg_name} for {col} and function {aggregate_datetimes_by}")
        aggregations[agg_name] = pd.NamedAgg(column=col, aggfunc=DATETIME_AGG[aggregate_datetimes_by])

    # Apply aggregations
    data = data_folder.data

    if data.isnull().values.any():
        raise ValueError('Input data cotains nulls. This module cannot work with null values. Please clean it before using it')

    grouped_data = data.groupby(group_by).agg(**aggregations)
    save_data_frame_to_directory(output_dataset, grouped_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--group-by-columns", dest="group_by_columns", type=str, required=True)
    parser.add_argument("--aggregate-booleans-by", dest="aggregate_booleans_by", type=str, choices=BOOL_AGG.keys())
    parser.add_argument("--aggregate-numbers-by", dest="aggregate_numbers_by", type=str, choices=NUMERIC_AGG.keys())
    parser.add_argument("--aggregate-strings-by", dest="aggregate_strings_by", type=str, choices=STRING_AGG.keys())
    parser.add_argument("--aggregate-dates-by", dest="aggregate_datetimes_by", type=str, choices=DATETIME_AGG.keys())
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, required=True)
    args = parser.parse_args()

    RunModule(**vars(args))
