import argparse
from typing import Union

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory


def RunModule(input_dataset: str, column_name: str, lag_columns: int, lag_by: int, output_dataset: str, drop_na: bool=True):
    data_folder = load_data_frame_from_directory(input_dataset)
    
    # Check if column is available
    if data_folder.schema:
        _ = data_folder.get_column_index(column_name)
    else:
        print('[DEBUG] Column checking ignored as schema not available.')

    if lag_by == 0:
        raise ValueError("Parameter lag_by is incorrect. It has to be different of zero")

    data = data_folder.data
    for lag in range(1, lag_columns+1):
        print(f'[DEBUG] Creating column with lag {lag}')
        data[f'{column_name}_lag{lag}'] = data[column_name].shift(lag*lag_by)

    if drop_na:
        data.dropna(inplace=True)

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--column-name", dest="column_name", type=str, required=True)
    parser.add_argument("--lag-columns", dest="lag_columns", type=int, required=True)
    parser.add_argument("--lag-by", dest="lag_by", type=int, requiered=False, default=1)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str)
    args = parser.parse_args()

    RunModule(**vars(args))
