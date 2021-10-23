import argparse
from typing import Union

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory


def RunModule(input_dataset: str, 
              column_name: str, 
              lag_columns: int, 
              lag_by: int, 
              output_dataset: str,
              average: bool = False, 
              drop_nulls: bool=True):
    
    data_folder = load_data_frame_from_directory(input_dataset)
    
    if ',' in column_name:
        print('[DEBUG] Multiple columns has been indicated. Identifying them...')
        column_names = [col.strip() for col in column_name.split(',')]
    else:
        column_names = [column_name]

    # Check if column is available
    if data_folder.schema:
        for column in column_names:
            _ = data_folder.get_column_index(column)
    else:
        print('[DEBUG] Column checking ignored as schema not available.')

    if lag_by == 0:
        raise ValueError("Parameter lag_by is incorrect. It has to be different of zero")

    data = data_folder.data
    for column in column_names:
        if average:
            print(f'[DEBUG] Creating {column} with averaged lag')
            data[f'{column}_lag_avg{lag_columns}'] = data[column].rolling(lag_columns*lag_by, center=False).mean()
        else:
            for lag in range(1, lag_columns+1):
                print(f'[DEBUG] Creating {column} with lag {lag}')
                data[f'{column}_lag{lag}'] = data[column].shift(lag*lag_by)

    if drop_nulls:
        print(f'[DEBUG] Dropping nulls')
        data.dropna(inplace=True)

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--column-name", dest="column_name", type=str, required=True)
    parser.add_argument("--lag-columns", dest="lag_columns", type=int, required=True)
    parser.add_argument("--lag-by", dest="lag_by", type=int, required=False, default=1)
    parser.add_argument("--average", dest="average", type=bool, required=False, default=False)
    parser.add_argument("--drop-nulls", dest="drop_nulls", type=bool, required=False, default=True)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str)
    args = parser.parse_args()

    RunModule(**vars(args))
