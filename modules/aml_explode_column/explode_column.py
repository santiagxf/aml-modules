import argparse
import pandas as pd
import numpy as pd
import itertools
from typing import Union
from enum import Enum

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

class ExplodeStrategy(Enum):
    COLUMNS = 'Into columns'
    ROWS = 'Into rows'

def RunModule(input_dataset: str, column_name: str, explode_mode: str, output_dataset: str, new_columns_name:Union[str, None]=None):
    data_folder = load_data_frame_from_directory(input_dataset)
    
    # Check if column is available
    if data_folder.schema:
        column_type = data_folder.get_column_type(column_name)
        if column_type != 'Object':
            raise TypeError(f'Column {column_name} is not of type Object, but {column_type}. Type is not compatible.')
    else:
        print('[DEBUG] Column type checking ignored as schema not available.')

    data = data_folder.data
    if len(data.index) > 0:
        if explode_mode == ExplodeStrategy.COLUMNS:
            # Calculate number of columns to use
            if new_columns_name:
                columns_name = [col.strip() for col in new_columns_name.split(',')]
            else:
                max_count = data[column_name].map(len).max()
                columns_name = [f'{column_name}_{index}' for index in range(0, max_count)]
            
            padded_data = np.array(list(itertools.zip_longest(*data[column_name].tolist(), fillvalue=None))).T
            data[columns_name] = pd.DataFrame(padded_data, index=data.index)

        elif explode_mode == ExplodeStrategy.ROWS:
            data = data.explode(column_name)
        else:
            raise ValueError(f'{explode_mode} is not a valid explode mode')
    else:
        print('[DEBUG] Nothing to do as dataframe is empty.')

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--column-name", dest="column_name", type=str, required=True)
    parser.add_argument("--explode-mode", dest="explode_mode", type=str, choices=list(map(str, ExplodeStrategy)), default=ExplodeStrategy.ROWS)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str)
    parser.add_argument("--new-columns-name", dest="new_columns_name", type=str, required=False)
    args = parser.parse_args()

    RunModule(**vars(args))
