import argparse
import numpy as np
from typing import Union

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName


def RunModule(input_dataset: str, column_name: str, output_dataset: str):
    data_folder = load_data_frame_from_directory(input_dataset)
    
    # Check if column is available
    if data_folder.schema:
        column_type = data_folder.get_column_type(column_name)
        if column_type != ColumnTypeName.DATETIME:
            raise TypeError("The type of the column is not datetime")
    else:
        print('[DEBUG] Column checking ignored as schema not available.')

    data = data_folder.data
    data[f'{column_name}_year'] = data[column_name].dt.year
    data[f'{column_name}_month'] = data[column_name].dt.month
    data[f'{column_name}_day'] = data[column_name].dt.day
    data[f'{column_name}_quarter'] = data[column_name].dt.quarter
    data[f'{column_name}_semester'] = np.where(data[f'{column_name}_quarter'].isin([1,2]),1,2)
    data[f'{column_name}_dayofweek'] = data[column_name].dt.day_name()
    data[f'{column_name}_weekend'] = np.where(data[f'{column_name}_dayofweek'].isin(['Sunday','Saturday']),1,0)
    data[f'{column_name}_dayofyear'] = data[column_name].dt.dayofyear
    data[f'{column_name}_weekofyear'] = data[column_name].dt.weekofyear

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--column-name", dest="column_name", type=str, required=True)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str)
    args = parser.parse_args()

    RunModule(**vars(args))
