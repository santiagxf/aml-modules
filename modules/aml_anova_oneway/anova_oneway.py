import argparse
from typing import Union

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName

def RunModule(input_dataset: str, groups_column_name: str, control_group_value: str, evaluation_results: str):
    data_folder = load_data_frame_from_directory(input_dataset)
    
    # Check if column is available
    if data_folder.schema:
        groups_column_type = data_folder.get_column_type(groups_column_name)
        if groups_column_type != ColumnTypeName.STRING:
            raise ValueError(f"Column {groups_column_name} has an incorrect type. Expecting {ColumnTypeName.STRING} but got {groups_column_type}")
    else:
        print('[DEBUG] Column checking ignored as schema not available.')


    data = data_folder.data
    save_data_frame_to_directory(evaluation_results, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--groups-column-name", dest="groups_column_name", type=str, required=True)
    parser.add_argument("--control-group-value", dest="control_group_value", type=str, required=True)
    parser.add_argument("--evaluation-results", dest="evaluation_results", type=str)
    args = parser.parse_args()

    RunModule(**vars(args))