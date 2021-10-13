import argparse
from typing import Union

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

SPLIT_MODE_COLUMNS = 'Into columns'
SPLIT_MODE_ROWS = 'Into rows'
SPLIT_MODE_ARRAY = 'As array-like'
SPLIT_MODE = [SPLIT_MODE_COLUMNS, SPLIT_MODE_ROWS, SPLIT_MODE_ARRAY]

def RunModule(input_dataset: str, column_name: str, split_mode: str, split_by: str, output_dataset: str, new_columns_name:Union[str, None]=None):
    data_folder = load_data_frame_from_directory(input_dataset)
    
    # Check if column is available
    if data_folder.schema:
        column_type = data_folder.get_column_type(column_name)
        if column_type != 'String':
            raise TypeError(f'Column {column_name} is not of type String, but {column_type}. Type is not compatible.')
    else:
        print('[DEBUG] Column type checking ignored as schema not available.')

    if not split_by:
        print('[DEBUG] Setting split_by to a white space.')
        split_by = ' '

    data = data_folder.data
    if len(data.index) > 0:
        if split_mode == SPLIT_MODE_COLUMNS:
            # Calculate number of columns to use
            if new_columns_name:
                columns_name = [col.strip() for col in new_columns_name.split(',')]
            else:
                columns_name = [f'{column_name}_{index}' for index in range(0, len(data[column_name].iloc[0].split(split_by)))]
            
            data[columns_name] = data[column_name].str.split(split_by, expand=True)

        elif split_mode == SPLIT_MODE_ROWS or split_mode == SPLIT_MODE_ARRAY:
            data[column_name] = data[column_name].apply(lambda x: x.split(split_by))
        
            if split_mode == SPLIT_MODE_ROWS:
                data = data.explode(column_name)
        else:
            raise ValueError(f'{split_mode} is not a valid split mode')
    else:
        print('[DEBUG] Nothing to do as dataframe is empty.')

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", type=str, required=True)
    parser.add_argument("--column-name", dest="column_name", type=str, required=True)
    parser.add_argument("--split-by", dest="split_by", type=str, required=False, default=' ')
    parser.add_argument("--split-mode", dest="split_mode", type=str, choices=SPLIT_MODE, default=SPLIT_MODE_ARRAY)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str)
    parser.add_argument("--new-columns-name", dest="new_columns_name", type=str, Required=False)
    args = parser.parse_args()

    RunModule(**vars(args))
