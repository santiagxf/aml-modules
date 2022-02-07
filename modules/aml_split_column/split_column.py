from typing import Union
from jobtools.runner import TaskRunner
from jobtools.arguments import StringEnum

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

class SplitMode(StringEnum):
    COLUMNS = 'Into columns'
    ROWS = 'Into rows'
    ARRAY = 'As array-like'

def RunModule(dataset: str, output_dataset: str, column_name: str, split_mode: SplitMode = SplitMode.ARRAY,
              split_by: str = ' ', new_columns_name:Union[str, None]=None):
    data_folder = load_data_frame_from_directory(dataset)
    
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
        if split_mode == SplitMode.COLUMNS:
            # Calculate number of columns to use
            if new_columns_name:
                columns_name = [col.strip() for col in new_columns_name.split(',')]
            else:
                # Hint: this may be too much
                max_count = data[column_name].str.split(split_by).len().max()
                columns_name = [f'{column_name}_{index}' for index in range(0, max_count)]
            
            data[columns_name] = data[column_name].str.split(split_by, expand=True)

        elif split_mode == SplitMode.ROWS or split_mode == SplitMode.ARRAY:
            data[column_name] = data[column_name].apply(lambda x: x.split(split_by))
        
            if split_mode == SplitMode.ROWS:
                data = data.explode(column_name)
        else:
            raise ValueError(f'{split_mode} is not a valid split mode')
    else:
        print('[DEBUG] Nothing to do as dataframe is empty.')

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    tr = TaskRunner()
    tr.run(RunModule)
