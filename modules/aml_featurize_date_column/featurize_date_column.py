import numpy as np
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName


def run_module(dataset: str, output_dataset: str, column_name: str, contains_time: bool = False):
    data_folder = load_data_frame_from_directory(dataset)
    
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

    if contains_time:
        data[f'{column_name}_hour'] = data[column_name].dt.hour
        data[f'{column_name}_minute'] = data[column_name].dt.minute
        data[f'{column_name}_ampm'] = np.where(data[f'{column_name}_hour'] < 12, 'am', 'pm')

    save_data_frame_to_directory(output_dataset, data)

