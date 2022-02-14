import pandas as pd
import scipy.stats as stats

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName

def RunModule(dataset: str, column_name: str, groups_column_name: str, evaluation_results: str):
    data_folder = load_data_frame_from_directory(dataset)
    
    # Check if column is available
    if data_folder.schema:
        column_type = data_folder.get_column_type(column_name)
        if column_type != ColumnTypeName.NUMERIC:
            raise ValueError(f"Column {column_name} has an incorrect type. Expecting {ColumnTypeName.NUMERIC} but got {column_type}")
        
        # Just check if exists
        _ = data_folder.get_column_type(groups_column_name)
    else:
        print('[DEBUG] Column checking ignored as schema not available.')


    data = data_folder.data

    groups = data[[groups_column_name, column_name]].groupby([groups_column_name])
    samples = [groups.get_group(group_key)[column_name] for group_key in groups.groups.keys()]
    results = stats.f_oneway(*samples)
    evaluation = pd.DataFrame([{ 'statistic': results.statistic, 'pvalue': results.pvalue }])

    save_data_frame_to_directory(evaluation_results, evaluation)
