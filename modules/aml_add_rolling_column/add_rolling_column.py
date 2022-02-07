import pandas as pd
from jobtools.runner import TaskRunner
from jobtools.arguments import StringEnum
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.data_frame_visualizer import ColumnTypeName

class RollingDirection(StringEnum):
    FORWARD='Forward'
    BACKWARD='Backward'

AGG_FUNCTIONS = {
    ColumnTypeName.NUMERIC: {
        'Maximum': 'max',
        'Minimum': 'min',
        'Mean': 'mean',
        'Mode': lambda x:x.value_counts().index[0],
        'Sum': 'sum',
        'Variance': 'var',
        'Concatenate (comma separated)': lambda x: ','.join(x)
    },
    ColumnTypeName.BINARY: {
        'True for any': lambda x: any(x),
        'True for all': lambda x: all(x),
        'False for any': lambda x: not all(x),
        'False for all': lambda x: not any(x)
    },
    ColumnTypeName.STRING: {
        'Concatenate (comma separated)': lambda x: ','.join(x),
        'First': lambda x: x.iloc[0],
        'Last': lambda x: x.iloc[-1],
        'Mode': lambda x:x.value_counts().index[0],
        'Count': lambda x:x.count()
    },
    ColumnTypeName.DATETIME: {
        'Maximum': 'max',
        'Minimum': 'min',
        'Number of days': lambda x: max(x) - min(x)
    }
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

def RunModule(dataset: str,
              output_dataset: str,
              column_name: str, 
              window_size: int, 
              aggregate_by: str,
              direction: RollingDirection = RollingDirection.BACKWARD, 
              drop_nulls: bool = True):
    
    data_folder = load_data_frame_from_directory(dataset)
    
    if ',' in column_name:
        print('[DEBUG] Multiple columns has been indicated. Identifying them...')
        column_names = [col.strip() for col in column_name.split(',')]
    else:
        column_names = [column_name]

    # Check if column is available
    if data_folder.schema:
        column_types = [data_folder.get_column_type(column) for column in column_names]
    else:
        raise ValueError('Schema is not available for the given dataframe. Module can\'t be used')

    if window_size == 0:
        raise ValueError("Parameter window_size is incorrect. It has to be different of zero")

    data = data_folder.data
    for column, type_ in zip(column_names, column_types):
        print(f'[DEBUG] Creating rolling window for {column}')
        rolling_name = f'{column}_{AGG_SHORT[aggregate_by]}_{window_size}'
        agg_func = AGG_FUNCTIONS[type_][aggregate_by]
        
        if direction==RollingDirection.FORWARD:    
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        else:
            indexer = window_size

        data[rolling_name] = data[column].rolling(indexer, center=False).agg(agg_func)

    if drop_nulls:
        print(f'[DEBUG] Dropping nulls')
        data.dropna(inplace=True)

    save_data_frame_to_directory(output_dataset, data)

if __name__ == "__main__":
    tr = TaskRunner()
    tr.run(RunModule)
