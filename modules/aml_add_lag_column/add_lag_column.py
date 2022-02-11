from jobtools.runner import TaskRunner
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

def RunModule(dataset: str, output_dataset: str, column_name: str, lag_columns: int, 
              lag_by: int = 1, average: bool = False, drop_nulls: bool = True):
    """
    Adds one or many lag columns into the data set. The amount of time steps to go back can be
    indicated. This module also allows you to average all the lag columns (average lag). This is
    useful, image a time series with one sample per day, to form columns like "the average value
    of the series on the same day of week in the past three weeks". We can do that by indicating
    `lag_columns=3`, `lag_by=7`, `average=True`. Multiple columns can be generated by indicated
    `comma-separated` column names.

    Parameters
    ----------
    dataset : str
        Path to the dataset directory.
    output_dataset : str
        Path where the transformed dataset should be placed.
    column_name : str
        Name of the column(s) to generate the lag columns from. Comma-separated for multiple columns.
    lag_columns : int
        Number of lag columns to generate for each column indicated before.
    lag_by : int, optional
        Number of time-steps for each lag column, by default 1
    average : bool, optional
        Indicates if the generated lag columns should be all averaged to geneate a new one, by default False
    drop_nulls : bool, optional
        Indicates if rows with `null` values should be dropped from the dataset, by default True

    Raises
    ------
    ValueError
        If an invalid lag value is indicated.
    """
    
    data_folder = load_data_frame_from_directory(dataset)
    
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
    tr = TaskRunner()
    tr.run(RunModule)