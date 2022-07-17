import pandas as pd
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.core import Run
from jobtools.arguments import StringEnum

class PromoteStrategy(StringEnum):
    ALL_MODELS = 'All models metrics'
    BEST_MODEL = 'Best model metrics'

class CompareStrategy(StringEnum):
    BIGGER_BETTER = 'Bigger is better'
    SMALLER_BETTER = 'Smaller is better'


def run_module(evaluation_results: str, promoted_metrics: str, compare_by: str,
               promote_method: PromoteStrategy = PromoteStrategy.BEST_MODEL,
               compare_by_logic: CompareStrategy = CompareStrategy.BIGGER_BETTER, 
               models_name: str = None):
    # Load the metrics as a Pandas dataframe
    results = load_data_frame_from_directory(evaluation_results).data
    models_count = results.shape[0]

    # Get the parent run
    parent_run = Run.get_context().parent

    # Check if the metric exists in the available metrics
    if compare_by not in results.columns:
        # Try with underscores. AML seems not to be very consistent on this
        compare_by = compare_by.replace(' ', '_')
        if compare_by not in results.columns:
            print(f'[WARNING] Failed to find {compare_by} in available metrics. Using the first one')
            compare_by = results.columns[0]

    if models_name:
        models_name = [name.strip() for name in models_name.split(',')]
        for missing_index in range(len(models_name), models_count):
            models_name.append(f'unlabeled model {missing_index}')
    else:
        # Generate models names like 'model A', 'model B' on the fly
        models_name = [f"model {chr(65 + index)}" for index in range(0, models_count)]

    # Filter the rows based on the metric you are looking for to compare and the
    # logic to compare
    if promote_method == PromoteStrategy.BEST_MODEL:
        if compare_by_logic == CompareStrategy.BIGGER_BETTER:
            results = results.loc[results[compare_by].argmax()]
        else:
            results = results.loc[results[compare_by].argmin()]

    metrics = results.to_dict()
    for metric, value in metrics.items():
        if type(value) is dict:
            for model, point in value.items():
                if len(value) > 1:
                    parent_run.log(name=f"{metric} ({models_name[model]})", value=point)
                else:
                    parent_run.log(name=f"{metric}", value=point)
        else:
            parent_run.log(name=f"{metric}", value=value)

    # Save the promoted metrics
    if (type(results) is pd.DataFrame):
        save_data_frame_to_directory(promoted_metrics, data=results)
    else:
        save_data_frame_to_directory(promoted_metrics, data=results.to_frame().T)
