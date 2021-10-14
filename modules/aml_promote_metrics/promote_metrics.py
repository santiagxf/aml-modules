import argparse
import pandas as pd

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.core import Run

COMPARE_BIGGER_BETTER = 'Bigger is better'
COMPARE_SMALLER_BETTER = 'Smaller is better'
PROMOTE_ALL_MODELS = 'All models metrics'
PROMOTE_BEST_MODEL = 'Best model metrics'

def RunModule(evaluation_results: str, promote_method: str, compare_by: str, compare_by_logic: str, models_name: str, promoted_metrics: str):
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
    if promote_method == PROMOTE_BEST_MODEL:
        if compare_by_logic == COMPARE_BIGGER_BETTER:
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
        print(f"[DEBUG] Skipping saving since the filtered data is not a valid Pandas.DataFrame. Type is {str(type(results))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--evaluation-results", dest="evaluation_results", required=True, type=str, help="Evaluation results")
    parser.add_argument("--promote-method", dest="promote_method", type=str, choices= [PROMOTE_BEST_MODEL, PROMOTE_ALL_MODELS], required=False, default=PROMOTE_BEST_MODEL)
    parser.add_argument("--compare-by", dest="compare_by", type=str, help="Name of the metrics to compared against", required=False)
    parser.add_argument("--compare-by-logic", dest="compare_by_logic", type=str, choices=[COMPARE_BIGGER_BETTER, COMPARE_SMALLER_BETTER], required=False, default=COMPARE_BIGGER_BETTER)
    parser.add_argument("--models-name", dest="models_name", type=str, required=False, default=None)
    parser.add_argument("--promoted-metrics", dest="promoted_metrics", type=str, help="Promotion results")
    args = parser.parse_args()

    RunModule(**vars(args))