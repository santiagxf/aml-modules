import argparse
from numpy.core.fromnumeric import partition
import pandas as pd

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.core import Run

COMPARE_BIGGER_BETTER = 'Bigger is better'
COMPARE_SMALLER_BETTER = 'Smaller is better'
PROMOTE_ALL_MODELS = 'All models metrics'
PROMOTE_BEST_MODEL = 'Best model metrics'

parser = argparse.ArgumentParser("promote-metrics")
parser.add_argument("--evaluation-results", dest="evaluation_results", required=True, type=str, help="Evaluation results")
parser.add_argument("--promote-method", dest="promote_method", type=str, choices= [PROMOTE_BEST_MODEL, PROMOTE_ALL_MODELS], required=False, default=PROMOTE_BEST_MODEL)
parser.add_argument("--compare-by", dest="compare_by", type=str, help="Name of the metrics to compared against", required=False, default='Accuracy' )
parser.add_argument("--compare-by-logic", dest="compare_by_logic", type=str, choices=[COMPARE_BIGGER_BETTER, COMPARE_SMALLER_BETTER], required=False, default=COMPARE_BIGGER_BETTER)
parser.add_argument("--models-name", dest="models_name", type=str, required=False, default=None)
parser.add_argument("--promoted-metrics", dest="promoted_metrics", type=str, help="Promotion results")
args = parser.parse_args()

evaluation_results = args.evaluation_results
promote_method = args.promote_method
compare_by = args.compare_by
compare_by_logic = args.compare_by_logic
models_name = args.models_name
promoted_metrics = args.promoted_metrics

# Load the metrics as a Pandas dataframe
results = load_data_frame_from_directory(evaluation_results).data

# Get the parent run
parent_run = Run.get_context().parent

# Check if the metric exists in the available metrics
if compare_by not in results.columns:
    print('Failed to find', compare_by, 'in available metrics. Using the first one')
    compare_by = results.columns[0]

if models_name:
    models_name = models_name.split(',')
else:
    # Generate models names like 'model A', 'model B' on the fly
    models_name = [f"model {chr(65 + index)}" for index in range(0,5)]

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
    print("Skipping saving since the filtered data is not a valid Pandas.DataFrame")
