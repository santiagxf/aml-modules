import os
from random import choice
import sys
import argparse
import pandas as pd
import numpy as np

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.core import Run

COMPARE_BIGGER_BETTER = 'Bigger is better'
COMPARE_SMALLER_BETTER = 'Smaller is better'
PROMOTE_ALL_MODELS = 'All models'
PROMOTE_BEST_MODEL = 'Best model'

parser = argparse.ArgumentParser("promote-metrics")
parser.add_argument("--evaluation-results", dest="evaluation_results", required=True, type=str, help="Evaluation results")
parser.add_argument("--promote-method", dest="promote_method", type=str, choices= [PROMOTE_BEST_MODEL, PROMOTE_ALL_MODELS], required=False, default=PROMOTE_BEST_MODEL)
parser.add_argument("--compare-by", dest="compare_by", type=str, help="Name of the metrics to compared against", required=False, default='Accuracy' )
parser.add_argument("--compare-by-logic", dest="compare_by_logic", type=str, choices=[COMPARE_BIGGER_BETTER, COMPARE_SMALLER_BETTER], required=False, default=COMPARE_BIGGER_BETTER)
parser.add_argument("--promoted-metrics", dest="promoted_metrics", type=str, help="Promotion results")
args = parser.parse_args()

evaluation_results = args.evaluation_results
promote_method = args.promote_method
compare_by = args.compare_by
compare_by_logic = args.compare_by_logic
promoted_metrics = args.promoted_metrics

# Load the metrics as a Pandas dataframe
results = load_data_frame_from_directory(evaluation_results).data

# Get the parent run
parent_run = Run.get_context().parent

# Check if the metric exists in the available metrics
if compare_by not in results.columns:
    print('Failed to find', compare_by, 'in available metrics. Using the first one')
    compare_by = results.columns[0]

# Filter the rows based on the metric you are looking for to compare and the
# logic to compare
if promote_method == PROMOTE_BEST_MODEL:
    if compare_by_logic == COMPARE_BIGGER_BETTER:
        results = results.iloc[results[compare_by].argmax()]
    else:
        results = results.iloc[results[compare_by].argmin()]

metrics = results.to_dict()
for metric, values in metrics.items():
    if type(values) is dict:
        for model, point in values.items():
            if promote_method == PROMOTE_BEST_MODEL:
                parent_run.log(name=f"{metric}", value=point)
            else:
                parent_run.log(name=f"{metric} ({model})", value=point)
    else:
        parent_run.log(name=f"{metric}", value=point)

# Save the promoted metrics
save_data_frame_to_directory(promoted_metrics, data=results)
