import argparse
import pandas as pd
from pathlib import Path
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import PickleTransformationDirectory


def RunModule(input_dataset: str, transformation: str, output_dataset: str):
    data_folder = load_data_frame_from_directory(input_dataset)
    data = data_folder.data
    tranformation = PickleTransformationDirectory(transformation).load(Path(transformation))

    transformed_data = tranformation.transform.transform(data)

    if transformed_data.shape[-1] == data.shape[-1]:
        column_names = data.columns
    else:
        column_names = [f"col{index}" for index in range(0, transformed_data.shape[-1])]

    df = pd.DataFrame(data = transformed_data, columns = column_names)
    save_data_frame_to_directory(output_dataset, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", required=True, type=str, help="Input dataset")
    parser.add_argument("--transformation", dest="transformation", type=str, help="Transformation with Scikit-learn transformation API")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Transformed dataset")
    args = parser.parse_args()

    RunModule(**vars(args))
