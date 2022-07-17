import pandas as pd
from pathlib import Path
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import PickleTransformationDirectory


def run_module(dataset: str, transformation: str, output_dataset: str):
    data_folder = load_data_frame_from_directory(dataset)
    data = data_folder.data
    tranformation = PickleTransformationDirectory(transformation).load(Path(transformation))

    transformed_data = tranformation.transform.transform(data)

    if transformed_data.shape[-1] == data.shape[-1]:
        column_names = data.columns
    else:
        column_names = [f"col{index}" for index in range(0, transformed_data.shape[-1])]

    df = pd.DataFrame(data = transformed_data, columns = column_names)
    save_data_frame_to_directory(output_dataset, df)

