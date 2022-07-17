import mlflow
import pandas as pd
import os
import sys
import subprocess
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.uri import append_to_uri_path
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.file_utils import TempDir
from jobtools.arguments import StringEnum
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

class OutputMode(StringEnum):
    APPEND = 'Append score columns to output'
    RESULTS_ONLY = 'Score columns only'

def run_module(dataset: str, output_dataset: str, model: str, mode: OutputMode):
    data_folder = load_data_frame_from_directory(dataset)
    df: pd.DataFrame = data_folder.data

    with TempDir() as tmp:
        print(f"[DEBUG] Getting artifacts")
        underlying_model_uri = ModelsArtifactRepository.get_underlying_uri(model)
        local_path = _download_artifact_from_uri(
                append_to_uri_path(underlying_model_uri, MLMODEL_FILE_NAME), output_path=tmp.path()
            )

        req_file_path = os.path.join(local_path, "requirements.txt")
        print(f"[DEBUG] Installing packages from {req_file_path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file_path])
        
        print(f"[DEBUG] Loading model from {local_path}")
        mlflow_model = Model.load(local_path)
        
        output_data_path = os.path.join(tmp.path(), "outputs.json")
        input_data_path = os.path.join(tmp.path(), "inputs.json")
        df.to_json(input_data_path)

        print(f"[DEBUG] Running model from backend")
        backend = mlflow.pyfunc.backend.PyFuncBackend(mlflow_model.flavors[mlflow.pyfunc.FLAVOR_NAME])
        backend.predict(model_uri=model,
                        input_path=input_data_path,
                        output_path=output_data_path,
                        content_type="json",
                        json_format="split")

    print(f"[DEBUG] Returning predicts from JSON")
    predictions = pd.read_json(output_data_path)

    if mode == OutputMode.APPEND:
        save_data_frame_to_directory(output_dataset, pd.concat([df, predictions], axis=1))
    else:
        save_data_frame_to_directory(output_dataset, predictions)
