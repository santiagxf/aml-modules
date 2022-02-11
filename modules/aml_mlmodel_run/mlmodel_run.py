import mlflow
import azureml.core as aml
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

def RunModule(dataset: str, output_dataset: str, model: str):
    data_folder = load_data_frame_from_directory(dataset)

    ws: aml.Workspace = aml.Run.get_context().experiment.workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    
    model = mlflow.pyfunc.load_model(model)

    predictions = model.predict(data_folder.data)

    save_data_frame_to_directory(output_dataset, predictions)