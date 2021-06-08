import argparse
from numpy.core.fromnumeric import partition
from sklearn.decomposition import PCA

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory


def RunModule(input_dataset: str, number_of_dimensions: int, normalize: bool, output_dataset: str, output_model: str):
    data = load_data_frame_from_directory(input_dataset).data

    pca_transform = PCA(n_components=number_of_dimensions, svd_solver='randomized', whiten=True).fit(data)
    transformed_data = pca_transform.transform(data)

    save_data_frame_to_directory(output_dataset, transformed_data)
    save_pickle_transform_to_directory(output_model, pca_transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract-pca")
    parser.add_argument("--dataset", dest="input_dataset", required=True, type=str, help="Input dataset")
    parser.add_argument("--number-of-dimensions", dest="number_of_dimensions", type=int, help="Number of dimensions to reduce to", required=True)
    parser.add_argument("--normalize", dest="normalize", type=bool, help="Whether or not to normalize data to zero mean", required=True, default=True)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Transformed dataset")
    parser.add_argument("--output-model", dest="output_model", type=str, help="Trained model")
    args = parser.parse_args()

    RunModule(**vars(args))
