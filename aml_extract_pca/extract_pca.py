import argparse
import pandas as pd
from numpy.core.fromnumeric import partition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory

PCA_SOLVERS = ['auto', 'full', 'arpack', 'randomized']

def RunModule(input_dataset: str, number_of_dimensions: int, normalize: bool, solver: str, 
              output_dataset: str, output_model: str, output_eigenvectors: str):
    data_folder = load_data_frame_from_directory(input_dataset)

    if data_folder.schema:
        if any([col['type'] != 'Numeric' for col in data_folder.schema['columnAttributes']]):
            raise TypeError('Dataset contains non-numeric columns. Cannot apply PCA to non-numeric types.')
    else:
        print('Column type checking ignored as schema not available.')
    
    if number_of_dimensions <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    if solver not in PCA_SOLVERS:
        raise ValueError(f'Solver {solver} is not a valid PCA solver. Possible values are {PCA_SOLVERS}')

    data = data_folder.data
    tranformations = []

    if normalize:
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        tranformations.append(('normalize', scaler))

    pca = PCA(n_components=number_of_dimensions, svd_solver=solver, whiten=True).fit(data)
    transformed_data = pca.transform(data)
    tranformations.append(('pca', pca))

    tranformations_pipe = Pipeline(tranformations)

    components_name = [f"col{index}" for index in range(0, number_of_dimensions)]
    df = pd.DataFrame(data = transformed_data, columns = components_name)

    save_data_frame_to_directory(output_dataset, df)
    save_pickle_transform_to_directory(output_model, tranformations_pipe)

    if output_eigenvectors:
        eigenvectors = pd.DataFrame(data = pca.components_.T, columns = components_name)
        save_data_frame_to_directory(output_eigenvectors, eigenvectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract-pca")
    parser.add_argument("--dataset", dest="input_dataset", required=True, type=str, help="Input dataset")
    parser.add_argument("--number-of-dimensions", dest="number_of_dimensions", type=int, help="Number of dimensions to reduce to", required=True)
    parser.add_argument("--normalize", dest="normalize", type=bool, help="Whether or not to normalize data to zero mean", required=True, default=True)
    parser.add_argument("--solver", dest="solver", type=str, choices=PCA_SOLVERS, help="Solver to calculate PCA")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Transformed dataset")
    parser.add_argument("--output-model", dest="output_model", type=str, help="Trained model")
    parser.add_argument("--output-eigenvectors", dest="output_eigenvectors", required=False, type=str, help="Eigenvectors values")
    args = parser.parse_args()

    RunModule(**vars(args))
