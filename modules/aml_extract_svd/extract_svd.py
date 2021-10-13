import argparse
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory

SVD_SOLVERS = ['arpack', 'randomized']

def RunModule(input_dataset: str, number_of_dimensions: int, interations: int, solver: str, 
              output_dataset: str, output_model: str, output_singular_values: str, output_components: str):
    data_folder = load_data_frame_from_directory(input_dataset)

    if data_folder.schema:
        if any([col['type'] != 'Numeric' for col in data_folder.schema['columnAttributes']]):
            raise TypeError('Dataset contains non-numeric columns. Cannot apply SVD to non-numeric types.')
    else:
        print('Column type checking ignored as schema not available.')
    
    if number_of_dimensions <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    if solver not in SVD_SOLVERS:
        raise ValueError(f'Solver {solver} is not a valid SVD solver. Possible values are {SVD_SOLVERS}')

    data = data_folder.data
    tranformations = []

    svd = TruncatedSVD(n_components=number_of_dimensions, n_iter=interations, algorithm=solver).fit(data)
    USigma = svd.transform(data)
    Sigma = svd.singular_values_
    VT = svd.components_
    tranformations.append(('svd', svd))

    tranformations_pipe = Pipeline(tranformations)

    components_name = [f"col{index}" for index in range(0, number_of_dimensions)]
    df = pd.DataFrame(data = USigma, columns = components_name)

    save_data_frame_to_directory(output_dataset, df)
    save_pickle_transform_to_directory(output_model, tranformations_pipe)

    if output_singular_values:
        singular_values = pd.DataFrame(data = Sigma, columns = ["Sigma"])
        save_data_frame_to_directory(output_singular_values, singular_values)
    
    if output_components:
        components = pd.DataFrame(data = VT.T, columns = components_name)
        save_data_frame_to_directory(output_components, components)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", required=True, type=str, help="Input dataset")
    parser.add_argument("--number-of-dimensions", dest="number_of_dimensions", type=int, help="Number of dimensions to reduce to", required=True)
    parser.add_argument("--iterations", dest="iterations", type=int, required=False, default=5)
    parser.add_argument("--solver", dest="solver", type=str, choices=SVD_SOLVERS, help="Solver to calculate SVD")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Transformed dataset")
    parser.add_argument("--output-model", dest="output_model", type=str, help="Trained model")
    parser.add_argument("--output-singular-values", dest="output_singular_values", type=str)
    parser.add_argument("--output-components", dest="output_components", type=str)
    args = parser.parse_args()

    RunModule(**vars(args))
