import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from jobtools.arguments import StringEnum

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory

class SVDSolver(StringEnum):
    ARPACK = 'arpack'
    RANDOMIZED = 'randomized'

def run_module(dataset: str, output_dataset: str, output_model: str, output_singular_values: str, output_components: str,
               number_of_dimensions: int, solver: SVDSolver, iterations: int = 5):
    data_folder = load_data_frame_from_directory(dataset)

    if data_folder.schema:
        if any([col['type'] != 'Numeric' for col in data_folder.schema['columnAttributes']]):
            raise TypeError('Dataset contains non-numeric columns. Cannot apply SVD to non-numeric types.')
    else:
        print('Column type checking ignored as schema not available.')
    
    if number_of_dimensions <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    data = data_folder.data
    tranformations = []

    svd = TruncatedSVD(n_components=number_of_dimensions, n_iter=iterations, algorithm=str(solver)).fit(data)
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
