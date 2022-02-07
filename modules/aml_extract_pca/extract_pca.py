import pandas as pd
from numpy.core.fromnumeric import partition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from jobtools.runner import TaskRunner
from jobtools.arguments import StringEnum

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory

class PCASolvers(StringEnum):
    AUTO = 'auto'
    FULL = 'full'
    ARPACK = 'arpack'
    RANDOMIZED = 'randomized'

def RunModule(dataset: str, number_of_dimensions: int, solver: PCASolvers, 
              output_dataset: str, output_model: str, output_eigenvectors: str, normalize: bool = True):
    data_folder = load_data_frame_from_directory(dataset)

    if data_folder.schema:
        if any([col['type'] != 'Numeric' for col in data_folder.schema['columnAttributes']]):
            raise TypeError('Dataset contains non-numeric columns. Cannot apply PCA to non-numeric types.')
    else:
        print('Column type checking ignored as schema not available.')
    
    if number_of_dimensions <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    data = data_folder.data
    tranformations = []

    if normalize:
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        tranformations.append(('normalize', scaler))

    pca = PCA(n_components=number_of_dimensions, svd_solver=str(solver), whiten=True).fit(data)
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
    tr = TaskRunner()
    tr.run(RunModule)