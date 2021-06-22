import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram

from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.transformation_directory import save_pickle_transform_to_directory

CONNECTIVITY_TYPE_NONE = 'none'
CONNECTIVITY_TYPE_KNN = 'knn'
CONNECTIVITY_TYPE_GRID = 'grid'
AFFINITY_EUCLIDEAN = 'euclidean'
LINKAGE_WARD = 'ward'
CONNECTIVITY_TYPE = [CONNECTIVITY_TYPE_NONE, CONNECTIVITY_TYPE_KNN, CONNECTIVITY_TYPE_GRID]
AFFINITY = [AFFINITY_EUCLIDEAN, 'l1', 'l2', 'manhattan', 'cosine']
LINKAGE = [LINKAGE_WARD, 'complete', 'average', 'single']


def plot_dendrogram(model, file_name:str, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,10))
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig(file_name, format='png', bbox_inches='tight')

def RunModule(input_dataset: str, number_of_features: int, normalize: bool, connectivity_type: str,
              affinity:str, linkage: str, output_dataset: str, output_model: str):

    data_folder = load_data_frame_from_directory(input_dataset)
    data = data_folder.data
    tranformations = []
    connectivity = None

    if number_of_features <= 0:
        raise ValueError('The number of components cannot be less or equal to zero.')

    if data.shape[-1] <= number_of_features:
        raise ValueError(f'The number of components ({number_of_features}) should be smaller than the number of features ({data.shape[-1]})')

    if linkage == LINKAGE_WARD and affinity != AFFINITY_EUCLIDEAN:
        raise ValueError(f"Affinity '{affinity}' cannot be used with linkage '{LINKAGE_WARD}'. Only '{AFFINITY_EUCLIDEAN} can be used")
 
    if normalize:
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        tranformations.append(('normalize', scaler))

    if connectivity_type == CONNECTIVITY_TYPE_GRID:
        connectivity = grid_to_graph(data[0].shape[0], 1)
    elif connectivity_type == CONNECTIVITY_TYPE_KNN:
        connectivity = kneighbors_graph(data.T, n_neighbors=math.floor(number_of_features/2), mode='connectivity', metric='minkowski', include_self=False)

    agglo = FeatureAgglomeration(connectivity=connectivity, n_clusters=number_of_features, affinity=affinity, linkage=linkage, compute_distances=True).fit(data)
    transformed_data = agglo.transform(data)
    tranformations.append(('agglomeration', agglo))

    tranformations_pipe = Pipeline(tranformations)

    components_name = [f"col{index}" for index in range(0, number_of_features)]
    df = pd.DataFrame(data = transformed_data, columns = components_name)

    # Save outputs
    save_data_frame_to_directory(output_dataset, df)
    save_pickle_transform_to_directory(output_model, tranformations_pipe)
    plot_dendrogram(agglo, 'outputs/dendrogram.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("aml-module")
    parser.add_argument("--dataset", dest="input_dataset", required=True, type=str, help="Input dataset")
    parser.add_argument("--number-of-features", dest="number_of_features", type=int, help="Number of dimensions to reduce to", required=True)
    parser.add_argument("--normalize", dest="normalize", type=bool, help="Whether or not to normalize data to zero mean", required=True, default=True)
    parser.add_argument("--connectivity-type", dest="connectivity_type", choices=CONNECTIVITY_TYPE, default=CONNECTIVITY_TYPE_NONE)
    parser.add_argument("--affinity", dest="affinity", choices=AFFINITY, default=AFFINITY_EUCLIDEAN)
    parser.add_argument("--linkage", dest="linkage", choices=LINKAGE, default=LINKAGE_WARD)
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Transformed dataset")
    parser.add_argument("--output-model", dest="output_model", type=str, help="Trained model")
    args = parser.parse_args()

    RunModule(**vars(args))
