import numpy as np
import shutil
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, DataFrameDirectory
from azureml.studio.core.io.transformation_directory import PickleTransformationDirectory

from modules.aml_extract_pca.extract_pca import RunModule as extract_pca

def test_pca_matches_shapes():
    args = {
        'input_dataset': 'modules/samples_data/dataset',
        'number_of_dimensions': 5,
        'normalize': True,
        'solver': 'randomized',
        'output_dataset': 'modules/samples_data/transformed',
        'output_model': 'modules/samples_data/transform',
        'output_eigenvectors': 'modules/samples_data/eigen'
    }

    extract_pca(**args)

    input_data = load_data_frame_from_directory(args['input_dataset']).data
    output_data = load_data_frame_from_directory(args['output_dataset']).data
    output_eigen = load_data_frame_from_directory(args['output_eigenvectors']).data

    assert(input_data.shape[0] == output_data.shape[0])
    assert(output_data.shape[-1] == args['number_of_dimensions'])
    assert(output_eigen.shape[-1] == args['number_of_dimensions'])

    if args['normalize']:
        assert(np.all(np.isclose(output_data.mean(axis=0), 0)))
    
    shutil.rmtree(args['output_model'])
    shutil.rmtree(args['output_dataset'])
    shutil.rmtree(args['output_eigenvectors'])