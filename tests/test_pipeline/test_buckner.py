from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pymshbm.io.bids import SurfaceRun
from pymshbm.pipeline.buckner import read_surface_bold, run_buckner_workflow, validate_fit
from pymshbm.types import MSHBMParams


def test_read_gifti_timepoint_arrays_preserves_vertex_order(tmp_path):
    data = np.arange(15, dtype=np.float32).reshape(3, 5)
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(x, intent='NIFTI_INTENT_TIME_SERIES') for x in data])
    path = tmp_path / 'bold.func.gii'
    nib.save(image, path)
    np.testing.assert_array_equal(read_surface_bold(path, expected_vertices=5), data)


def test_read_single_matrix_gifti_uses_vertices_by_time(tmp_path):
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    path = tmp_path / 'bold.func.gii'
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data, intent='NIFTI_INTENT_TIME_SERIES')]), path)
    np.testing.assert_array_equal(read_surface_bold(path, expected_vertices=5), data.T)


def test_wrong_mesh_fails_clearly(tmp_path):
    path = tmp_path / 'bold.func.gii'
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.zeros(5, np.float32))]), path)
    with pytest.raises(ValueError, match='40962'):
        read_surface_bold(path)


def test_empty_run_list_and_existing_output_fail_before_fitting(tmp_path):
    with pytest.raises(ValueError, match='No runs'):
        run_buckner_workflow([], tmp_path / 'out', tmp_path)
    run = SurfaceRun('01', None, 'rest', None, Path('lh'), Path('rh'))
    (tmp_path / 'out').mkdir()
    with pytest.raises(FileExistsError):
        run_buckner_workflow([run], tmp_path / 'out', tmp_path)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf])
def test_invalid_concentration_cannot_be_published(bad):
    params = MSHBMParams(mu=np.ones((3, 2)), epsil=np.ones(2), sigma=np.array([1., bad]),
                         theta=np.ones((4, 2)) / 2, kappa=np.ones(2),
                         s_lambda=np.ones((4, 2, 1)) / 2)
    with pytest.raises(ValueError, match='concentration'):
        validate_fit(params)
