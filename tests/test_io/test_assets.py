import json

import pytest

from pymshbm.io.assets import load_assets, prepare_assets


def test_missing_assets_has_setup_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match='pymshbm-fetch-assets'):
        load_assets(tmp_path)


def test_setup_validates_surface_directory_before_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match='cortex.label'):
        prepare_assets(tmp_path / 'out', tmp_path / 'subjects')
    assert not (tmp_path / 'out').exists()


def test_manifest_tampering_fails_before_loading_mat(tmp_path):
    names = ['MSHBM_prior_15.mat', 'lh.cortex.label', 'rh.cortex.label', 'ColorMap_15.txt']
    for name in names:
        (tmp_path / name).write_bytes(b'changed')
    (tmp_path / 'assets.json').write_text(json.dumps({
        'files': {name: {'sha256': '0' * 64} for name in names},
    }))
    with pytest.raises(ValueError, match='checksum'):
        load_assets(tmp_path)
