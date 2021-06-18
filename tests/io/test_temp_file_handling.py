import os

import pytest
from kiez.io.temp_file_handling import create_tempfile_preferably_in_dir


@pytest.mark.parametrize("persistent", [True, False])
def test_normal_dir_handling(persistent, tmp_path):
    suffix = "testsuffix"
    prefix = "testprefix"
    tmp = create_tempfile_preferably_in_dir(
        suffix=suffix, prefix=prefix, directory=tmp_path, persistent=persistent
    )
    assert os.path.basename(tmp).startswith(prefix)
    assert os.path.basename(tmp).endswith(suffix)
    assert str(tmp_path) in tmp


@pytest.mark.parametrize("persistent", [True, False])
def test_unavailable_handling(persistent):
    suffix = "testsuffix"
    prefix = "testprefix"
    tmp = create_tempfile_preferably_in_dir(
        suffix=suffix, prefix=prefix, directory="/notexistent", persistent=persistent
    )
    assert os.path.basename(tmp).startswith(prefix)
    assert os.path.basename(tmp).endswith(suffix)
