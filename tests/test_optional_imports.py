import pytest

from spatialcore.utils import optional_import


def test_optional_import_error_message():
    with pytest.raises(ImportError, match="required for this workflow"):
        optional_import("package_that_should_not_exist_spatialcore", "spatial")
