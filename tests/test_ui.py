import pytest
from src.utils.formatting import preview_upload

def test_preview_upload_none():
    # Test the preview logic when no file path is provided
    preview, path, status = preview_upload(None)
    assert preview is None
    assert path is None
    assert status == "Upload a photo to begin."


def test_preview_upload_invalid_file():
    # Test the preview logic when an invalid file is provided
    preview, path, status = preview_upload("non_existent_file.jpg")
    assert preview is None
    assert path is None
    assert "Could not read that image" in status