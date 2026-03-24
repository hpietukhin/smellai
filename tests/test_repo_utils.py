import shutil
from pathlib import Path
from repo_utils import (
    create_temp_repo_folder,
    remove_temp_repo_folder,
    temp_repo_context,
)


def test_create_temp_repo_folder():
    temp_dir = create_temp_repo_folder()
    try:
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        # Check if path contains temp
        assert "temp" in str(temp_dir)
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_remove_temp_repo_folder():
    temp_dir = create_temp_repo_folder()
    assert temp_dir.exists()

    remove_temp_repo_folder(temp_dir)
    assert not temp_dir.exists()


def test_remove_temp_repo_folder_safety():
    # Create a non-temp dir
    non_temp_dir = Path("non_temp_test_dir")
    if non_temp_dir.exists():
        non_temp_dir.rmdir()
    non_temp_dir.mkdir()

    try:
        # Try to remove it using the safe function
        remove_temp_repo_folder(non_temp_dir)

        # It should still exist because of safety check
        assert non_temp_dir.exists()
    finally:
        # Cleanup manually
        if non_temp_dir.exists():
            non_temp_dir.rmdir()


def test_temp_repo_context():
    path_captured = None
    with temp_repo_context() as temp_dir:
        assert temp_dir.exists()
        path_captured = temp_dir

    assert not path_captured.exists()
