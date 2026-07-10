"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from unittest.mock import MagicMock, patch

import pooch
import pytest
import requests

from snicarfx.cli.download import ARCHIVE_NAME, DATA_VERSION_HASH, ZENODO_RECORD, main


def test_zenodo_metadata_and_hash(api_url):
    """
    Verifies connectivity and data version integrity based on JSON
    metadata (and without downloading actual data files).

    """

    # test that Zenodo repository can be reached
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        # get metadata
        data = response.json()

    except requests.exceptions.ConnectionError:
        pytest.fail("Cannot connect to Zenodo. Check internet connection.")
    except requests.exceptions.Timeout:
        pytest.fail("Connection to Zenodo timed out.")
    except requests.exceptions.HTTPError as e:
        pytest.fail(f"Zenodo returned an error: {e}")
    except ValueError as e:
        pytest.fail(f"Failed to parse Zenodo response as JSON: {e}")

    # look for the archive name in remote files
    files = data.get("files", [])
    target_file = None
    for f in files:
        if f["key"] == ARCHIVE_NAME:
            target_file = f
            break

    # raise error if no file name in the archive matches the one
    # expected by snicar-fx
    if not target_file:
        pytest.fail(f"File '{ARCHIVE_NAME}' not found in Zenodo record {ZENODO_RECORD}")

    # get remote hash
    remote_hash = target_file.get("checksum", "")

    # compare with hash currently expected by snicar-fx
    if remote_hash != DATA_VERSION_HASH:
        pytest.fail(
            f"Data version mismatch!\n"
            f"Expected: {DATA_VERSION_HASH}\n"
            f"Remote:   {remote_hash}\n"
            f"Please update DATA_VERSION_HASH in src/snicarfx/cli/download.py"
        )


def test_main_execution_path(capsys):
    """
    Test that the main() function executes correctly. The actual
    fetch/extraction is mocked to avoid redundant work during tests.
    """
    with patch("snicarfx.cli.download.pooch.create") as mock_create:
        # setup mock to simulate a successful download without doing it
        mock_downloader = MagicMock()
        mock_create.return_value = mock_downloader

        # execute the function under test
        main()

        # verify the logic flowed correctly
        mock_create.assert_called_once()
        mock_downloader.fetch.assert_called_once()

        # verify the fetch was called with the correct arguments
        call_args = mock_downloader.fetch.call_args
        assert call_args.args[0] == ARCHIVE_NAME
        assert isinstance(call_args.kwargs["processor"], pooch.Unzip)

        # verify output messages were printed
        captured = capsys.readouterr()
        assert "Downloading" in captured.out
        assert "Download finished." in captured.out


def test_main_execution_error_path(capsys):
    """
    Test that the main() function handles errors correctly.
    """
    with (
        patch("snicarfx.cli.download.pooch.create") as mock_create,
        patch("snicarfx.cli.download.sys.exit") as mock_exit,
    ):
        # setup mock to simulate a failure
        mock_downloader = MagicMock()
        mock_downloader.fetch.side_effect = ValueError("Simulated download error")
        mock_create.return_value = mock_downloader

        # execute the function
        main()

        # verify error handling logic ran
        mock_exit.assert_called_once_with(1)
        captured = capsys.readouterr()
        assert "Error: Data version mismatch" in captured.err
