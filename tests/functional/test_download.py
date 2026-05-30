"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pytest
import requests
from snicarfx.cli.download import ZENODO_RECORD, ARCHIVE_NAME, DATA_VERSION_HASH


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
