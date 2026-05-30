"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import argparse
import sys

import pooch

from snicarfx import Session

ZENODO_RECORD = "20457918"
ARCHIVE_NAME = "snicar-fx-data.zip"

# MD5 hash locks this code to a specific data version
DATA_VERSION_HASH = "md5:1bac8bd07512e2fcf017ef2fb123daca"


def main():
    """Download and extract the data archive to the project root."""

    # setup argument parser
    argparse.ArgumentParser(
        description=(
            "Download and extract snicar-fx data archive in snicar-fx data folder."
            )
    )

    # setup pooch
    downloader = pooch.create(
        path=pooch.os_cache("snicarfx"),
        base_url=f"doi:10.5281/zenodo.{ZENODO_RECORD}",
        registry={ARCHIVE_NAME: DATA_VERSION_HASH},
    )

    # extract to snicar-fx package root
    extract_path = Session.get_package_root()

    print(f"Downloading {ARCHIVE_NAME} (Version: {DATA_VERSION_HASH}...)")
    print(f"Extracting to: {extract_path}")

    # download and unzip in snicarfx root
    try:
        downloader.fetch(
            ARCHIVE_NAME,
            processor=pooch.Unzip(extract_dir=extract_path),
            progressbar=True,
        )
        print("Download finished.")

    except ValueError as e:
        print("\nError: Data version mismatch or download failed.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
