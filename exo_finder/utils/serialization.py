import json
import zipfile
from pathlib import Path
from typing import Any

import msgpack


def unzip_msgpack(file_path: Path) -> Any:
    with zipfile.ZipFile(file_path, "r") as zf:
        with zf.open(file_path.stem) as f:
            # Use MessagePack to unpack the data
            return msgpack.unpackb(f.read(), strict_map_key=False)


def zip_msgpack(data: Any, path: Path):
    # Pack data using MessagePack
    packed_data = msgpack.packb(data)

    # Create a ZIP file and write the packed data
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write the packed data to a file inside the ZIP
        # The filename inside the ZIP is the stem of the provided path
        zf.writestr(path.stem, packed_data)


def unzip_json_file(file_path: Path, parse_json: bool = True) -> dict | str:
    with zipfile.ZipFile(file_path, "r") as zf:
        file_name = file_path.stem if file_path.stem in zf.namelist() else zf.namelist()[0]
        with zf.open(file_name) as f:
            decoded = f.read().decode("utf-8")
            return json.loads(decoded) if parse_json else decoded


def zip_json_data(data: dict | str, file_path: Path):
    """
    Zips JSON data to a file that can be read by unzip_json_file.

    Args:
        data: Dictionary to be serialized as JSON
        file_path: Path where the zip file will be saved
    """
    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert data to JSON string and encode as UTF-8
    if isinstance(data, dict):
        data = json.dumps(data)

    json_bytes = data.encode("utf-8")

    # Create a zip file with the JSON data
    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # The file inside the zip should have the same name as the zip file (without extension)
        zf.writestr(file_path.stem, json_bytes)
