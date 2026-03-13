import requests
from pathlib import Path
import xml.etree.ElementTree as ET


def main(base_url: str, output_dir: Path, depth: str):
    # check input data type
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    if not isinstance(output_dir, Path):
        raise TypeError("Output directory must be a pathlib.Path object.")
    if not isinstance(depth, str):
        raise TypeError("Depth must be a string")
    # check input values
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if any(Path(output_dir).iterdir()):
        raise FileExistsError(f'Output directory {output_dir} must be empty.')
    try: int(depth)
    except: raise ValueError("Depth must be an integer in the form of a string")

    # controls how deep in folder structure to download files
    headers = {"Depth": depth}

    # WebDAV requires PROPFIND
    response = requests.request("PROPFIND", base_url, headers=headers)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    files = []
    namespaces = {"d": "DAV:"}
    for resp in root.findall("d:response", namespaces):
        href = resp.find("d:href", namespaces).text
        if href.endswith("/"):
            continue

        filename = href.split("/")[-1]
        files.append(filename)

    print(f"Found {len(files)} files")

    for file in files:
        url = base_url + file
        output_path = output_dir / file

        print("Downloading", file)

        r = requests.get(url)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)

    print("Done.")

if __name__ == "__main__":
    BASE_URL = "https://cloud.geo.tuwien.ac.at/public.php/dav/files/JZnp7H8CAtJtbT5/"
    OUTPUT_DIR = Path("../data/raw/ASCAT_ERA5")
    main(base_url=BASE_URL, output_dir=OUTPUT_DIR, depth="1")