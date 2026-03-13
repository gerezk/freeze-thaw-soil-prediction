import requests
import os
import xml.etree.ElementTree as ET

BASE_URL = "https://cloud.geo.tuwien.ac.at/public.php/dav/files/JZnp7H8CAtJtbT5/"
OUTPUT_DIR = "../data/raw/ASCAT_ERA5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Depth": "1"
}

# WebDAV requires PROPFIND
response = requests.request("PROPFIND", BASE_URL, headers=headers)

response.raise_for_status()

root = ET.fromstring(response.text)

namespaces = {
    "d": "DAV:"
}

files = []

for resp in root.findall("d:response", namespaces):
    href = resp.find("d:href", namespaces).text

    if href.endswith("/"):
        continue

    filename = href.split("/")[-1]
    files.append(filename)

print(f"Found {len(files)} files")

for file in files:
    url = BASE_URL + file
    output_path = os.path.join(OUTPUT_DIR, file)

    print("Downloading", file)

    r = requests.get(url)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(r.content)

print("Done.")