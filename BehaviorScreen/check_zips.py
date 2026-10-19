"""Why do some Marques zips fail the integrity check? Compares sizes with Mendeley and tests each zip."""
import sys
import zipfile
import zlib
from pathlib import Path

from marques_cv import API, PARTS, VERSION, get_json, subfolders

root = Path(sys.argv[1])
listed = {}
for dataset in PARTS:
    for fid in ["root", *subfolders(dataset)]:
        for f in get_json(f"{API}/{dataset}/files?folder_id={fid}&version={VERSION}"):
            listed[f["filename"]] = f["size"]

for path in sorted(root.rglob("*.zip")):
    expected = listed.get(path.name)
    print(f"{path.name}: {path.stat().st_size} bytes on disk, {expected} listed by Mendeley")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            print(f"  opens: {len(members)} members, zip64: {any(m.file_size >= 2**32 or m.header_offset >= 2**32 for m in members)}")
            print(f"  first bad member: {archive.testzip()}")
    except (zipfile.BadZipFile, zlib.error, OSError) as error:
        print(f"  {type(error).__name__}: {error}")
