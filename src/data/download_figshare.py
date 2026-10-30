from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

ARTICLE_ID = 1512427
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination.name}")
        return
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as output, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
                progress.update(len(chunk))

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    article = response.json()
    print(f"Figshare article: {article.get('title', 'Brain tumor dataset')}")
    for item in article.get("files", []):
        name = item["name"]
        destination = RAW_DIR / name
        download_file(item["download_url"], destination)
        if destination.suffix.lower() == ".zip":
            extract_dir = RAW_DIR / destination.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(extract_dir)
            print(f"Extracted: {destination.name}")

if __name__ == "__main__":
    main()
