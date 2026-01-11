"""Minimal CLI to exercise the package when running `python -m raschka_llm`.
"""
from pathlib import Path
import urllib.request

try:
    from .dataloader import TextDataset
except ImportError:
    # Support running this file directly for debugging
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import TextDataset

def download_verdict_file():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = Path(__file__).parent.parent.parent.joinpath("data", "the-verdict.txt")
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded the-verdict.txt to {file_path}")


def main():
    sample = Path(__file__).parent.parent.parent.joinpath("data", "sample.txt")
    if sample.exists():
        ds = TextDataset.from_file(sample)
        print(f"Loaded {len(ds)} lines from {sample}")
        print("First line tokens:", ds[0])
    else:
        print("No sample data found. Create `data/sample.txt` to try this out.")

    # Download the test file
    download_verdict_file()


if __name__ == "__main__":
    main()
