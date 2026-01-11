from pathlib import Path

from raschka_llm import TextDataset, simple_tokenize


def test_from_file_and_tokenize(tmp_path):
    # Use the sample file included in the repo
    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "data" / "sample.txt"
    assert sample.exists(), "sample.txt should exist for tests"

    ds = TextDataset.from_file(sample)
    assert len(ds) >= 1
    tokens = ds[0]
    assert isinstance(tokens, list)
    assert len(tokens) >= 1

    s = "hello world"
    toks = simple_tokenize(s)
    assert toks == ["hello", "world"]
