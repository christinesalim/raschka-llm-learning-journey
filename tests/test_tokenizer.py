import sys
from pathlib import Path
import re
import requests

# Add src to path so we can import raschka_llm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from raschka_llm.tokenizer import SimpleTokenizer, build_vocab_from_text
from raschka_llm.dataloader import TextDataset


def test_simple_tokenize():
    """Test basic tokenization with re.split."""
    text = "Hello, world!"
    # Simulate the tokenization that happens in encode method
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]
    print(f"\nSimple tokenize result: {tokens}")
    assert "Hello" in tokens
    assert "," in tokens
    assert "world" in tokens
    assert "!" in tokens


def test_tokenizer_class():
    """Test SimpleTokenizer class encode/decode."""
    # Create a simple vocabulary
    vocab = {
        "Hello": 0,
        ",": 1,
        "world": 2,
        "!": 3,
        "How": 4,
        "are": 5,
        "you": 6,
        "?": 7
    }
    tokenizer = SimpleTokenizer(vocab)

    # Test encode method
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    print(f"\nEncoded IDs: {ids}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert ids == [0, 1, 2, 3]

    # Test decode method
    decoded = tokenizer.decode(ids)
    print(f"Decoded text: {decoded}")
    assert isinstance(decoded, str)
    assert "Hello" in decoded
    assert "world" in decoded


def test_tokenizer_encode_decode():
    """Test encoding and decoding with punctuation."""
    vocab = {
        "It": 0,
        "'": 1,
        "s": 2,
        "a": 3,
        "test": 4,
        "!": 5
    }
    tokenizer = SimpleTokenizer(vocab)

    text = "It's a test!"

    # Encode
    ids = tokenizer.encode(text)
    print(f"\nEncoded IDs: {ids}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # Should encode: ["It", "'", "s", "a", "test", "!"]
    assert ids == [0, 1, 2, 3, 4, 5]

    # Decode
    decoded = tokenizer.decode(ids)
    print(f"Decoded text: {decoded}")
    assert "It" in decoded
    assert "test" in decoded


def test_from_file_and_tokenize():
    """Test loading from file with TextDataset."""
    repo_root = Path(__file__).resolve().parent.parent
    sample = repo_root / "data" / "sample.txt"
    assert sample.exists(), "sample.txt should exist for tests"

    # Test that TextDataset can load the file
    ds = TextDataset.from_file(sample)
    assert len(ds) >= 1
    tokens = ds[0]
    print(f"\nDataset first line tokens: {tokens}")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1


def test_build_vocab_from_file():
    """Test building vocab from a file."""
    repo_root = Path(__file__).resolve().parent.parent
    sample = repo_root / "data" / "sample.txt"
    assert sample.exists(), "sample.txt should exist for tests"

    # Read the file
    with open(sample, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the text
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]

    # Build a simple vocab from unique tokens
    vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}

    print(f"\nBuilt vocab with {len(vocab)} unique tokens from file")
    assert len(vocab) > 0
    assert isinstance(vocab, dict)

    # Test that we can create a tokenizer with this vocab
    tokenizer = SimpleTokenizer(vocab)
    assert tokenizer.str_to_int == vocab
    assert len(tokenizer.int_to_str) == len(vocab)


def test_build_vocab_from_text_function():
    """Test build_vocab_from_text function with downloaded file."""
    repo_root = Path(__file__).resolve().parent.parent
    file_path = repo_root / "data" / "the-verdict.txt"

    # Download the file if it doesn't exist
    if not file_path.exists():
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        print(f"\nDownloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(response.content)

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Build vocab using our function
    vocab = build_vocab_from_text(text)
    
    #Debug
    print ("Last 5 vocab entries")
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    print(f"\nBuilt vocab with {len(vocab)} unique tokens")
    print(f"First 10 tokens: {list(vocab.keys())[:10]}")

    # Verify vocab has special tokens
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab
    assert len(vocab) > 0
    assert isinstance(vocab, dict)

    # Test that we can create a tokenizer with this vocab
    tokenizer = SimpleTokenizer(vocab)

    # Test encoding/decoding
    test_text = "Hello, world!"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {test_text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")

    assert isinstance(ids, list)
    assert isinstance(decoded, str)

