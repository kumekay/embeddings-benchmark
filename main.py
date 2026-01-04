import time
import requests
import logging
import argparse
import os
import tiktoken
from datetime import datetime

# Configuration
OLLAMA_URL = str(os.getenv("OLLAMA_URL", "http://localhost:11434")).removesuffix("/")
OLLAMA_EMBEDDINGS_URL = f"{OLLAMA_URL}/api/embeddings"
OLLAMA_TAGS_URL = f"{OLLAMA_URL}/api/tags"
OLLAMA_PULL_URL = f"{OLLAMA_URL}/api/pull"
MODELS = ["qwen3-embedding:0.6b", "embeddinggemma:300m", "nomic-embed-text"]
# A dummy "Code" payload to test heavy load
PAYLOAD_TEXT = """
from typing import List, TypeVar, Sequence
T = TypeVar("T")
def quicksort(arr: Sequence[T]) -> List[T]:
    '''
    Sort a sequence using the quicksort algorithm.

    This implementation uses a functional approach that creates new lists
    for the left, middle, and right partitions around a pivot element.

    Args:
        arr: A sequence of comparable items (numbers, strings, etc.)

    Returns:
        A new list containing the elements from arr in sorted order

    Raises:
        TypeError: If the elements in arr cannot be compared with <, ==, and >

    Time Complexity:
        - Best case: O(n log n) when pivot selection results in balanced partitions
        - Average case: O(n log n) for randomly ordered input
        - Worst case: O(n²) when pivot selection results in highly unbalanced partitions
          (e.g., already sorted arrays with poor pivot selection)

    Space Complexity: O(n) for the additional lists created during partitioning
        Plus O(log n) for the recursion stack in average case, O(n) in worst case

    Example:
        >>> quicksort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
        >>> quicksort(['z', 'a', 'c', 'b'])
        ['a', 'b', 'c', 'z']
    '''
    if len(arr) <= 1:
        return list(arr)  # Convert to list to ensure consistent return type

    try:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]  # type: ignore
        middle = [x for x in arr if x == pivot]  # type: ignore
        right = [x for x in arr if x > pivot]  # type: ignore

    except TypeError as e:
        raise TypeError(f"Elements in the sequence must be comparable: {e}")

    return quicksort(left) + middle + quicksort(right)
"""


def setup_logging(output_file: str | None = None) -> None:
    """Setup logging with optional file output.

    Args:
        output_file: Optional file path to write log output to.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]  # type: ignore
    if output_file:
        handlers.append(logging.FileHandler(output_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=handlers,
    )


def normalize_model_name(model_name: str) -> str:
    """Ensure model name has a tag, append :latest if missing.

    Args:
        model_name: The model name to normalize.

    Returns:
        Model name with a tag appended if missing.
    """
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name


def normalize_model_list(models: list[str]) -> list[str]:
    """Normalize all model names in a list.

    Args:
        models: List of model names to normalize.

    Returns:
        List of normalized model names.
    """
    return [normalize_model_name(model) for model in models]


def get_available_models() -> list[str]:
    """Get list of available models from Ollama.

    Returns:
        List of available model names.
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        else:
            logging.error(f"Failed to get models: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        return []


def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama registry.

    Args:
        model_name: Name of the model to pull.

    Returns:
        True if pull was successful, False otherwise.
    """
    logging.info(f"Pulling model: {model_name}")
    try:
        response = requests.post(
            OLLAMA_PULL_URL, json={"name": model_name}, stream=True
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode("utf-8")
                        if data.strip():
                            logging.info(f"Pull progress: {data}")
                    except Exception:
                        pass
            logging.info(f"Successfully pulled model: {model_name}")
            return True
        else:
            logging.error(f"Failed to pull {model_name}: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Error pulling model {model_name}: {e}")
        return False


def ensure_models_available(models: list[str]) -> list[str]:
    """Check if models are available and pull missing ones.

    Args:
        models: List of model names to ensure are available.

    Returns:
        List of successfully available model names.
    """
    # Normalize model names first
    normalized_models = normalize_model_list(models)
    original_to_normalized = dict(zip(models, normalized_models))

    available_models = get_available_models()
    logging.info(f"Available models: {available_models}")

    # Check for missing models using normalized names
    missing_models = [
        (original, normalized)
        for original, normalized in original_to_normalized.items()
        if normalized not in available_models
    ]

    if missing_models:
        missing_normalized = [normalized for _, normalized in missing_models]
        logging.info(f"Missing models: {missing_normalized}")

        # Pull missing models and track successes
        successful_models = []
        for original, normalized in missing_models:
            if pull_model(normalized):
                successful_models.append(normalized)
            else:
                logging.error(
                    f"Failed to pull model {normalized}, skipping benchmark for this model"
                )

        # Update the models list with successfully available ones
        final_models = [
            normalized
            for normalized in normalized_models
            if normalized in available_models or normalized in successful_models
        ]
    else:
        logging.info("All required models are available")
        final_models = normalized_models

    return final_models


def get_token_count(text: str) -> int:
    """Get approximate token count using tiktoken.

    Args:
        text: Text to count tokens for.

    Returns:
        Approximate token count.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Could not use tiktoken: {e}, using word count estimate")
        return int(len(text.split()) * 1.3)


def benchmark_model(model_name: str, token_count: int = 0, difficulty: int = 1) -> None:
    """Benchmark an embedding model.

    Args:
        model_name: Name of the model to benchmark.
        token_count: Token count of the payload text for throughput calculation.
    """
    logging.info(f"--- Benchmarking: {model_name} ---")

    # 1. Warmup (load model into RAM/VRAM)
    try:
        requests.post(
            OLLAMA_EMBEDDINGS_URL, json={"model": model_name, "prompt": "warmup"}
        )
    except Exception as e:
        logging.error(f"Failed to load {model_name}: {e}")
        return

    # 2. Run Benchmark (30 iterations)
    durations = []
    for i in range(30):
        start = time.time()
        response = requests.post(
            OLLAMA_EMBEDDINGS_URL,
            json={
                "model": model_name,
                "prompt": PAYLOAD_TEXT * difficulty,
                "options": {"num_ctx": 4096},  # Ensure context handles the load
            },
        )
        end = time.time()

        if response.status_code == 200:
            durations.append(end - start)
        else:
            logging.error(f"Error: {response.text}")

    # 3. Calculate Stats
    avg_time = sum(durations) / len(durations)

    logging.info(f"Average Latency: {avg_time:.4f} seconds")

    if token_count > 0:
        tps = token_count / avg_time
        logging.info(f"Throughput: {tps:.2f} tokens/sec")

    logging.info(f"-----------------------------------")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument(
        "-f", "--file", type=str, help="Output file for logging results"
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        type=int,
        default=4,
        help="Difficulty multiplier for payload size, each unit approximately 500 tokens (default: 4)",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        help="Model to test (can be repeated multiple times to test multiple models)",
    )
    args = parser.parse_args()

    # Setup logging with optional file output
    setup_logging(args.file)

    # Use custom models if provided, otherwise use defaults
    models_to_test = args.models if args.models else MODELS

    logging.info(f"Starting embedding model benchmark at {datetime.now()}")
    logging.info(f"Models to test: {normalize_model_list(models_to_test)}")

    logging.info("Checking model availability...")
    available_models = ensure_models_available(models_to_test.copy())

    if not available_models:
        logging.error("No models available for benchmarking")
        exit(1)

    # Get accurate token count
    token_count = get_token_count(PAYLOAD_TEXT)
    logging.info(f"Token count: {token_count:.0f}")

    if args.file:
        logging.info(f"Results will be saved to: {args.file}")
    else:
        logging.info("Results will only be displayed (no file output)")
    logging.info("=" * 50)

    for model in available_models:
        benchmark_model(model, token_count, difficulty=args.difficulty)

    logging.info("Benchmark completed")
