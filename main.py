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
PAYLOAD_TEXT = (
    """
def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    * 15
)  # Multiply to increase payload size


def setup_logging(output_file=None):
    """Setup logging with optional file output"""
    handlers: list[logging.Handler] = [logging.StreamHandler()]  # type: ignore
    if output_file:
        handlers.append(logging.FileHandler(output_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=handlers,
    )


def normalize_model_name(model_name):
    """Ensure model name has a tag, append :latest if missing"""
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name


def normalize_model_list(models):
    """Normalize all model names in a list"""
    return [normalize_model_name(model) for model in models]


def get_available_models():
    """Get list of available models from Ollama"""
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


def pull_model(model_name):
    """Pull a model from Ollama registry"""
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


def ensure_models_available(models):
    """Check if models are available and pull missing ones"""
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


def get_token_count(text):
    """Get approximate token count using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Could not use tiktoken: {e}, using word count estimate")
        return int(len(text.split()) * 1.3)


def benchmark_model(model_name, token_count=0):
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
                "prompt": PAYLOAD_TEXT,
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
    args = parser.parse_args()

    # Setup logging with optional file output
    setup_logging(args.file)

    logging.info(f"Starting embedding model benchmark at {datetime.now()}")
    logging.info(f"Models to test: {normalize_model_list(MODELS)}")

    logging.info("Checking model availability...")
    available_models = ensure_models_available(MODELS.copy())

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
        benchmark_model(model, token_count)

    logging.info("Benchmark completed")
