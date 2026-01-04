import time
import requests
import logging
import argparse
import tiktoken
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELS = ["qwen3-embedding:0.6b", "qwen3-embedding:4b", "nomic-embed-text"]
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
        requests.post(OLLAMA_URL, json={"model": model_name, "prompt": "warmup"})
    except Exception as e:
        logging.error(f"Failed to load {model_name}: {e}")
        return

    # 2. Run Benchmark (30 iterations)
    durations = []
    for i in range(30):
        start = time.time()
        response = requests.post(
            OLLAMA_URL,
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
    logging.info(f"Models to test: {MODELS}")
    # Get accurate token count
    token_count = get_token_count(PAYLOAD_TEXT)
    logging.info(f"Token count: {token_count:.0f}")

    if args.file:
        logging.info(f"Results will be saved to: {args.file}")
    else:
        logging.info("Results will only be displayed (no file output)")
    logging.info("=" * 50)

    for model in MODELS:
        benchmark_model(model, token_count)

    logging.info("Benchmark completed")
