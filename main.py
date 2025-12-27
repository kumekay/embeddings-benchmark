import time
import requests
import json
import logging
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODELS = ["qwen3-embedding:0.6b", "qwen3-embedding:4b", "nomic-embed-text"]

# A dummy "Code" payload (approx 500 tokens) to test heavy load
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


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("log_benchmark.txt"), logging.StreamHandler()],
)


def benchmark_model(model_name):
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
    # Rough token count estimate (words * 1.3)
    est_tokens = len(PAYLOAD_TEXT.split()) * 1.3
    tps = est_tokens / avg_time

    logging.info(f"Average Latency: {avg_time:.4f} seconds")
    logging.info(f"Est. Throughput: {tps:.2f} tokens/sec")
    logging.info(f"-----------------------------------")


if __name__ == "__main__":
    logging.info(f"Starting embedding model benchmark at {datetime.now()}")
    logging.info(f"Models to test: {MODELS}")
    logging.info("=" * 50)

    for model in MODELS:
        benchmark_model(model)

    logging.info("Benchmark completed")
