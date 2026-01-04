# Computation benchmark for embeddings models

A simple tool to check how fast are different embeddings models are on your machine. You will need [Ollama](https://ollama.com/) installed.

The script uses the following default models:
- qwen3-embedding:0.6b
- embeddinggemma:300m
- nomic-embed-text

You can customize the Ollama server URL by setting the `OLLAMA_URL` environment variable (default: http://localhost:11434).

### Benchmark Details

The benchmark performs the following steps for each model:
1. Warmup: Loads the model into RAM/VRAM
2. Performance test: Runs 30 iterations with a code payload
3. Reports average latency and throughput (tokens/second)

The test payload is a Python quicksort implementation with documentation, providing a realistic code snippet for embedding testing.


## Usage

Run the benchmark with:

```bash
uv run python main.py
```

By default, output is only displayed in the console. To save results to a file, use the `-f` option:

```bash
uv run python main.py -f output.txt
```

### Command Line Options

- `-f, --file FILE`: Save logging output to the specified file (optional)
- `-d, --difficulty INT`: Difficulty multiplier for payload size, each unit approximately 500 tokens (default: 4)
- `-m, --model MODEL`: Model to test (can be repeated multiple times to test multiple models)
- `-h, --help`: Show help message

### Examples

```bash
# Display results only (default behavior)
uv run python main.py

# Save results to a custom file
uv run python main.py -f benchmark_results.txt

# Test with custom difficulty
uv run python main.py -d 8

# Test specific models
uv run python main.py -m nomic-embed-text -m qwen3-embedding:0.6b

# Test with custom difficulty and save to file
uv run python main.py -d 2 -f custom_results.txt -m nomic-embed-text

# Show help
uv run python main.py --help
```
