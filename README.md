# Computation benchmark for embeddings models

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
- `-h, --help`: Show help message

### Examples

```bash
# Display results only (default behavior)
uv run python main.py

# Save results to a custom file
uv run python main.py -f benchmark_results.txt

# Show help
uv run python main.py --help
```
