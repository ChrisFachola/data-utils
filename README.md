# Data Utils

A Python package for efficient data processing and ETL operations, with a focus on memory optimization and large file handling.

## Features

- 📊 Memory-efficient CSV operations
  - Preview headers and data
  - Drop columns
  - Get file dimensions
  - Slice data
  - Split files
  - Concatenate files

- 🧮 Memory optimization utilities
  - Monitor memory usage
  - Optimize DataFrame dtypes
  - Calculate chunking strategies
  - Memory-efficient data processing

- ☁️ BigQuery integration
  - Easy client setup
  - Credential management
  - Environment-based configuration

## Installation

```
pip install data-utils
```

Or with Poetry:
```
poetry add data-utils
```

## Quick Start

### CSV Handling

```python
from data_utils.csv import CsvHandler

# Initialize handler
handler = CsvHandler('path/to/file.csv')

# Preview data
headers, preview = handler.peek(num_lines=5)

# Drop columns
handler.drop_columns_by_name(['column1', 'column2'])

# Split file into parts
handler.split_into_parts('output_prefix', n_parts=3)

# Concatenate multiple files
CsvHandler.concatenate_files(
    input_files=['file1.csv', 'file2.csv'],
    output_file='combined.csv'
)
```

### Memory Optimization

```python
from data_utils.memory import optimize_dataframe_dtypes, get_memory_usage

# Optimize DataFrame dtypes
df_optimized = optimize_dataframe_dtypes(df)

# Get memory usage
memory_info = get_memory_usage()
print(f"Current memory usage: {memory_info['mb']:.2f} MB")
```

### BigQuery Integration

```python
from data_utils.bigquery import get_bigquery_client

# Initialize client
client = get_bigquery_client()

# Or with specific credentials
client = get_bigquery_client(credentials_path='path/to/credentials.json')
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-credentials.json
```

## Development

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/data-utils.git
cd data-utils
```

2. Install dependencies:
```
poetry install
```

3. Install pre-commit hooks:
```
pre-commit install
```

### Running Tests

```
poetry run pytest
```

### Code Quality

This project uses several tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks:
```
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Authors

- Christian Fachola (christian.fachola@gmail.com)

## License
This project is licensed under the GNU General Public License v3.0 - see the [COPYING](COPYING) file for details.
