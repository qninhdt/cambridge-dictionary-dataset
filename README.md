# Cambridge Dictionary Dataset & Sense Alignment Pipeline

A Python-based pipeline to collect, crawl, and align word senses from the Cambridge Dictionary. This project provides utilities to crawl entries, store them in SQLite, generate ground truth mappings using LLMs, and evaluate/run production-grade sense alignment using a hybrid LLM and Cross-Encoder approach.

## Features

- **Collector**: Extract word list and entries from Cambridge Dictionary browse pages.
- **Crawler**: Crawl detailed page content (senses, definitions, examples, IPA, parts of speech) and store them in an SQLite database.
- **Ground Truth Generator**: Generate gold standard alignments using LLMs.
- **Evaluator**: Evaluate sense alignment pipelines (e.g. Cross-Encoder reranker + LLMs).
- **Production Aligner**: Run the final production alignment with checkpoint support.

## Project Structure

- `run.py`: Command Line Interface (CLI) entry point.
- `src/`: Core Python modules.
  - `crawler/`: Browse collectors and page crawlers.
  - `alignment/`: LLM alignment, evaluation, and production runner.
  - `utils/`: Common utilities.
- `data/`: Directory where crawled database, ground truth, and outputs are stored.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/qninhdt/cambridge-dictionary-dataset.git
   cd cambridge-dictionary-dataset
   ```

2. **Configure environment variables:**
   Create a `.env` file in the root directory and add your keys (e.g. for LLM providers):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=your_custom_api_base_url_if_any
   ```

## Usage

You can run the pipeline commands using `run.py`:

### 1. Collect Words
Collect entry words from Cambridge browse pages:
```bash
python run.py collect --output cambridge_entries.tsv
```

### 2. Crawl Dictionary Entries
Crawl detailed page content and store it in SQLite:
```bash
python run.py crawl --words cambridge_entries.tsv --db data/cambridge.db --workers 5
```

### 3. Generate Ground Truth Alignments
Generate LLM-aligned mappings for evaluation:
```bash
python run.py generate-gt --model gpt-4o --output data/ground_truth.json
```

### 4. Evaluate Pipeline
Evaluate performance of the hybrid model:
```bash
python run.py evaluate --model gpt-4o --gt data/ground_truth.json
```

### 5. Run Sense Alignment
Run production sense alignment with checkpointing:
```bash
python run.py align --model gpt-4o --output data/word_senses_alignment.json
```

## License

This project is licensed under the [MIT License](LICENSE).
