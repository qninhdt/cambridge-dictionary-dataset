#!/usr/bin/env python3
import argparse
import sys
import os
from dotenv import load_dotenv

# Ensure the root directory is on the path so we can import src.*
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="VocaGraph construction, crawling, and alignment pipeline CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="The command to run")

    # Command: collect
    collect_parser = subparsers.add_parser(
        "collect", 
        help="Collect words and entries from Cambridge dictionary browse pages."
    )
    collect_parser.add_argument(
        "--output", 
        default="cambridge_entries.tsv", 
        help="Path to output TSV file (default: cambridge_entries.tsv)"
    )

    # Command: crawl
    crawl_parser = subparsers.add_parser(
        "crawl", 
        help="Crawl detailed entry content from Cambridge dictionary and store in SQLite."
    )
    crawl_parser.add_argument(
        "--words", 
        help="Path to words file (slugs/entries) to import into DB before starting"
    )
    crawl_parser.add_argument(
        "--db", 
        default="data/cambridge.db", 
        help="SQLite database path (default: data/cambridge.db)"
    )
    crawl_parser.add_argument(
        "--workers", 
        type=int, 
        default=5, 
        help="Number of parallel crawler workers (default: 5)"
    )
    crawl_parser.add_argument(
        "--load-only", 
        action="store_true", 
        help="Only load the words list into the database, don't start the crawling"
    )
    crawl_parser.add_argument(
        "--stats", 
        action="store_true", 
        help="Show crawler progress statistics and exit"
    )

    # Command: generate-gt
    gt_parser = subparsers.add_parser(
        "generate-gt", 
        help="Generate gold standard / ground truth alignments using an LLM."
    )
    gt_parser.add_argument(
        "--model", 
        required=True, 
        help="Model ID to use (e.g. cd/gpt-5.5)"
    )
    gt_parser.add_argument(
        "--reasoning-effort", 
        default="medium", 
        help="Reasoning effort: low, medium, high, or none (default: medium)"
    )
    gt_parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of words to align"
    )
    gt_parser.add_argument(
        "--output", 
        default="data/ground_truth.json", 
        help="JSON output path (default: data/ground_truth.json)"
    )
    gt_parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel LLM calling workers (default: 16)"
    )
    gt_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of tasks to group into a single LLM request (default: 1)"
    )

    # Command: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate the hybrid alignment pipeline (Jina Cross-Encoder + LLM)."
    )
    eval_parser.add_argument(
        "--model", 
        required=True, 
        help="Model ID to resolve ambiguous senses"
    )
    eval_parser.add_argument(
        "--reasoning-effort", 
        default="medium", 
        help="Reasoning effort: low, medium, high, or none (default: medium)"
    )
    eval_parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of words to evaluate"
    )
    eval_parser.add_argument(
        "--gt", 
        default="data/ground_truth.json", 
        help="Ground truth JSON file path (default: data/ground_truth.json)"
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel evaluation/LLM workers (default: 16)"
    )
    eval_parser.add_argument(
        "--no-reranker",
        action="store_false",
        dest="use_reranker",
        default=True,
        help="Disable Cross-Encoder reranker and route all senses directly to LLM"
    )
    eval_parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.35,
        help="Cross-Encoder score threshold to automatically accept mapping (default: 0.35)"
    )
    eval_parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.05,
        help="Cross-Encoder score threshold to automatically reject mapping (default: 0.05)"
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of tasks to group into a single LLM request (default: 1)"
    )
    eval_parser.add_argument(
        "--difficulty-method",
        choices=["mul", "add", "token"],
        default="token",
        help="Method to calculate task difficulty: 'mul' for multiplication (grid size), 'add' for addition (item sum), 'token' for tiktoken-based prompt token count (default: token)"
    )

    # Command: align
    align_parser = subparsers.add_parser(
        "align", 
        help="Run the production alignment pipeline for the entire database (LLM only, no reranker, with checkpointing)."
    )
    align_parser.add_argument(
        "--model", 
        required=True, 
        help="Model ID to resolve ambiguous senses"
    )
    align_parser.add_argument(
        "--reasoning-effort", 
        default="none", 
        help="Reasoning effort: low, medium, high, or none (default: none)"
    )
    align_parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of words to align"
    )
    align_parser.add_argument(
        "--output", 
        default="data/word_senses_alignment.json", 
        help="JSON output/checkpoint path (default: data/word_senses_alignment.json)"
    )
    align_parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel evaluation/LLM workers (default: 16)"
    )
    align_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of tasks to group into a single LLM request (default: 8)"
    )
    align_parser.add_argument(
        "--difficulty-method",
        choices=["mul", "add", "token"],
        default="token",
        help="Method to calculate task difficulty: 'mul' for multiplication (grid size), 'add' for addition (item sum), 'token' for tiktoken-based prompt token count (default: token)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "collect":
        from src.crawler.word_collector import collect_words
        print(f"Collecting entries to: {args.output}")
        collect_words(args.output)

    elif args.command == "crawl":
        from src.crawler.dictionary_crawler import crawl_dictionary
        crawl_dictionary(
            words_file=args.words,
            db_path=args.db,
            workers=args.workers,
            load_only=args.load_only,
            stats_only=args.stats
        )

    elif args.command == "generate-gt":
        from src.alignment.ground_truth_generator import generate_ground_truth
        generate_ground_truth(
            model_id=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
            output_path=args.output,
            workers=args.workers,
            batch_size=args.batch_size
        )

    elif args.command == "evaluate":
        from src.alignment.hybrid_evaluator import evaluate_hybrid_mapping
        evaluate_hybrid_mapping(
            model_id=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
            gt_path=args.gt,
            workers=args.workers,
            use_reranker=args.use_reranker,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
            batch_size=args.batch_size,
            difficulty_method=args.difficulty_method
        )

    elif args.command == "align":
        from src.alignment.aligner import align_database
        align_database(
            model_id=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
            output_path=args.output,
            workers=args.workers,
            batch_size=args.batch_size,
            difficulty_method=args.difficulty_method
        )



if __name__ == "__main__":
    main()
