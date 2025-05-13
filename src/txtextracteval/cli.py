import argparse
import logging
import sys
import os
import logging.handlers # Import for file handling

# Import necessary functions from the package
# from txtextracteval import load_config, run_experiment, generate_markdown_report, __version__ # Commented out report
from txtextracteval import load_config, run_experiment, generate_markdown_report, __version__ # Import without report
# Keep extractor import only if needed for direct single-method runs (maybe remove later)
from txtextracteval.extractors import TesseractExtractor

# Basic logging setup (consider moving to __init__.py or a dedicated logging setup function)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Use logger configured in __init__.
# logger = logging.getLogger("txtextracteval.cli") # Logger obtained after setup

def main():
    # --- Setup Logging --- #
    LOG_FILENAME = 'txtextracteval.log' # Log file name
    DEFAULT_LOG_LEVEL = logging.INFO
    VERBOSE_LOG_LEVEL = logging.DEBUG

    # Define the format for log messages
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create a handler that writes log records to a file (append mode)
    file_handler = logging.FileHandler(LOG_FILENAME, mode='a')
    file_handler.setFormatter(log_formatter)
    # File handler level will be set based on verbosity args later

    # Create a handler that writes log records to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    # Console handler level will be set based on verbosity args later

    # Get the root logger for the package
    package_logger = logging.getLogger('txtextracteval')
    # Set the initial lowest level (DEBUG) to capture everything from modules
    # Handlers will filter further based on args
    package_logger.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)
    package_logger.addHandler(console_handler)
    package_logger.propagate = False # Prevent root logger double handling

    # Get a logger specific to this CLI module AFTER setup
    logger = logging.getLogger("txtextracteval.cli")
    # --- End Logging Setup --- #

    parser = argparse.ArgumentParser(
        description=f"txtextracteval v{__version__}: Benchmark text extraction methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config_file", help="Path to the YAML configuration file defining the experiment.")
    parser.add_argument("--src_img", help="Path to a single source image file (for quick runs without config file).")
    parser.add_argument("--gt_file", help="Path to the ground truth text file for the single source image.")
    parser.add_argument("--out_dir", help="Directory where results (report, images) will be saved. Overrides config file setting if provided.")
    # Potentially add --models, --transforms for simple CLI runs later
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging to file and console.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Adjust HANDLER logging levels based on verbose flag
    if args.verbose:
        file_handler.setLevel(VERBOSE_LOG_LEVEL)
        console_handler.setLevel(VERBOSE_LOG_LEVEL)
        logger.info("Verbose logging (DEBUG) enabled for file and console.")
    else:
        file_handler.setLevel(DEFAULT_LOG_LEVEL) # Log INFO and higher to file
        console_handler.setLevel(DEFAULT_LOG_LEVEL) # Log INFO and higher to console
        # Optional: You could set console to WARNING if you want less console output by default
        # console_handler.setLevel(logging.WARNING)

    config = None
    results = []

    try:
        if args.config_file:
            logger.info(f"Loading experiment from config file: {args.config_file}")
            config = load_config(args.config_file)
        elif args.src_img and args.gt_file:
            logger.info("Running simple evaluation for single image.")
            # Construct a minimal config for a simple run
            # Uses defaults for methods (Tesseract only), transforms (None), metrics, report name
            # Output directory defaults or uses --out_dir
            output_dir_simple = args.out_dir if args.out_dir else "./txtextract_results_simple"
            config = {
                "images": [args.src_img],
                "ground_truth": [args.gt_file],
                "methods": [{"type": "tesseract"}], # Default to just Tesseract for simple run
                "transformations": [], # No transforms for simple run
                "metrics": ["cer", "wer", "latency"], # Default metrics
                "output": {
                    "directory": output_dir_simple,
                    "report_filename": "simple_report.md"
                }
            }
            logger.warning("Simple run mode uses default Tesseract method and no transformations.")
            # Basic validation for simple run files
            if not os.path.exists(args.src_img):
                 raise FileNotFoundError(f"Source image not found: {args.src_img}")
            if not os.path.exists(args.gt_file):
                 raise FileNotFoundError(f"Ground truth file not found: {args.gt_file}")
        else:
            parser.error("Either --config_file or both --src_img and --gt_file must be provided.")

        # Override output directory from config if specified via CLI arg
        if args.out_dir:
            config['output']['directory'] = args.out_dir
            logger.info(f"Output directory overridden by CLI argument: {args.out_dir}")

        # --- Run the experiment --- #
        logger.info("Starting experiment run...")
        results = run_experiment(config)
        logger.info("Experiment run finished.")

        # --- Generate the report --- #
        if results:
            output_dir = config['output']['directory']
            report_filename = config['output']['report_filename']
            logger.info(f"Generating report: {os.path.join(output_dir, report_filename)}")
            # Pass args.config_file as config_file_path. It will be None if not provided (e.g. simple run)
            generate_markdown_report(results, config, output_dir, report_filename, config_file_path=args.config_file)
            logger.info(f"Report generation complete. Results are in: {output_dir}")
        else:
            logger.warning("Experiment produced no results. Generating empty report.")
            # Still generate a basic report indicating no results
            output_dir = config['output']['directory']
            report_filename = config['output']['report_filename']
            os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
            # Call with empty results
            generate_markdown_report([], config, output_dir, report_filename, config_file_path=args.config_file)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or value error: {e}")
        sys.exit(1)
    except ImportError as e:
         logger.error(f"Import error (missing dependency?): {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Use exception for traceback
        sys.exit(1)

if __name__ == "__main__":
    main() 