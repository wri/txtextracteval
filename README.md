# txtextracteval: Text Extraction Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`txtextracteval` is a Python toolkit for comparing text extraction methods on images of printed documents, particularly those captured by mobile phones which may suffer from quality issues. It provides an end-to-end pipeline to evaluate and compare various approaches, including:

*   Traditional OCR engines (Tesseract and EasyOCR)
*   Vision-enabled Large Language Model APIs (from providers like Google Gemini, OpenAI, or Anthropic)
*   Local Transformer models (e.g., TrOCR, SmolVLM via HuggingFace, or via Ollama)


The toolkit automates image preprocessing (applying transformations like blur, rotation, brightness changes), runs the configured extraction methods on the original and transformed images, computes accuracy (CER, WER) and performance (latency, cost) metrics against ground truth text, and generates a consolidated Markdown report for easy comparison.

This experiment aims to help teams understand which method works best for different image qualities and scenarios, guiding informed decisions for real-world applications.

## Features

*   **Multi-Method Comparison:** Evaluate OCR engines, API providers, and local model pipelines side-by-side.
*   **Image Degradation Simulation:** Apply configurable transformations (blur, brightness, rotation) to test robustness against common mobile capture issues.
*   **Comprehensive Metrics:** Calculate Character Error Rate (CER), Word Error Rate (WER), latency (seconds per extraction), and estimated cost (for API methods).
*   **YAML Configuration:** Define experiments reproducibly using a clear YAML format.
*   **Command-Line Interface:** Run evaluations easily using `uv run txtextracteval` with either a config file or direct image/GT arguments.
*   **Detailed Markdown Reports:** Automatically generate reports including transformed image previews, extracted text outputs, and summary tables for easy analysis.
*   **Extensible:** Designed with a modular structure to facilitate adding new extraction methods or image transformations.

## Installation

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/wri/txtextracteval.git 
    cd txtextracteval
    ```

2.  **Create and activate virtual environment:**
    `uv` handles this seamlessly. Run:
    ```bash
    uv venv
    ```
    This creates a `.venv` directory (if it doesn't exist) and activates it.

3.  **Install dependencies:**
    Sync the environment with the project's dependencies listed in `pyproject.toml`:
    ```bash
    uv sync
    ```
    This will install `txtextracteval` and all its dependencies (like `opencv-python`, `pytesseract`, `transformers`, `google-genai`, etc.).

### API Key / Endpoint Setup

*   **LLM API provider:** If using the `llm_api` method with `provider: gemini`, `provider: openai`, or `provider: anthropic` you need an API key.
    *   Create a file named `.env` in the project root directory.
    *   Add your API key to the `.env` file:
        ```dotenv
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ANTHROPIC_API_KEY="YOUR_API_KEY_HERE"
        OPENAI_API_KEY="YOUR_API_KEY_HERE"
        ```
    The application uses `python-dotenv` to load this key automatically. Alternatively, set the corrsponding API key environment variable directly in your shell.

*   **Ollama:** If using the `llm_api` method with `provider: ollama`, ensure you have an Ollama instance running locally.
    *   Follow the [Ollama installation guide](https://ollama.com/).
    *   Make sure the desired multimodal model (e.g., `llava`, `gemma3` with vision support) is pulled: `ollama pull gemma3`.
    *   The default endpoint used by the tool is `http://localhost:11434`. You can override this in the configuration if needed.

## Configuration (`config.yaml`)

Experiments are defined using a YAML file (e.g., `config.yaml`). Here's an example structure:

```yaml
# List of input image paths or path to a directory
images:
  - data/samples/receipt.jpg
  - data/samples/document.png

# List of corresponding ground truth text file paths
# Must match the order of images if both are lists
ground_truth:
  - data/samples/receipt.txt
  - data/samples/document.txt

# List of extraction methods to evaluate
methods:
  - type: tesseract # Uses local Tesseract installation
    config: # Optional Tesseract settings
      lang: eng # Language (e.g., eng+fra for multiple)
      psm: 3    # Page Segmentation Mode

  - type: hf_ocr # Uses Hugging Face transformers
    config:
      # Optional: Specify model (defaults to ds4sd/SmolDocling-256M-preview)
      model: microsoft/trocr-base-printed
      # device: 0 # Optional: Specify GPU device index (defaults to CPU: -1)

  - type: llm_api # Uses LLM APIs
    config:
      provider: gemini # 'gemini' or 'ollama'
      model: gemini-1.5-flash-latest # Specific Gemini model
      # api_key_env: CUSTOM_GEMINI_KEY_VAR # Optional: Override default env var name
      prompt: "Extract the text content accurately." # Optional: Custom prompt

  - type: llm_api
    config:
      provider: ollama
      model: llava # Model running on local Ollama instance
      endpoint: http://127.0.0.1:11434 # Optional: Override default Ollama endpoint
      timeout: 180 # Optional: Timeout in seconds for API call (default: 120)

# Optional: List of image transformations to apply
transformations:
  - name: blur # Name matching registered transform function
    params:
      kernel_size: 3 # Parameters for the transform function
  - name: rotate
    params:
      angle: -5
  - name: brightness
    params:
      factor: 0.7 # Decrease brightness by 30%

# Optional: List of metrics to calculate (defaults shown)
metrics:
  - cer
  - wer
  - latency # Automatically included via ExtractionResult
  - cost    # Automatically included via ExtractionResult

# Output configuration
output:
  directory: ./evaluation_results/run_01 # Where to save report and variants
  report_filename: comparison_report.md # Name of the Markdown report
```

## Usage

Run experiments using the CLI:

```bash
# Ensure your virtual environment is active or use uv run directly
uv run txtextracteval --config_file path/to/your/config.yaml
```

**Options:**

*   `--config_file` (`-c`): Path to the YAML configuration file (required for full experiments).
*   `--src_img`: Path to a single source image (for quick test runs). Requires `--gt_file`.
*   `--gt_file`: Path to the ground truth text for the single source image. Requires `--src_img`.
*   `--out_dir`: Output directory path. Overrides the `directory` setting in the config file if provided.
*   `--verbose` (`-v`): Enable detailed DEBUG level logging.
*   `--version`: Show the version number and exit.
*   `--help`: Show help message and exit.

**Simple Run Example (without config file):**

This runs only the default Tesseract method with no transformations on a single image.

```bash
uv run txtextracteval --src_img data/samples/receipt.jpg --gt_file data/samples/receipt.txt --out_dir ./simple_run_output
```

## Output

The tool generates the following in the specified output directory:

1.  **Markdown Report (`<report_filename>.md`):** Contains:
    *   Experiment overview.
    *   A summary table comparing average metrics (latency, cost, CER, WER) across all methods.
    *   Detailed results for each source image, broken down by image variant (original + transformations).
    *   For each variant, an embedded preview of the transformed image.
    *   A table showing the extracted text (truncated), metrics, latency, cost, and any errors for each method applied to that variant.
2.  **Image Variants:** Saved versions of the original image and each applied transformation (e.g., `image1_original.png`, `image1_blur_k3.png`). These are linked in the report.

## Running Tests

To run the test suite (requires `pytest` and `pytest-mock`):

```bash
uv run pytest
```
Fair warning: the test suite is not complete, or even functioning half decently.

## Extensibility

This codebase is designed to be extensible:

*   **Extractors:** Add a new class inheriting from `BaseExtractor` in `src/txtextracteval/extractors/`, implement the `extract` method, and register the class key/type in `EXTRACTOR_REGISTRY` within `src/txtextracteval/runner.py`.
*   **Transformations:** Add a new function in `src/txtextracteval/transforms/` (e.g., `opencv_transforms.py`) that takes a NumPy array image and parameters, returns a transformed NumPy array, and register its name/function in `TRANSFORM_REGISTRY` within `src/txtextracteval/runner.py`.

## TODO

- [ ] Refactor the approach to using HuggingFace models
- [ ] Revisit the testing suite

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
