<p align="center">
  <img src="assets/cover.png" alt="Wyvern">
</p>

# Wyvern: An Agentic Framework for Generating Grounded Multimodal Reports

This repository contains the code of Wyvern, a framework that allows the automated generation of technical reports about a topic provided as input from the user.
Wyvern generates grounded, multimodal technical reports (text + figures with supporting citations) and was evaluated against recent baselines.

> ### Key results
> - **Figure informativeness** judged superior to a recent baseline in **87%** of cases (human evaluation).
> - **Overall usefulness**: Wyvern reports preferred over three alternative methods in **63–100%** of pairwise comparisons (human evaluation).
> - **Citation quality**: gains of up to **$2.3\times$ citation recall** and **$1.6\times$ citation precision** over baselines.

See the [paper](https://arxiv.org/pdf/2608.14446) for full methodology and results.

## 1. Preparation of the environment
Create the environment (we have used Python 3.12.11). The default dependencies include `torch==2.7.1`. For GPU acceleration, install the appropriate PyTorch build afterward.
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# For GPU acceleration, install the PyTorch build matching your platform from https://pytorch.org/get-started/locally/
python -m playwright install chromium
```

Install the Pandoc command line tool, which is used to convert the generated Markdown reports to HTML (we have employed Pandoc 3.9.0.2).
Instructions are available [here](https://pandoc.org/installing.html#linux).
```

## 2. Configuration of API keys
Create a `secret.toml` file with the following structure:
```toml
[retriever]
SERPER_API_KEY = "<your-api-key>"

[inference]
BASE_URL = "<your-base-url>"
API_KEY = "<your-api-key>"
REASONING_BASE_URL = "<your-base-url>"
REASONING_API_KEY = "<your-api-key>"
```
`BASE_URL` configures the API endpoint for the main report-generation model, while `REASONING_BASE_URL` configures the endpoint for the reasoning model. They can be the same if both models are served by the same provider.

## 3. Usage
To run the report generator, one needs to define the topic of interest and can choose among various arguments.
The base code line to be used is:

```bash
python main.py
```

Then different options can be specified as descibed in the following sections.

### 3.1 Topic definition
The topic can either be a string that completely defines the topic or it can be described more in detail in a file.

1. In case a string is considered enough to specify the topic, it is sufficient to provide it as argument to the script by adding this option:

    ```bash
    --topic "<topic-title>"
    ```

    For instance, to indicate the desired topic of KV cache, we can write:

    ```bash 
    python main.py --topic "KV cache"
    ``` 

2. In case the user wants to provide a more detailed description, a topic file `topic_info.md` can be created. The structure of the topic file must be as follows:
    ```markdown
    ## Tentative title
    <insert-tentative-title>
    ## Aspects of interest
    <insert-description-of-the-aspects-of-interest>
    ## Aspects not of interest
    <insert-description-of-the-aspects-not-of-interest>
    ## URLs
    * <insert-url>
    * <insert-url>
    * <insert-url>
    ```
    All the specified fields can also be left blank.
    - **Tentative title**: topic definition as a short string.
    - **Aspects of interest**: description of the aspects that we are more interested in about the topic (e.g., "test-time optimizations"). This information will be used to guide the search queries generation.
    - **Aspects not of interest**: description of the aspects that are not of interest. This can allow the search queries to not specifically target those aspects, but the retrieved documents may contain information about it anyway.
    - **URLs**: URLs of webpages to be included in the initial data pool. These can be documents that the users has already collected and deems as interesting.

    Once the file has been created, the following argument must be included when running the script:
    ```bash
    python main.py --topic-file topic_info.md
    ```

### 3.2 Report generator options
There are different arguments that can be used to generate a report, additionally to those described in [section 3.1](#31-topic-definition). The most important ones are listed below:

- `--generate-report`: this argument must be used to generate the report in its most basic format, i.e. without figures, tables, and table of content.

For all the following arguments, either the `--generate-report` argument must be specified to generate a report draft, or a report must be loaded with `--load-report-draft <path-of-the-report>`.

- `--include-images`: this argument includes the image on the report. 
- `--include-table`: include a summary table into the report, to have a general comparative analysis on specific methodologies in the topic domain.
- `--refine-report`: performs a second pass on the report's content section-by-section, verifying its consistency.
- `--auto-correct`: apply the decomposition into atomic claims, their evaluation and eventually their revision.
- `--table-of-content`: this argument allows to include a table of content at the beginning of the report, to improve the readability. It also re-organizes the references numbering.

### 3.3 Example
To generate a report about the topic described in detail in the file `topic_info.md` we can use the following command:

```bash
python main.py --topic-file topic_info.md --generate-report --include-images --include-table --refine-report --auto-correct --table-of-content
```
To set the base and the reasoning model use the arguments `--llm` and `--reasoning-llm`, respectively.

Examples of the generated reports are available [here](./examples/).


## Disclaimer & security notes
This is **research code accompanying a paper**, released for reproducibility. It is **not hardened for production use**. In particular:
- Wyvern **fetches and parses arbitrary web pages** and feeds them to LLMs — treat all retrieved content and model output as untrusted input.
- The pipeline **invokes external tools (e.g. Pandoc) via the shell** on generated file paths. Run it only on inputs and in environments you trust.
- Generated reports are **LLM-produced and may contain errors or unsupported claims** despite the grounding and claims auto-revision stage. Verify before relying on them.


## Citation
If you use Wyvern in your work, please cite:

```
@misc{motetti2026wyvern,
      title={Wyvern: An Agentic Framework for Generating Grounded Multimodal Reports}, 
      author={Beatrice Alessandra Motetti and Emilien Guandalino and Daniele Jahier Pagliari and Alessio Burrello and Lorenz K. Müller and Konstantin Berestizshevsky and Lukas Cavigelli},
      year={2026},
      eprint={2608.14446},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.14446}, 
}
```