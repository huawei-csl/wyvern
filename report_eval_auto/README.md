# Automatic Evaluation

To automatically evaluate the report using the LLM-as-a-judge approach, you can use the code provided in this subfolder. It supports two evaluation setups:
- **Absolute grading**: assign a score to the input report in a 1-5 Likert scale on different rubrics.
    ```bash
    cd report_eval/automatic
    python eval_absolute.py \
        --report-path <path to the report> \
        --topic-file <path to the JSON file of the topic>
    ```
    Optionally, it is possible to specify also the model and the temperature to be used for the evaluation.

- **Relative grading**: given a pair of reports, assess which report is better over a set of rubrics.
    ```bash
    cd report_eval/automatic
    python eval_relative.py \
        --report-A-path <path to report A> \
        --report-B-path <path to report B> \
        --topic-file <path to the JSON file of the topic>
    ```
    Also in this setting it is possible to specify also the model and the temperature to be used for the evaluation.

In both cases, reports should be provided in Markdown format.