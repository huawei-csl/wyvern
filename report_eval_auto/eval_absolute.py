import argparse
import copy
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import backoff
from langchain_openai import ChatOpenAI

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils.figures_extraction import FiguresMasker
from utils.general import remove_urls
from utils.prompting import queryLLM, set_api_keys


def format_prompt(prompt_template: str, 
                  topic: str, 
                  rubric: dict, 
                  report: str) -> str:
    """Fill the prompt template with rubric and report.

    Parameters
    ----------
    - prompt_template [`str`]: template for the prompt to be filled
    - topic [`str`]: topic of the report
    - rubric [`dict`]: dictionary with the description of the criterion, and of the scores
    - report [`str`]: report to be evaluated

    Output
    ------
    - `str`: filled version of the prompt
    """
    prompt_template_copy = copy.deepcopy(prompt_template)
    data = copy.deepcopy(rubric)
    data.update({
        "instruction": f"You are a Wikipedia editor. Your task is write a wikipedia page for the topic: {topic}",
        "response": report
    })
    filled_prompt = prompt_template_copy.format(**data)
    return filled_prompt


@backoff.on_exception(backoff.constant, Exception, interval=0, max_tries=3)
def get_scores_dict(report: str,
                    topic: str,
                    model: ChatOpenAI,
                    prompt_template_path: str,
                    rubric_path: str) -> dict:
    """Get the result of the absolute evaluation with the model.
    
    Parameters
    ----------
    - report [`str`]: text of the report to be evaluated
    - topic [`str`]: topic of the report
    - model [`langchain_openai.ChatOpenAI`]: model for the evaluation,
    - prompt_template_path [`str`]: path to the file with the template for the prompt,
    - rubric_path [`str`]: path to the file with the rubrics

    Output
    ------
    - `dict`: return the scores dictionary
    """
    grading = {}
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()
    with open(rubric_path, 'r') as file:
        rubrics = json.load(file)

    for rubric in rubrics:
        grading[rubric["criteria_description"]] = {}
        
        prompt = format_prompt(prompt_template=prompt_template, topic=topic, rubric=rubric, report=report)

        output = queryLLM(
            model, prompt
        )

        match = re.split(r'\[?RESULT\]?', output)
        if len(match) >= 2:
            feedback = match[-2].strip()
            score = match[-1].replace(":", "").strip()
        else:
            raise ValueError(f"The structure of the model's response was not parsable: {output}")

        grading[rubric["criteria_description"]] = {"feedback": feedback, "score": score}
    return grading



def main(args):
    set_api_keys()

    model = ChatOpenAI(
        model_name=args.model_id,
        temperature=args.temperature,
        max_tokens=10000,
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"]
        )
    
    # Set a default path where to save the results of the evaluation if not provided
    if args.save_path is None:
        SAVE_PATH = (Path(args.report_path).parents[1] / "evaluation" / "automatic" / "absolute").as_posix()
    else:
        SAVE_PATH = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving the results of the evaluation to '{SAVE_PATH}'")

    experiment_time = f"{datetime.today():%Y-%m-%d_%H-%M-%S}"

    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        filename=os.path.join(SAVE_PATH, f"evaluation_log_{experiment_time}.log"),
        level=logging.INFO,
        format='%(asctime)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',)
    logger.handlers = [h for h in logger.handlers \
                       if not isinstance(h, logging.StreamHandler)]
    logger.info("Process has PID {}".format(os.getpid()))
    logger.info(f"{str(args)}")

    logger.info(f"Evaluating report '{args.report_path}'")
    logger.info(f"Evaluating report with {args.model_id}")

    evaluation_results = {}
    evaluation_results["report_path"] = args.report_path
    evaluation_results["model_id"] = args.model_id

    # ** Load the topic file **
    with open(args.topic_file) as f:
        json_file = json.load(f)
    if (json_file["topic_description"] is not None) and (json_file["topic_description"] != "null"):
        TOPIC = json_file["topic"] + ". " + json_file["topic_description"]
    else:
        TOPIC = json_file["topic"]
    evaluation_results["topic"] = TOPIC
    logger.info(f"The topic of the report is: {TOPIC}")
    
    # ** Load the report **
    with open(args.report_path) as f:
        report = f.read()
    # Mask figures and links
    fig_masker = FiguresMasker()
    report = fig_masker.replace_base64_with_links(report)
    report = report.replace("<sup>", "").replace("</sup>", "").replace("<small>", "").replace("</small>", "")
    report = remove_urls(report)
    if "## References" in report:
        report = report.split("## References")[0]
    else:
        report = report.split("# References")[0]
    
    # ** Conduct the evaluation **
    grading_dict = get_scores_dict(
        report=report,
        topic=TOPIC,
        model=model,
        prompt_template_path=args.prompt_template_path,
        rubric_path=args.rubric_path
        )
    
    # ** Save the evaluation results **
    evaluation_results["results"] = grading_dict
    with open(os.path.join(
        SAVE_PATH, 
        f"results_{model.model_name.split("/")[-1]}_{experiment_time}.json"
        ), 'w') as file:
        json.dump(evaluation_results, file)

    from utils.prompting import (total_input_tokens, total_output_tokens)
    logger.info("Total input tokens: {}, total output tokens: {}".format(
        total_input_tokens, total_output_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", type=str, help="Path to the report.")
    parser.add_argument("--topic-file", type=str, required=True,
                        help="Path of the JSON file with the topic information.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="""Path where to save the results of the evaluation. If not provided,
                        the results will be saved in the evaluation folder of the report.""")
    parser.add_argument("--prompt-template-path", type=str,
                        default=str(Path(__file__).resolve().parent / "resources" / "template_absolute_eval.prompt"),
                        help="Path to prompt template for the absolute evaluation.")
    parser.add_argument("--rubric-path", type=str, 
                        default=str(Path(__file__).resolve().parent / "resources" / "rubrics.json"), 
                        help="Path to the JSON file with the rubrics.")
    parser.add_argument("--model-id", type=str, help="Model to use for the rubric evaluation.")
    parser.add_argument("--temperature", type=float, help="Temperature for the inference.")

    args = parser.parse_args()

    main(args)