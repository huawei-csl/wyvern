import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from utils.general import (convert_markdown_to_html, find_last_report_version,
                           handle_exception, write_generated_report)
from utils.prompting import set_api_keys
from utils.report_generation import (NumberingHandler, OutlineGenMode,
                                     ReportGenerator, ReportGenType)
from utils.scrape import url_to_filename
from utils.search import QueryType, SearchModule, TopicDefiner


def main(args):
    TOPIC = args.topic
    topic_descr = args.topic_description
    given_urls = args.urls
    TOOL = args.main_tool
    GEN_MODE = args.report_gen_mode
    EXPERIMENTS_BASE = args.path_experiments
    EXPERIMENT_TIME_ID = f"{datetime.today():%Y-%m-%d_%H-%M-%S}"
    EXPERIMENT_FOLDER = os.path.join(EXPERIMENTS_BASE,
                                    TOPIC.replace("/", "_"),
                                    str(GEN_MODE),  
                                    EXPERIMENT_TIME_ID)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    plot_folder = os.path.join(EXPERIMENT_FOLDER, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    with open(os.path.join(EXPERIMENT_FOLDER, "topic.json"), "w", encoding='utf-8') as fp:
        json.dump({
            "topic": TOPIC,
            "topic_description": topic_descr,
            "given_urls": given_urls
        }, fp, ensure_ascii=False, indent=4)

    os.environ["TOKEN_LOG"] = os.path.join(EXPERIMENT_FOLDER, "token_log.log")
    
    # Logging
    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        filename=os.path.join(EXPERIMENT_FOLDER, "log.log"),
        level=logging.INFO,
        format='%(asctime)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',)

    logger.handlers = [h for h in logger.handlers \
                       if not isinstance(h, logging.StreamHandler)]
    logger.info("Process has PID {}".format(os.getpid()))
    logger.info(f"{str(args)}")

    # logger.info(topic_str_info)

    # Set the uncaught exception handler
    sys.excepthook = handle_exception
    report_file_path_out = ""
    # pandarallel.initialize(nb_workers=args.nb_workers, progress_bar=True)

    # -------------------------------------
    # LLM selection
    # -------------------------------------
    TEMPERATURE = args.temperature
    
    # Set a rate limiter
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    # Define the base and the reasoning agents
    llm = ChatOpenAI(
        model_name=args.llm,
        temperature=args.temperature,
        rate_limiter=rate_limiter,
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        timeout=300
        )
    BASE_MODEL_ID = llm.model_name
    BASE_MODEL_NAME = llm.model_name.split("/")[1]
    
    reasoning_llm = ChatOpenAI(
        model_name=args.reasoning_llm,
        temperature=args.temperature,
        rate_limiter=rate_limiter,
        base_url=os.environ["REASONING_BASE_URL"],
        api_key=os.environ["REASONING_API_KEY"],
        timeout=300
        )
    REASONING_MODEL_ID = reasoning_llm.model_name
    REASONING_MODEL_NAME = reasoning_llm.model_name.split("/")[1]

    LOAD_RESEARCH = args.load_research

    search_module = SearchModule(
        topic=TOPIC, 
        topic_description=topic_descr, 
        experiment_folder=EXPERIMENT_FOLDER
        )
    
    if LOAD_RESEARCH is None:
        df = search_module.search(
            model=llm,
            reasoning_model=reasoning_llm,
            num_queries=args.num_queries, 
            num_top_results=args.num_top_results, 
            query_gen_type=args.query_gen_type, 
            length_upper_filter=args.length_upper_filter,
            completeness_threshold=args.completeness_threshold,
            search_threshold=args.search_threshold,
            link_relevance_threshold=args.link_relevance_threshold,
            max_new_links=args.max_new_links,
            sim_threshold=args.sim_threshold,
            tool=TOOL, 
            given_urls=given_urls)
    else:
        df = search_module.load_research(LOAD_RESEARCH)
    
    # Save content and summaries in a specific folder for convenience of checks
    db_folder = os.path.join(EXPERIMENT_FOLDER, "db")
    os.makedirs(db_folder, exist_ok=True)
    for ind, row in df.iterrows():
        with open(os.path.join(db_folder,
                               f"{ind + 1}_{url_to_filename(row.url)}.md"), "w") as g:
            if isinstance(row[f"content"], str):
                g.write(row[f"content"])
        with open(os.path.join(db_folder,
                               f"{ind + 1}_SUMMARY_{url_to_filename(row.url)}.md"), "w") as g:
            g.write(row.summary)

    # -------------------------------------
    # Summaries concatenation
    # -------------------------------------
    summaries_string = ""
    for i, row in enumerate(df.iterrows()):
        summaries_string = summaries_string + \
            f"\n[{i + 1}, {row[1].identifier}] {row[1].summary}"

    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "logs"), exist_ok=True)
    with open(os.path.join(EXPERIMENT_FOLDER, 
                           "logs", 
                           "information_string.txt"), "w") as f:
        f.write(summaries_string)

    # -------------------------------------
    # Report generation
    # -------------------------------------
    report_folder = os.path.join(EXPERIMENT_FOLDER, "reports")
    os.makedirs(report_folder, exist_ok=True)

    report_generator = ReportGenerator(
        llm=llm,
        reasoning_llm=reasoning_llm,
        topic=TOPIC,
        information=summaries_string,
        references_df=df,
        topic_description=topic_descr
        )
    
    if args.load_report_draft is not None:
        logger.info(f"Loading the report from '{args.load_report_draft}'...")
        with open(args.load_report_draft) as f:
            report = f.read()

    if args.generate_report:
        logger.info("Generating the report...")
        report = report_generator.generate(mode=GEN_MODE, 
                                           outline_gen_mode=args.outline_gen_mode)

    if (args.load_report_draft is not None) or (args.generate_report):
        report_file_path = os.path.join(report_folder, 
                                        f"report_v{find_last_report_version(report_folder) + 1}.md")
        write_generated_report(report_file_path, report)

        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )


    if args.include_images:
        if LOAD_RESEARCH is None:
            figures_path = str(Path(report_folder).parent / "figures")
        else:
            if args.figures_folder is not None:
                figures_path = args.figures_folder
            else:
                figures_path = str(Path(LOAD_RESEARCH).parent.parent / "figures")
        logger.info(f"Including images from folder '{figures_path}'...")
        figures_log_path = os.path.join(EXPERIMENT_FOLDER, "figures_logs")

        report = report_generator.include_images(
            figures_path=figures_path,
            report_text=report,
            log_path=figures_log_path,
        )   

        report_file_path = os.path.join(
            report_folder, 
            f"report_v{find_last_report_version(report_folder) + 1}_with_images.md"
            )
        write_generated_report(report_file_path, report)

        # Convert from markdown to HTML with pandoc
        # NOTE: pypandoc was not applying style correctly, so using command line 
        # version
        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )

    if args.include_table:
        logger.info("Including a summarization table...")
        report = report_generator.include_table(report)

        report_file_path = os.path.join(
            report_folder, 
            f"report_v{find_last_report_version(report_folder) + 1}_with_table.md"
            )
        write_generated_report(report_file_path, report)

        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )

    if args.refine_report:
        logger.info("Refining the report...")
        # Refine the report flow section by section
        report = report_generator.generate(ReportGenType.REFINE,
                                           report)
        report_file_path = os.path.join(
            report_folder, 
            f"report_v{find_last_report_version(report_folder) + 1}_refined.md"
            )
        write_generated_report(report_file_path, report)
        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )


    if args.do_citation_check:
        logger.info("Checking the citations...")
        cit_log_folder = os.path.join(EXPERIMENT_FOLDER, "cit_log")
        os.makedirs(cit_log_folder, exist_ok=True)
        report = report_generator.check_citations(report=report,
                                                  information=summaries_string,
                                                  log_folder=cit_log_folder)

        report_file_path = os.path.join(
            report_folder, 
            f"report_v{find_last_report_version(report_folder) + 1}_checked_citations.md"
            )
        write_generated_report(report_file_path, report)
        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )

    if args.enrich_sections is not None:
        logger.info(f"Enrich sections {args.enrich_sections}...")
        sections_to_refine = [el.strip() for el in args.enrich_sections.strip('[]').split(',')]
        report = report_generator.enrich_sections(report, sections_to_refine)
        report_file_path = os.path.join(
            report_folder, 
            f"report_v{find_last_report_version(report_folder) + 1}_enriched.md"
            )
        write_generated_report(report_file_path, report)
        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out,
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )

    if args.evaluate or args.auto_correct:
        # Evaluate the report
        logger.info("Running the 1st evaluation round...")
        recall_df = report_generator.evaluate(report, 
                                              os.path.join(EXPERIMENT_FOLDER,
                                                           "evaluation"))
        
    if args.auto_correct:
        # Run the auto-correction of the claims
        report = report_generator.auto_correct_claims(report,
                                                      recall_df,
                                                      report_folder)
        
        if args.evaluate:
            logger.info("""Running the 2nd evaluation round after the auto-correction
            of the claims...""")
            recall_df = report_generator.evaluate(report, 
                                                os.path.join(EXPERIMENT_FOLDER,
                                                            "evaluation"))

    if args.table_of_content:
        logger.info("Including the table of content...")
        num_report = find_last_report_version(report_folder) + 1

        num_handler = NumberingHandler()
        report = num_handler.adjust_figures_identifiers(report)
        report, new_ref_df = num_handler.adjust_references_identifiers(report, report_generator.references_df)
        new_ref_df.to_csv(os.path.join(EXPERIMENT_FOLDER, "dataframes", "adjusted_df.csv"))
        
        # Convert for markdown
        report_toc_md = report_generator.include_toc(report, type="markdown")
        report_file_path = os.path.join(
            report_folder, 
            f"report_v{num_report}_toc.md"
            )
        write_generated_report(report_file_path, report_toc_md)

        # Convert for html
        report_toc_html = report_generator.include_toc(report, type="html")
        report_file_path = os.path.join(
            report_folder, 
            f"report_v{num_report}_toc_TMP_HTML.md"
            )
        write_generated_report(report_file_path, report_toc_html)
        report_file_path_out = report_file_path.split(".md")[0] + ".html"
        convert_markdown_to_html(
            report_file_path,
            report_file_path_out.replace("_TMP_HTML", ""),
            page_title=f"Report: {TOPIC}",
            embed_resources=True,
        )
        os.remove(report_file_path)

    else:
        logger.info("Not including the table of content...")

    from utils.prompting import (total_input_tokens, total_output_tokens)
    logger.info("Total tokens")
    logger.info(f"Total input tokens: {total_input_tokens}")
    logger.info(f"Total output tokens: {total_output_tokens}")
    return report_file_path_out.replace("_TMP_HTML", "")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-research", type=str, default=None,
                        help="Path to the dataframe with the collected resources")
    parser.add_argument("--topic", type=str, default=None, 
                        help="""Topic of the report. With the option 
                        --load-research, the topic is inferred from the path""")
    parser.add_argument("--topic-file", type=str, default=None,
                        help="""Path of the file with the information about the
                        topic of interest""")
    parser.add_argument('--figures-folder', type=str, default=None,
                        help="""Path of the figures folder, that may be 
                        associated to a different experiment as the research
                        file""")
    parser.add_argument("--load-report-draft", type=str, default=None,
                        help="Load a previous report version")
    parser.add_argument("--generate-report", action="store_true", help="""
                        Whether to generate the report""")
    parser.add_argument("--include-images", action="store_true", help="""
                        Whether to include images in the report""")
    parser.add_argument("--refine-report", action="store_true", help="""
                        Whether to apply post-gneration text refinement""")
    parser.add_argument("--do-citation-check", action="store_true",
                        help="Do a final citation check")
    parser.add_argument("--include-table", action="store_true", 
                        help="""Add a table as comparative analysis at the end
                        of the report""")
    parser.add_argument("--table-of-content", default=False, 
                        action=argparse.BooleanOptionalAction, help="""Include
                        the table of content at the beginning of the report""")
    parser.add_argument("--enrich-sections", type=str, default=None,
                        help="""List of sections to be enriched. It requires
                        loading a report draft. Indicate them as a string 
                        formatted as list, e.g. '[3.2, 3.3, 3.5]'""")
    
    parser.add_argument("--main-tool", type=str, default="trafilatura", 
                        help="Main parsing tool to use")
    parser.add_argument("--path-experiments", type=str, default="experiments",
                        help="Path where to save the experiments")
    parser.add_argument("--llm", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="Default LLM id")  
    parser.add_argument("--temperature", type=float, default=0.,
                        help="Default LLM temperature value")
    parser.add_argument("--num-queries", type=int, default=30,
                        help="Number of queries on the topic to generate")
    parser.add_argument("--num-top-results", type=int, default=3,
                        help="Number of results to keep from the SERP")
    parser.add_argument("--query-gen-type", type=str, default="generate",
                        choices=["template", "generate"], help="""Generation 
                        mode for the Google Search queries. They can either be
                        create from a predefined template, or generated with an
                        LLM.""")
    parser.add_argument("--length-upper-filter", type=float, default=30000,
                        help="""Filter on the upper length of a document. 
                        If less than 1, it is interpreted as a quantile,
                        otherwise as a static threshold on the number of tokens.""")
    parser.add_argument("--completeness-threshold", type=float, default=5,
                        help="Completeness score threshold")
    parser.add_argument("--search-threshold", type=float, default=5,
                        help="Extra search need score threshold")
    parser.add_argument("--link_relevance_threshold", type=int, default=7,
                        help="Link relevance score threshold")
    parser.add_argument("--max-new-links", type=int, default=40,
                        help="""Maximum number of URLs to parse after the 
                        relevance filtering""")
    parser.add_argument("--reasoning-llm", type=str, default="deepseek-ai/DeepSeek-R1",
                        help="Default reasoning LLM id")  
    parser.add_argument("--reasoning-temperature", type=float, default=0.,
                        help="Default reasoning LLM temperature value")  
    parser.add_argument("--report-gen-mode", type=str, default="outline+report",
                        choices=["outline", "outline+report", "report",
                                 "report_no_info"],help="Generation mode for the report") 
    parser.add_argument("--outline-gen-mode", type=str, default="2-stage",
                        choices=["1-stage", "2-stage"])
    parser.add_argument("--nb-workers", type=int, default=8, help="""Number of
                        workers to use for the parsing stage of the documents""")
    parser.add_argument("--topic-description", type=str, default="",
                        help="More extensive description of the topic")
    parser.add_argument("--urls", type=list, default=[], help="List of URLs")
    parser.add_argument("--sim-threshold", type=float, default=0.9,
                        help="""Threshold to be applied on the similarity value 
                        to perform deduplication""")
    parser.add_argument("--evaluate", action="store_true", help="""Do a first 
                        evaluation round after the report generation and prior
                        to the table of content positioning.""")
    parser.add_argument("--auto-correct", action="store_true", help="""Whether to 
                        apply autocorrection to the claims.""")

    args = parser.parse_args()

    if args.query_gen_type == "generate":
        args.query_gen_type = QueryType.GENERATE
    elif args.query_gen_type == "template":
        args.query_gen_type = QueryType.TEMPLATE
    else:
        raise ValueError(f"""Supported query generation types are 'template' 
                         and 'generate'. You inserted {args.query_gen_type}""")
    
    if args.report_gen_mode == "outline":
        args.report_gen_mode = ReportGenType.OUTLINE
    elif args.report_gen_mode == "outline+report":
        args.report_gen_mode = ReportGenType.OUTLINE_REPORT
    elif args.report_gen_mode == "report_no_info":
        args.report_gen_mode = ReportGenType.REPORT_NO_INFO
    elif args.report_gen_mode == "report":
        args.report_gen_mode = ReportGenType.REPORT
    else:
        raise ValueError(("Supported report generation types are 'outline', " 
                          "'outline+report', 'report', 'report_no_info'. You "
                          f"inserted '{args.report_gen_mode}'"))
    
    if args.outline_gen_mode == "1-stage":
        args.outline_gen_mode = OutlineGenMode.ONE_STAGE
    elif args.outline_gen_mode == "2-stage":
        args.outline_gen_mode = OutlineGenMode.TWO_STAGE
    else:
        raise ValueError(("Supported outline generation modes are '1-stage' "
                          f"and '2-stage'. You inserted {args.outline_gen_mode}"))
    
    LOAD_RESEARCH = args.load_research
    if LOAD_RESEARCH is None:
        TOPIC = args.topic
        if TOPIC is None:
            if args.topic_file is None:
                raise ValueError("Insert a topic or a topic file!")
            else:
                topic_definer = TopicDefiner(model_name=args.llm)
                topic_descr, TOPIC, given_urls = topic_definer.define_topic(args.topic_file)
        else:
            topic_descr = None
            given_urls = []
        topic_str_info = f"Inserted topic: {TOPIC}\nDescription: {topic_descr}"
    else:
        topic_info_path = (Path(LOAD_RESEARCH).parents[1] / "topic.json").absolute().as_posix()
        if os.path.isfile(topic_info_path):
            with open(topic_info_path) as fp:
                topic_info = json.load(fp)
                TOPIC = topic_info["topic"]
                topic_descr = topic_info["topic_description"]
                given_urls = topic_info["given_urls"]
        else:
            TOPIC = Path(LOAD_RESEARCH).parents[3].name
            topic_descr = None
            given_urls = []

        topic_str_info = f"Inferred topic: {TOPIC}"
    print(topic_str_info)
    print(len(TOPIC) * "-", "\n", TOPIC, "\n", len(TOPIC) * "-", "\n", sep="")
    args.topic_description = topic_descr
    args.urls = given_urls
    args.topic = TOPIC

    # Set the API keys
    set_api_keys()

    main(args)
