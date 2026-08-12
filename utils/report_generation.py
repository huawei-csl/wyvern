import copy
import html
import json
import logging
import os
import re
import time
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional

import backoff
import langchain_openai
import pandas as pd

from utils.eval import CitationsEvaluator, ClaimGranularityLevel
from utils.figures_extraction import FiguresMasker, extract_image_text, remove_duplicated_figures
from utils.general import (contains_markdown_table, convert_markdown_to_html,
                           convert_citations_to_superscript, encode_image,
                           extract_level_headings, extract_markdown_sections,
                           find_last_report_version, find_section_of_text_unit,
                           fix_citations, format_citations_to_superscript,
                           get_number_tokens, modify_latex_equations,
                           parse_llm_json, remove_surrounding_quotes, remove_urls,
                           remove_ai_generated_disclaimer,
                           write_generated_report)
from utils.prompting import PromptOracle, queryLLM, queryLLM_batch
from utils.scrape import url_to_filename
from utils.visualizations import plot_recall_stats

logger = logging.getLogger(__name__)


class OutlineGenMode(Enum):
    ONE_STAGE = auto()
    TWO_STAGE = auto()


class OutlineGenerator():
    def __init__(
            self,
            topic: str,
            information: str,
            model: langchain_openai.ChatOpenAI,
            description: str = ""
            ):
        """Outline generation object.
        
        Parameters
        ----------
        - topic [`str`]: topic
        - information [`str`]: concatenation of the references' text
        - model [`langchain_openai.ChatOpenAI`]: model for the generation
        - description [`str`, default = ""]: topic description
        """
        self.topic = topic
        self.information = information
        self.description = description
        self.model = model

    def generate(
            self,
            mode: OutlineGenMode,
            n_outlines: Optional[int] = None
            ) -> str:
        """Generate the outline for the report.
        
        Parameters
        ----------
        - mode [`OutlineGenMode`]: mode to generate the outline. 
        This can happen either in a single-stage or a two-stage format.
        - n_outlines [`Optional[int]`, default = None]: number of outlines 
        to be used for the two-stage generation

        Output
        ------
        - `str`: outline
        """
        
        if mode == OutlineGenMode.TWO_STAGE:            
            start_time = time.time()
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            outlines_list = queryLLM_batch(self.model, [PromptOracle.get_stage1_outline_prompt(self.topic, self.information, self.description)] * n_outlines)
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a") as ff:
                ff.write(f"(REPORTGEN) - A9: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            outlines_list = [outline.split("</think>")[1].strip() for outline in outlines_list if "</think>" in outline]
            logger.info(f"Initial outlines:\n{"\n".join(outlines_list)}")

            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            final_outline = queryLLM(self.model, PromptOracle.get_stage2_outline_prompt(self.topic, self.information, outlines_list, self.description))
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a") as ff:
                ff.write(f"(REPORTGEN) - A10: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            logger.info("Outline generation took {:.2f} seconds".format(time.time()-start_time))
            if "</think>" in final_outline:
                final_outline = final_outline.split("</think>")[1].strip()
            logger.info(f"Outline:\n{final_outline}")
            return final_outline
        
        elif mode == OutlineGenMode.ONE_STAGE:
            query = PromptOracle.get_outline_gen_prompt(self.topic, self.information, self.description)
            start_time = time.time()
            outline = queryLLM(self.model, query)
            if "</think>" in outline:
                outline = outline.split("</think>")[1].strip()
            logger.info("Outline generation took {:.2f} seconds".format(time.time()-start_time))
            logger.info(f"Outline:\n{outline}")
            return outline

        else:
            raise ValueError(f"Unknown outline generation mode!")


class ReportGenType(Enum):
    OUTLINE = auto()
    OUTLINE_REPORT = auto()
    REPORT = auto()
    REPORT_NO_INFO = auto()
    REFINE = auto()


class ReportGenerator(): 
    def __init__(self, 
                 llm: langchain_openai.ChatOpenAI,
                 reasoning_llm: Optional[langchain_openai.ChatOpenAI],
                 topic: str,
                 information: Optional[str] = None,
                 references_df: Optional[pd.DataFrame] = None,
                 topic_description: Optional[str] = None): 
        """Create a generator for the report.
        
        Parameters
        ----------
        - llm [`langchain_openai.ChatOpenAI`]: LLM to use for the generation
        - reasoning_llm [`langchain_openai.ChatOpenAI`]: reasoning model to be used
        for the generation
        - topic [`str`]: topic of the research
        - information [`str`]: collected data on the topic. Optional as report
        can be generated without it.
        - references_df [`pd.DataFrame`]: dataframe with all the references, 
        used to possibly generate the references section. Optional as the report
        can be generated without external information."""
        self.topic = topic
        self.information = information
        self.references_df = references_df
        self.llm = llm
        self.reasoning_llm = reasoning_llm
        self.topic_description = topic_description

    def generate(self,
                 mode: ReportGenType,
                 prev_report: Optional[str] = None,
                 outline_gen_mode: Optional[OutlineGenMode] = None,
                 outline_gen_num: Optional[int] = 3):
        
        # General check on the input
        if (mode != ReportGenType.REPORT_NO_INFO) and (
            (self.information is None) or (self.references_df is None)):
            raise ValueError(
                "No information or references available to generate the report"
                )
        
        # Generate the report according to the mode
        if mode == ReportGenType.OUTLINE:
            if prev_report is not None:
                raise ValueError("When in outline generation mode, the previous version of the report is not needed!")
            outline_generator = OutlineGenerator(
                topic=self.topic, 
                information=self.information,
                model=self.reasoning_llm,
                description=self.topic_description
                )
            return outline_generator.generate(outline_gen_mode, outline_gen_num)

        elif mode == ReportGenType.REPORT:
            if prev_report is not None:
                raise ValueError("When in report generation mode, the previous version of the report is not needed!")

            query = PromptOracle.get_report_prompt(self.topic, self.information)
            start_time = time.time()
            report = queryLLM(self.llm, query)
            logger.info("Report generation took {:.2f} seconds".format(time.time()-start_time))
            
            # Add the references section
            report = self.add_references_report(report)
            return report
        
        elif mode == ReportGenType.REPORT_NO_INFO:
            query = PromptOracle.get_report_prompt(self.topic)
            start_time = time.time()
            report = queryLLM(self.llm, query)
            logger.info("Report generation without collected information took {:.2f} seconds".format(
                time.time()-start_time))

            # Add the references section
            return report
        
        elif mode == ReportGenType.OUTLINE_REPORT:
            if prev_report is not None:
                raise ValueError("When in outline+report generation mode, the previous version of the report is not needed!")
            outline_generator = OutlineGenerator(
                topic=self.topic, 
                information=self.information,
                model=self.reasoning_llm,
                description=self.topic_description
                )
            outline = outline_generator.generate(outline_gen_mode, outline_gen_num)
            # Extract the 2nd level headings
            headings = extract_level_headings(outline, level=2)
            
            try:
                # Start the report with the title
                report = "# " + re.findall(r'^# (.+)$', outline, flags=re.MULTILINE)[0]
            except Exception as e:
                raise ValueError("Failed to extract the report title from the outline") from e

            prompts = []
            # Expand all the sections
            for heading in headings:
                prompt = PromptOracle.get_section_gen_prompt(self.topic, outline, self.information, heading)
                prompts.append(prompt)
            start_time = time.time()
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            sections = queryLLM_batch(self.llm, prompts)
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a") as ff:
                ff.write(f"(REPORTGEN) - A11: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            if sections[0].startswith("# "):
                # NOTE: the LLM sometimes changes the levels for the first section,
                # probably due to the numbering
                logger.info("Manually adjusting first section heading levels...")
                sections[0] = re.sub(r'^(#{1,6}) ', lambda match: '#' * (len(match.group(1)) + 1) + ' ', sections[0], flags=re.M)
            for section in sections:
                report += ("\n" + section + "\n")
            logger.info("Entire report generation took {:.2f} seconds".format(time.time()-start_time))
            # Change the format of equations
            report = modify_latex_equations(report)
            report = convert_citations_to_superscript(report)
            # Add the references section
            report = self.add_references_report(report)
            return report
        
        elif mode == ReportGenType.REFINE:
            if prev_report is None:
                raise ValueError("When in refinement generation mode, provide the previous version of the report!")
            refined_report = self._refine_report(prev_report)
            return refined_report
        else:
            raise ValueError("Unknown report mode generation.")

    @backoff.on_exception(
            backoff.constant,
            Exception,
            interval=0,
            max_tries=3,
        )
    def _refine_report(self, prev_report):
        # Hide the images
        figures_handler = FiguresMasker()
        report = figures_handler.replace_base64_with_links(prev_report)
        report = report.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>","")
        report = remove_urls(report)
        
        # Extract the 2nd level headings
        sections = extract_markdown_sections(report, exclude_references=False)
        condensed_sections = []
        sec_tmp = ""
        for sec in sections[1:]:
            if not(sec[0].startswith("## ")):
                sec_tmp += f"\n\n{sec[0]}\n\n{sec[1]}"
            else:
                if sec_tmp != "":
                    condensed_sections.append(sec_tmp.strip())
                sec_tmp = f"\n\n{sec[0]}\n\n{sec[1]}"
        # NOTE: the following line is commented to exclude the references
        # condensed_sections.append(sec_tmp.strip())
                
        prompts = [PromptOracle.get_refinement_prompt(self.topic, self.information, sec) \
                    for sec in condensed_sections]
        
        logger.info(f"Extracted {len(prompts)} sections...")
        start_time = time.time()
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        refined_sections = queryLLM_batch(self.llm, prompts)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A18: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        logger.info("Refining the report took {:.2f} seconds".format(time.time()-start_time))

        refined_report = sections[0][0]  # title
        for i, refined_section in enumerate(refined_sections):
            if "</think>" in refined_section:
                thinking = refined_section.split("</think>")[0]
                refined_sections[i] = refined_section.split("</think>")[1].strip()
            refined_report += f"\n\n{refined_sections[i].strip().strip('\'"')}\n\n"

        refined_report = modify_latex_equations(refined_report)
        refined_report = convert_citations_to_superscript(refined_report)
        refined_report = self.add_urls_to_citations(refined_report)
        refined_report += f"\n\n{sections[-1][0]}\n{sections[-1][1]}"  # references
        refined_report = figures_handler.reconvert_links_to_base64(refined_report)
        if ("source_placeholder_" in refined_report.split("## References")[0]) or \
        ("link_") in refined_report.split("## References")[0]:
            raise ValueError("Source or link placeholder detected")
        return refined_report

    def add_references_report(self,
                              report: str):
        """Given a report with citations and a resources `pandas.DataFrame`, prepare the references section.
        
        Parameters
        ----------
        - report [`str`]: markdown text of the report
        
        Output
        ------
        - `str`: original report with the references section"""

        # Find all the citations
        pattern = r'\[(\d+)([^\]]*)\]'
        matches = re.findall(pattern, report)
        integer_identifiers = [int(match[0]) for match in matches]
        
        # Include only the references which have been actually cited
        references_section = """\n\n\n## References\n"""
        for match in sorted(list(set(integer_identifiers))):
            try:
                ref_url = self.references_df.iloc[match - 1].url
                references_section += f"\n[{match}, {self.references_df.iloc[match - 1].identifier}] [{ref_url}]({ref_url})\n"
                report = report.replace(f"[{match}, {self.references_df.iloc[match - 1].identifier}]", f"[[{match}, {self.references_df.iloc[match - 1].identifier}]({ref_url})]")
            except IndexError as e:
                logger.info(f"[ERROR] {e}")
                logger.info(f"Match: {match}, leading to index {match - 1}")
                logger.info(f"Length of references_df: {len(self.references_df)}")
                logger.info(f"Indexes: {str(list(self.references_df.index.values))}")
                raise

        report += references_section
        return report
    
    @backoff.on_exception(
            backoff.constant,
            Exception,
            interval=0,
            max_tries=3,
        )    
    def check_citations(self, 
                        report: str,
                        information: str,
                        log_folder: str):
        if report is None:
            raise ValueError("When applying citations check, provide the report!")
        # Hide the images
        figures_handler = FiguresMasker()
        report_no_links = figures_handler.replace_base64_with_links(
            report)
        
        # Extract the 2nd level headings
        sections = extract_markdown_sections(report_no_links, exclude_references=False)
        condensed_sections = []
        sec_tmp = ""
        for sec in sections[1:]:
            if not(sec[0].startswith("## ")):
                sec_tmp += f"\n\n{sec[0]}\n\n{sec[1]}"
            else:
                if sec_tmp != "":
                    condensed_sections.append(sec_tmp.strip())
                sec_tmp = f"\n\n{sec[0]}\n\n{sec[1]}"
        # NOTE: the following line is commented to exclude the references
        # condensed_sections.append(sec_tmp.strip())
                
        prompts = [PromptOracle.get_citations_check_prompt(self.topic, sec, information) \
                    for sec in condensed_sections]
        logger.info(f"Extracted {len(prompts)} sections...")
        start_time = time.time()
        refined_sections = queryLLM_batch(self.reasoning_llm, prompts)
        logger.info("It took {:.2f} seconds".format(time.time()-start_time))

        refined_report = sections[0][0]  # title
        for i, refined_section in enumerate(refined_sections):
            if "</think>" in refined_section:
                thinking = refined_section.split("</think>")[0]
                refined_sections[i] = refined_section.split("</think>")[1].strip()
                with open(os.path.join(log_folder, "reasoning_citation_check.txt"), "a") as f:
                    f.write(f"{refined_sections[i].split('\n')[0]}\n{thinking}\n\n\n")
            refined_report += f"\n\n{refined_sections[i].strip().strip('\'"')}\n\n"
        
        refined_report = self.add_urls_to_citations(refined_report)
        
        refined_report += f"\n\n{sections[-1][0]}\n{sections[-1][1]}"  # references
        if ("source_placeholder_" in refined_report.split("## References")[0]) or \
        ("link_") in refined_report.split("## References")[0]:
            raise ValueError("Source or link placeholder detected")
        return figures_handler.reconvert_links_to_base64(refined_report)
    

    def add_urls_to_citations(self, text: str) -> str:
        """
        Add the given URL to citations that do not already have an associated URL.
        
        Parameters
        ----------
        - text [`str`]: The input markdown text.
        - url [`str`]: The URL to associate with citations that lack one.
        
        Output
        ------
        - `str`: The updated markdown text with URLs added to citations that were missing them.
        """
        
        # Match citations without URLs 
        citation_pattern_no_url = r'\[(\d+),\s*([^\]]+)\](?!\()'

        def add_url_to_citation(match):
            try:
                citation_number = match.group(1)
                citation_text = match.group(2)
                url = self.references_df.iloc[int(citation_number) - 1].url
            except Exception as e:
                raise ValueError(
                    f"Failed to add URL to citation [{citation_number}, {citation_text}]"
                ) from e
            return f'[[{citation_number}, {citation_text}]({url})]'
        
        updated_text = re.sub(citation_pattern_no_url, add_url_to_citation, text)        
        return updated_text
    
    @backoff.on_exception(
            backoff.constant,
            Exception,
            interval=0,
            max_tries=3,
        )
    def include_images(self,
                       figures_path: str,
                       report_text: str,
                       log_path: Optional[str] = None):
        if log_path is None:
            log_path = figures_path
        logger.info(f"Figures path: {figures_path}")
        
        # Extract the outline and the sections of the report text
        sections = extract_markdown_sections(report_text, exclude_references=False)
        references_section = sections[-1]
        sections = sections[:-1]
        outline = "\n".join([title for title, text in sections])

        # --------------------------------------------
        # Load the documents and extract their figures
        # --------------------------------------------
        # Now I can iterate over all the markdown files in the figures folder and 
        # extract the figures for each of them. All the figures need to be passed as
        # input to a model for the generation of their description
        markdown_files = [
            os.path.join(figures_path, path)
            for path in os.listdir(figures_path)
            if os.path.isfile(os.path.join(figures_path, path))
        ]

        # Check whether the files from which the images will be extracted are
        # included in the final references pool used to generate the report,
        # to cut out sources already deemed as unrelevant
        ind_to_keep = []
        url_list = []
        summaries_list = []
        identifiers_list = []
        full_identifiers_list = []
        for ind, file in enumerate(markdown_files):
            FLAG = False
            for url in self.references_df.url.values:
                if file.split("/")[-1].split("-with")[0] in url_to_filename(url):
                    logger.info(f"Match found: {file.split("/")[-1]} - {url}")
                    FLAG = True
                    ind_to_keep.append(ind)
                    url_list.append(url)
                    summaries_list.append(self.references_df[self.references_df.url == url].summary.values[0])
                    identifiers_list.append(self.references_df[self.references_df.url == url].identifier.values[0])
                    full_identifiers_list.append(f"{self.references_df[self.references_df.url == url].index.values[0] + 1}, {self.references_df[self.references_df.url == url].identifier.values[0]}")

                    break
            if FLAG is False:
                logger.info(f"No match for file {file.split("/")[-1]}")
        markdown_files = [file for i, file in enumerate(markdown_files) if i in ind_to_keep]


        all_images_dict = []
        for doc_path, source_url, summary, identifier, full_identifier in zip(markdown_files, url_list, summaries_list, identifiers_list, full_identifiers_list):
            with open(doc_path) as f:
                doc = f.read()

            # Extract the figures from the document text in markdown
            doc_figures_list = extract_image_text(doc)

            # Once the images are extracted, generate a description for each figure
            # given the image itself, its caption, and snippets of text that refer
            # to it. Store the description in the image dictionary object.
            descriptions = []
            prompts = []

            title = next((re.sub(r'^#+\s', '', line) for line in doc.split("\n") \
                          if line.startswith("#")), "")

            for image_dict in doc_figures_list:
                # Include the absolute path in the information dict
                image_dict["image_abs_path"] = os.path.join(figures_path, 
                                                            image_dict["image_path"])
                image_dict["source_url"] = source_url
                image_dict["page_title"] = f"{title} (from source: {identifier})"
                image_dict["summary"] = summary
                image_dict["full_identifier"] = full_identifier
                prompts.append(PromptOracle.get_description_gen_prompt(image_dict, self.topic))
            
            from utils.prompting import (total_input_tokens,
                                            total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            descriptions = queryLLM_batch(self.llm, prompts)
            from utils.prompting import (total_input_tokens,
                                            total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a") as ff:
                ff.write(f"(REPORTGEN) - A12: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            for i, description in enumerate(descriptions):
                if "</think>" in description:
                    descriptions[i] = description.split("</think>")[1].strip()
            for i, image_dict in enumerate(doc_figures_list):
                image_dict["description"] = descriptions[i]
            all_images_dict.extend(doc_figures_list)

        logger.info(f"Description generated for {len(all_images_dict)} images")
        # Save descriptions for check
        descriptions_folder = os.path.join(log_path, "descriptions")
        os.makedirs(descriptions_folder, exist_ok=True)
        with open(os.path.join(descriptions_folder, "images_info.json"), "w") as f:
            json.dump(all_images_dict, f)

        with open(os.path.join(descriptions_folder, "images_info_expanded.md"), "w") as f:
            for i, item in enumerate(all_images_dict):
                f.write(f"# FIGURE {i} (in doc: {item["figure_number"]})")
                f.write(f"\n\nSource: {item['source_url']}")
                f.write(f"\n\nPath: {item['image_abs_path']}")
                f.write(f"\n\nTitle: {item["page_title"]}")
                f.write(f"\n\n![Image](data:image/png;base64,{encode_image(item['image_abs_path'])})")
                f.write(f"\n\nDescription: {item['description']}")
                f.write(f"\n\nCaption: {item['caption']}")
                f.write(f"\n\nText snippets: {str(item['text_snippets'])}\n\n\n\n\n\n")

        # --------------------------------------------
        # Figures selection
        # --------------------------------------------
        prompt = PromptOracle.get_figures_selection_prompt(
            descriptions=[item["description"] for item in all_images_dict], 
            outline=outline,
            topic=self.topic)
    
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens

        # Check whether the length of the prompt with all the images description is 
        # not too high. Otherwise do a hierarchical split of the figures selection.
        max_figures_selection_tokens = 45000
        prompt_tokens = get_number_tokens(prompt, self.reasoning_llm.model_name)

        def _extract_figures_selection(response):
            if "</think>" in response:
                response = response.split("</think>")[-1]
            pattern = r"(```json\s*(.*?)\s*```|<JSON>(.*?)</JSON>|(\{[\s\S]*\}))"
            matches = re.findall(pattern, response, re.DOTALL)
            if len(matches) == 0:
                raise ValueError(("No json has been found in the response of the LLM!\n\n"
                                f"Response\n--------\n{response}"))
            for match in matches:
                candidate = match[1] or match[2] or match[3]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    logger.debug("Skipping invalid JSON candidate", exc_info=True)
                    continue
            raise ValueError(("No valid json has been found in the response of the LLM!\n\n"
                              f"Response\n--------\n{response}"))

        def _selected_indices(json_result):
            return [
                int(list(im.keys())[0].lower().replace("figure ", ""))
                for im in json_result["critical_figures"]
            ]

        def _figure_prompt(indices):
            return PromptOracle.get_figures_selection_prompt(
                descriptions=[all_images_dict[i]["description"] for i in indices],
                outline=outline,
                topic=self.topic)

        def _split_indices(indices):
            n_groups = max(2, (get_number_tokens(_figure_prompt(indices), self.reasoning_llm.model_name)
                               + max_figures_selection_tokens - 1) // max_figures_selection_tokens)
            while n_groups <= len(indices):
                groups = [
                    indices[i * len(indices) // n_groups:(i + 1) * len(indices) // n_groups]
                    for i in range(n_groups)
                ]
                prompts = [_figure_prompt(group) for group in groups]
                if all(get_number_tokens(group_prompt, self.reasoning_llm.model_name)
                       < max_figures_selection_tokens for group_prompt in prompts):
                    return groups, prompts
                n_groups += 1
            raise ValueError((f"Too many figures provided ({len(all_images_dict)}), "
                              "or at least one figure description is too long for "
                              "hierarchical figure selection."))

        # ******
        # CASE 1
        # ******
        if prompt_tokens < max_figures_selection_tokens:
            logger.info((
                f"Prompt for figures selection has less than {max_figures_selection_tokens} tokens "
                f"({prompt_tokens})"
                f", thus doing one single evaluation round with {self.reasoning_llm.model_name}"))
            start_time = time.time()
            response = queryLLM(self.reasoning_llm, prompt)
            logger.info("Selecting the images took {:.2f} seconds".format(
                time.time() - start_time))
            with open(os.path.join(descriptions_folder, 
                                   "reasoning_figures_selection.txt"), "w") as f:
                f.write(response)

            json_result = _extract_figures_selection(response)

        # ******
        # CASE 2
        # ******
        else:
            candidate_indices = list(range(len(all_images_dict)))
            round_number = 1
            with open(os.path.join(descriptions_folder, 
                                   "reasoning_figures_selection.txt"), "w") as f:
                while get_number_tokens(_figure_prompt(candidate_indices), self.reasoning_llm.model_name) >= max_figures_selection_tokens:
                    groups, group_prompts = _split_indices(candidate_indices)
                    start_time = time.time()
                    responses = queryLLM_batch(self.reasoning_llm, group_prompts)
                    logger.info(
                        "Selecting the images (round {}) took {:.2f} seconds".format(
                            round_number, time.time() - start_time))

                    next_candidate_indices = []
                    for group_i, (group, response) in enumerate(zip(groups, responses)):
                        f.write(f"ROUND {round_number} - GROUP {group_i + 1}\n-----------")
                        f.write(response)
                        f.write("\n\n\n\n")
                        json_group = _extract_figures_selection(response)
                        next_candidate_indices.extend(
                            group[i] for i in _selected_indices(json_group))

                    if len(next_candidate_indices) >= len(candidate_indices):
                        raise ValueError(("Hierarchical figure selection did not reduce "
                                          "the number of candidate figures."))
                    candidate_indices = next_candidate_indices
                    round_number += 1

            prompt = _figure_prompt(candidate_indices)
            start_time = time.time()
            response = queryLLM(self.reasoning_llm, prompt)
            logger.info(
                "Selecting the images (final round) took {:.2f} seconds".format(
                    time.time() - start_time))
            
            with open(os.path.join(descriptions_folder, 
                                   "reasoning_figures_selection.txt"), "a") as f:
                f.write("\n\n\n\nFINAL\n-----------")
                f.write(response)

            json_result = _extract_figures_selection(response)
            
            for i, item in enumerate(json_result["critical_figures"]):
                ind_selected = _selected_indices({"critical_figures": [item]})[0]
                json_result["critical_figures"][i] = {f"FIGURE {candidate_indices[ind_selected]}": list(item.values())[0]
            }

        # Remove duplicates, if any        
        json_result["critical_figures"] = remove_duplicated_figures(json_result["critical_figures"])

        with open(os.path.join(descriptions_folder, "selected_images_info.json"), "w") as f:
            json.dump(json_result, f)
        
        selected_ind = []
        for item in json_result["critical_figures"]:
            key = list(item.keys())[0]
            try:
                selected_ind.append(int(key.split(" ")[1]))
            except:
                # Sometimes the model uses just the integer id
                selected_ind.append(int(key))

        selected_images_dict = [el for i, el in enumerate(all_images_dict) if i in selected_ind]        
        with open(os.path.join(descriptions_folder, "selected_images_info.md"), "w") as f:
            for item in selected_images_dict:
                f.write(f"# FIGURE {item["figure_number"]}")
                f.write(f"\n\nSource: {item['source_url']}")
                f.write(f"\n\nPath: {item['image_abs_path']}")
                f.write(f"\n\nTitle: {item["page_title"]}")
                f.write(f"\n\n![Image](data:image/png;base64,{encode_image(item['image_abs_path'])})")
                f.write(f"\n\nDescription: {item['description']}")
                f.write(f"\n\nCaption: {item['caption']}")
                f.write(f"\n\nText snippets: {str(item['text_snippets'])}\n\n\n\n\n\n")
        
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A13: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        # --------------------------------------------
        # Figures insertion in the report
        # --------------------------------------------
        try:
            fig_sections_dict = {}
            for figure in json_result["critical_figures"]:
                figure_number, section_id = [
                    (int((re.findall(r'\d+', k))[0]),v)
                    for k,v in figure.items()
                    ][0]
                figure_info = all_images_dict[figure_number]
                if section_id in fig_sections_dict:
                    fig_sections_dict[section_id].append(figure_info)
                else:
                    fig_sections_dict[section_id] = [figure_info]

            global_fig_identifier = 0
            for key in fig_sections_dict.keys():
                for inner_dict in fig_sections_dict[key]:
                    inner_dict["figure_identifier"] = global_fig_identifier
                    inner_dict["placeholder_path"] = f"path_fig_{global_fig_identifier}"
                    global_fig_identifier += 1
        except Exception as e:
            raise ValueError("Failed to assign figure identifiers and placeholders") from e

        prompts = []
        new_sections = copy.deepcopy(sections)
        for section in fig_sections_dict.keys():
            for ind, s in enumerate(sections):
                if section in s[0]:
                    section_text = s[1]
                    prompts.append(PromptOracle.get_figures_insertion_prompt(
                        fig_sections_dict[section],
                        section_text,
                        self.topic
                        ))
        with open(os.path.join(descriptions_folder, "prompts_figures_insertion.txt"), "w") as f:
            for prompt in prompts:
                f.write(prompt)
                f.write("\n\n\n\n\n")            

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens    
        start_time = time.time()
        responses = queryLLM_batch(self.reasoning_llm,
                                   prompts)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A14: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        logger.info("Figure insertion completed in {:.2f} seconds".format(
            time.time() - start_time
        ))
        id_responses = 0
        for section in fig_sections_dict.keys():
            for ind, s in enumerate(sections):
                if section in s[0]:
                    section_text = s[1]
                    with open(os.path.join(
                        descriptions_folder,
                        f"reasoning_sec_{section.replace("/", "_")}_{datetime.today():%Y-%m-%d_%H-%M-%S}.md"
                        ),"w") as f:
                        f.write(responses[id_responses])

                    new_sections[ind] = (s[0], responses[id_responses])
                    id_responses += 1

        # Clean the new sections text and update the paths
        for i, elem in enumerate(new_sections):
            sec_text = elem[1]

            if "</think>" in sec_text:
                sec_text = sec_text.split("</think>")[1].replace(
                    "```markdown", "").replace("```", "")
                
            if "![Image]" in sec_text:
                pattern = r'!\[Image\]\s*\((.*?)\)\s*'
                
                if not re.search(pattern, sec_text):
                    logger.info(f"No image path found: {sec_text}") 
                else:
                    try:
                        key = ""
                        for k in fig_sections_dict.keys():
                            if k in re.sub(r'^#+\s', '', elem[0]):
                                key = k
                                break
                        
                        matches = re.findall(pattern, sec_text)
                        for match in matches:
                            ref_element = next(
                                (el for el in fig_sections_dict[key] if el["placeholder_path"] == match), None
                            )
                            
                            if ref_element:
                                encoded_image = f"data:image/png;base64,{encode_image(ref_element['image_abs_path'])}"
                                source_link = f"<small>Source: [[{ref_element['full_identifier']}]({ref_element['source_url']})]</small>"
                                sec_text = sec_text.replace(f"{match})", f"{encoded_image})\n{source_link}\n\n")

                                if ref_element['full_identifier'] not in references_section[1]:
                                    references_section = (references_section[0], 
                                                          references_section[1] + f"\n\n[{ref_element['full_identifier']}] [{ref_element['source_url']}]({ref_element['source_url']})")

                    except IndexError as e:
                        logger.info(f"[Index ERROR] {e}")
                        logger.info(sec_text)
                        raise
                    except KeyError as e:
                        logger.info(f"[Key ERROR] {e}")
                        logger.info(f"Key: {key}\nKeys:{list(fig_sections_dict.keys())}")
                        raise

            new_sections[i] = (elem[0], sec_text)

        new_report = ""
        for elem in new_sections:
            new_report += f"\n\n{elem[0]}\n{elem[1]}\n"
        new_report += f"\n\n{references_section[0]}\n{references_section[1]}"

        num_handler = NumberingHandler()
        new_report = num_handler.adjust_figures_identifiers(new_report)
        if ("source_placeholder_" in new_report.split("## References")[0]) or \
        ("link_") in new_report.split("## References")[0]:
            raise ValueError("Source or link placeholder detected")
        return new_report            
        
    @backoff.on_exception(
            backoff.constant,
            Exception,
            interval=0,
            max_tries=3,
        )
    def include_table(self, 
                      report: str):
        ref_masker = ReferencesMasker()
        report = ref_masker.hide_references(report)
        fig_masker = FiguresMasker()
        report = fig_masker.replace_base64_with_links(report)
    
        # Build the table from the collected information
        logger.info((
            f"Using {self.reasoning_llm.model_name} for table building "
            f"with temperature {self.reasoning_llm.temperature}"))                
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens              
        response = queryLLM(self.reasoning_llm, 
                            PromptOracle.get_build_table_prompt(
                                self.topic, 
                                self.information,
                                self.topic_description))
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A15: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        if "</think>" in response:
            response = response.split("</think>")[1]
        table = response.split("# TABLE")[1].split("# EXPLANATION")[0]
        table = fix_citations(table, self.references_df)
        explanation = response.split("# EXPLANATION")[1]
        explanation = fix_citations(explanation, self.references_df)
        logger.info(f"Generated table:\n{table}")
        logger.info(f"Generated explanations:\n{explanation}")

        # Assign the table to a section of the report
        logger.info((
            f"Assigning the section to the table with {self.llm.model_name} "
            f"with temperature {self.llm.temperature}")) 
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        assigned_section = queryLLM(self.llm, 
                                    PromptOracle.get_table_placement_prompt(
                                        report, 
                                        table))
        logger.info(f"Table assigned to section {assigned_section}")
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A16: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        try:
            headings = extract_level_headings(report, 2)
            if assigned_section in headings:
                ind = headings.index(assigned_section)
            else:
                tmp_headings = [h.replace("#", "").strip() for h in headings]
                ind = tmp_headings.index(assigned_section)

            if ind == (len(headings) - 1):
                section = f"{assigned_section}" + report.split(headings[ind])[1]
            else:
                section = f"{assigned_section}" + report.split(headings[ind])[1].split(headings[ind+1])[0]
        except IndexError as e:
            raise ValueError(
                f"Failed to extract assigned table section: {assigned_section}"
            ) from e
        # Insert the table within the assigned section
        logger.info((
            f"Placing the table with {self.llm.model_name} "
            f"with temperature {self.llm.temperature}")) 
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        new_section = queryLLM(self.llm, 
                               PromptOracle.get_insert_table_prompt(
                                   self.topic,
                                   table,
                                   explanation,
                                   section))
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(REPORTGEN) - A17: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        if "</think>" in new_section:
            new_section = "\n\n" + new_section.split("</think>")[1] + "\n\n"
        
        # In the table there may be citations not already present in the 
        # references.
        pattern = r'\[(\d+),\s*([^\]]+?)\]'
        matches_sec = re.findall(pattern, new_section)
        numbers_sec = set([int(num) for num, _ in matches_sec])

        references_sec = ref_masker.get_references()
        matches_ref = re.findall(pattern, references_sec)
        numbers_ref = set([int(num) for num, _ in matches_ref])

        for number in numbers_sec:
            if number not in numbers_ref:
                logger.info(("After table insertion, adding reference "
                             f"{number} to the references list"))
                references_sec += f"\n\n[{number}, {self.references_df.iloc[number - 1].identifier}] [{self.references_df.iloc[number - 1].url}]({self.references_df.iloc[number - 1].url})"

        ref_masker.update_references(references_sec)

        new_report = report.replace(section, new_section + "\n\n")
        new_report = new_report.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>","")
        new_report = remove_urls(new_report)
        new_report = self.add_urls_to_citations(new_report)
        new_report = format_citations_to_superscript(new_report)
        new_report = ref_masker.unhide_references(new_report)
        new_report = fig_masker.reconvert_links_to_base64(new_report)
        if ("source_placeholder_" in new_report.split("## References")[0]) or \
        ("link_") in new_report.split("## References")[0]:
            raise ValueError("Source or link placeholder detected")
        return new_report
    
    def include_toc(self, report: str, type: str = "markdown"):
        report = format_citations_to_superscript(report)
        toc_handler = TableOfContentHandler()
        report_toc = toc_handler.insert_toc(report, type)
        return report_toc       
    
    def enrich_sections(self,
                        report: str, 
                        sections_to_refine: List[str]):
        start_time = time.time()
        fig_masker = FiguresMasker()
        report = fig_masker.replace_base64_with_links(report)
        ref_masker = ReferencesMasker()
        report = ref_masker.hide_references(report)
        report = report.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>","")
        report = remove_urls(report)
        all_sections = extract_markdown_sections(report, 
                                                 exclude_references=False)

        texts = []
        for sec in sections_to_refine:
            for sec_all in all_sections:
                if " " + sec + " " in sec_all[0]:
                    texts.append(sec_all[0] + "\n\n" + sec_all[1])
                    break
        prompts = []
        for t in texts:
            prompts.append(PromptOracle.get_enrich_section_prompt(
                self.topic,
                t,
                self.information))

        responses = queryLLM_batch(self.llm, prompts)

        new_report = report
        for i, resp in enumerate(responses):
            new_report = new_report.replace(texts[i], responses[i])
        new_report = new_report.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>","")
        new_report = remove_urls(new_report)
        new_report = fig_masker.reconvert_links_to_base64(new_report)
                
        new_report = self.add_urls_to_citations(new_report)
        new_report = format_citations_to_superscript(new_report)
        new_report = ref_masker.unhide_references(new_report)
        new_report = modify_latex_equations(new_report)
        logger.info("Sections enrichment completed in {:.2f} seconds".format(
            time.time() - start_time
        ))
        return new_report
    
    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=0,
        max_tries=1,
        )
    def evaluate(self, report, eval_folder="eval"):
        logger.info(f"Saving the evaluation to {eval_folder}")
        os.makedirs(eval_folder, exist_ok=True)
        # The disclaimer is report metadata, not a factual claim to evaluate.
        report = remove_ai_generated_disclaimer(report)
        fig_masker = FiguresMasker()
        report = fig_masker.replace_base64_with_links(report)
        toc_handler = TableOfContentHandler()
        report = toc_handler.hide_toc(report)
        report = report.replace("<sup><small>", "").replace("</sup></small>", "").replace("<small><sup>", "").replace("</small></sup>", "")
        report = remove_urls(report)
        report = re.sub(r"Source:\s*\[\[.*?\]\(.*?\)\]", "", report)
        report = re.sub(r"!\[Image\]\(.*?\)", "", report)
        # Consider only the section texts, remove the headers, the empty lines, 
        # and the references
        report_paragraphs = []
        for line in report.split('\n'):
            if line.startswith("## References"):
                break
            if len(line) == 0 or line[0] == '#':
                continue
            report_paragraphs.append(line)
        # Create a new version of the report with only the section paragraphs by
        # concatenating the kept lines
        report_text = '\n'.join(report_paragraphs).strip()
        llm_nli = self.reasoning_llm
        llm_claims = self.llm

        evaluator = CitationsEvaluator(llm_nli, llm_claims)

        cit_rec, entail, sent_no_citations, sent_ref_not_found, sent_failed_support, \
            recall_df, prec_df = evaluator.evaluate_atomic_claims(report_text, 
                                                                  self.references_df, 
                                                                  ClaimGranularityLevel.PARAGRAPH,
                                                                  load_claims=None,
                                                                  eval_folder=eval_folder)
        logger.info("Citation recall {:.2f}%".format(cit_rec))
        plot_recall_stats(supported=entail,
                          no_cit=sent_no_citations,
                          ref_not_found=sent_ref_not_found,
                          unsupported_claim=sent_failed_support,
                          plot_folder=os.path.join(eval_folder, "plots"),
                          plot_name=f"recall_stats_{datetime.today():%Y-%m-%d_%H-%M-%S}.png")
        return recall_df
    
    @backoff.on_exception(
        backoff.constant,
        Exception,
        interval=0,
        max_tries=1,
        )
    def auto_correct_claims(self, report, claims, correction_folder):
        logger.info(f"Saving the correction to {correction_folder}")
        os.makedirs(correction_folder, exist_ok=True)
        ref_masker = ReferencesMasker()
        report = ref_masker.hide_references(report)
        fig_masker = FiguresMasker()
        report = fig_masker.replace_base64_with_links(report)
        
        claims_wrong = claims[claims.recall==0]
        claims_wrong = claims_wrong[~claims_wrong.recall_explained.isna()]

        # Associate to the claims the sections of the report from which they have 
        # been extracted
        sections = []
        for i, row in claims_wrong.iterrows():        
            sections.append(find_section_of_text_unit(row.text_unit,
                                                    remove_urls(report.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>",""))))
            if sections[-1] is None:
                raise ValueError("""No matching section has been retrieved. Check 
                                the find_section_of_text_unit function.""")
        claims_wrong["section"] = sections

        # Consider each text_unit separately and modify the claims
        grouped = claims_wrong.groupby("text_unit_id")
        logger.info(f"There are {len(claims_wrong)} to be revised, that affect {len(grouped)} text units")
        prompts = []
        text_units = []
        concat_explanations = []
        for text_unit_id, group in grouped:
            text_unit = group.text_unit.values[0]
            text_units.append(group.text_unit.values[0])
            refs = group.joint_ref_passages.values[0]
            section = group.section.values[0]
            claims = group.claim.values.tolist()
            explanations = [l.split("</think>")[1] if "</think>" in l else "" for l in group.recall_explained.values]

            tmp_e = ""
            for c, e in zip(claims, explanations):
                if e == "":
                    logger.warning(f"Skipping claim {c} for wrong explanation format: {e}")
                    continue
                tmp_e += f'''Claim: {c.strip()}\nExplanation: {e.strip()}\n\n'''
            
            concat_explanations.append(tmp_e.strip())
            
            prompts.append(PromptOracle.get_correct_claim_prompt(
                text_unit,
                section,
                claims,
                explanations,
                refs))
        # Modify the text units in parallel
        start_time = time.time()
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        responses = queryLLM_batch(self.llm, prompts)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(GROUNDING) - A21: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        logger.info("Revising the claims took {:.2f} seconds".format(
            time.time() - start_time
        ))
        report = fig_masker.reconvert_links_to_base64(report)

        # Debug version
        report_copy = copy.deepcopy(report)
        report_copy = remove_urls(report_copy).replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>","")
        
        for text_unit, explanation,response in zip(text_units, concat_explanations, responses):
            replacement_text_debug = f'<span title="{html.escape(explanation)}" style="color:#d00000"><s>{format_citations_to_superscript(text_unit)}</s></span><span style="background-color:#F6F5AE">{format_citations_to_superscript(response)}</span>'
            report_copy = report_copy.replace(text_unit, 
                                            replacement_text_debug)

        report_file_path_debug = os.path.join(correction_folder, f"report_v{find_last_report_version(correction_folder) + 1}_corrected_claims_debug.md")
        report_copy = format_citations_to_superscript(self.add_urls_to_citations(report_copy))
        report_copy = ref_masker.unhide_references(report_copy)
        write_generated_report(report_file_path_debug, report_copy)
        convert_markdown_to_html(
            report_file_path_debug,
            os.path.splitext(report_file_path_debug)[0] + ".html",
            self_contained=True,
        )

        # Polished version
        report_copy = copy.deepcopy(report)
        report_copy = remove_urls(report_copy.replace("<sup>", "").replace("</sup>","").replace("<small>", "").replace("</small>",""))
        for text_unit, response in zip(text_units, responses):
            if contains_markdown_table(response):
                if text_unit.split("|\n")[-1] not in report_copy:
                    raise ValueError("Table text unit not found in report copy")
                report_copy = report_copy.replace(text_unit.split("|\n")[-1], 
                                                response.split("|\n")[-1])
            else:
                if text_unit not in report_copy:
                    raise ValueError("Text unit not found in report copy")
                report_copy = report_copy.replace(text_unit, 
                                                  remove_surrounding_quotes(response))

        report_file_path_debug = os.path.join(correction_folder,
                                              f"report_v{find_last_report_version(correction_folder) + 1}_corrected_claims_polished.md")
        report_copy = format_citations_to_superscript(self.add_urls_to_citations(report_copy))
        
        # Revise the headings
        start_time = time.time()
        report_copy = self.revise_headings(report_copy)
        logger.info("Revising the headings took {:.2f} seconds".format(
            time.time() - start_time
        ))

        report_copy = ref_masker.unhide_references(report_copy)
        write_generated_report(report_file_path_debug, report_copy)
        convert_markdown_to_html(
            report_file_path_debug,
            os.path.splitext(report_file_path_debug)[0] + ".html",
            self_contained=True,
        )

        return report_copy
    

    @backoff.on_exception(
            backoff.constant,
            Exception,
            interval=0,
            max_tries=3,
            )
    def revise_headings(self, report: str) -> str:
        """Revise the headings of the report to better reflect the sections' content.

        Parameters
        ----------
        - report [`str`]: markdown text

        Output
        ------
        - `str`: markdown text with updated headings
        """
        figures_handler = FiguresMasker()
        report_no_links = figures_handler.replace_base64_with_links(report)
        report_no_links = report_no_links.replace("<small><sup>", "").replace("</small></sup>","").replace("<sup><small>", "").replace("</sup></small>","")
        report_no_links = remove_urls(report_no_links)

        title = extract_level_headings(report_no_links, 1)[0]
        headings = extract_level_headings(report_no_links, 2)

        prompts= []
        # Exclude the references section, which is always last
        for i, heading in enumerate(headings[:-1]):  
            section_text = report_no_links.split(heading)[1].split(headings[i+1])[0].strip()
            prompt = PromptOracle.get_headings_revision_prompt(title, heading, section_text)
            prompts.append(prompt)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        responses = queryLLM_batch(self.llm, prompts)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(GROUNDING) - A22: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        for response in responses:
            logger.info(response)
            dict_headings = parse_llm_json(response)
            for key in dict_headings.keys():
                report = report.replace(key, dict_headings[key])

        return report



class ReferencesMasker():
    def __init__(self):
        """Mask the references and store a copy of them inside the object, 
        allowing eventually the unmasking.
        """
        self.references = ""

    def hide_references(self, report):
        self.references = "\n\n## References\n\n" + report.split("## References")[1]
        report = report.split("## References")[0].strip()
        return report
    
    def unhide_references(self, report):
        report += self.references
        return report
    
    def get_references(self):
        return self.references
    
    def update_references(self, references):
        self.references = references


class TableOfContentHandler():
    def __init__(self):
        """Mask the table of content, and re-insert it.
        """

    def hide_toc(self, text: str) -> str:
        """Remove the '## Outline' section (including its content) from the markdown text.
        """
        # Matches from '## Outline' to before the next '## ' or EOF
        pattern = r'\n## Outline\b.*?(?=\n## |\Z)'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text.strip()
    
    def insert_toc(self, text: str, type: str = "markdown") -> str:
        """Insert the '## Outline' section
        """
        sections = extract_markdown_sections(text)
        outline = ""
        for section in sections[1:]:
            level = section[0].count("#")
            if type == "markdown":
                anchor = self._heading_to_anchor_md(section[0])
            elif type == "html":
                anchor = self._heading_to_anchor_html(section[0])
            else:
                raise ValueError((f"Wrong type provided ({type}). "
                                 "Only 'markdown' and 'html' are supported."))

            outline += f"{'    ' * (level-2)}- {anchor}\n"

        outline = "## Outline\n\n<details>\n<summary>Expand/Collapse Outline</summary>\n\n" + outline + "\n</details>\n"

        return text.replace(sections[0][0], f"{sections[0][0]}\n\n{outline}")

    def _heading_to_anchor_md(self, heading: str) -> str:
        # Remove headings levels
        heading_text = re.sub(r'^#+\s*', '', heading)
        anchor = heading_text.lower()
        # Remove special characters (except hyphens and spaces)
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = anchor.replace(' ', '-')
        return f"[{heading_text}](#{anchor})"
    
    def _heading_to_anchor_html(self, heading: str) -> str:
        heading_text = re.sub(r'^#+\s*', '', heading)
        # Remove any sequence of numbers with periods at the start 
        # (e.g., "2.", "2.2", "2.2.2", etc.)
        heading_text_without_numbers = re.sub(r'^\d+(\.\d+)*\s*', 
                                              '', 
                                              heading_text)
        anchor = heading_text_without_numbers.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = anchor.strip().replace(' ', '-')
        return f"[{heading_text}](#{anchor})"

class NumberingHandler():
    def __init__(self):
        """Adjust the numbering of figures and references in the report.
        """
        pass

    def adjust_figures_identifiers(self, report: str):
        """Adjust the numbering of figures in the report, assigning consecutive
        integers in order of appearance.
        
        Parameters
        ----------
        - report [`str`]: text of the report in markdown format
        
        Output
        ------
        - `str`: text of the report in markdown format with updated figure 
        identifiers
        """
        # match "Figure X" or "figure X"
        pattern = r'\bfigure\s*(\d+)\b|\bFigure\s*(\d+)\b'
        matches = re.findall(pattern, report)
        numbers = [int(match[0] if match[0] else match[1]) for match in matches]

        # Create a mapping of the old number to the new sequential number,
        # ensuring the order is preserved
        unique_numbers = sorted(set(numbers), key=numbers.index)
        number_mapping = {old: new + 1 for new, old in enumerate(unique_numbers)}

        def replace(match):
            number = int(match[0].lower().replace("figure ", "").strip() if match[0] else match[1].lower().replace("figure ", "").strip())
            return f"Figure {number_mapping[number]}"

        new_report = re.sub(pattern, replace, report)
        return new_report

    def adjust_references_identifiers(self, 
                                      report: str, 
                                      ref_df: pd.DataFrame) -> str:
        """Adjust the numbering of references in the report, assigning consecutive
        integers in order of appearance.
        
        Parameters
        ----------
        - report [`str`]: text of the report in markdown format
        - ref_df [`pd.DataFrame`]: dataframe with the references
        
        Output
        ------
        - `str`: text of the report in markdown format with updated references 
        identifiers
        """
        # Match pattern [integer, some text]
        pattern = r'\[(\d+),\s*([^\]]+?)\]'

        # List of tuples (number, text)
        matches = re.findall(pattern, report)
        numbers = [int(num) for num, _ in matches]

        # Generate unique number mapping in order of appearance
        unique_numbers = sorted(set(numbers), key=numbers.index)
        number_mapping = {old: new + 1 for new, old in enumerate(unique_numbers)}

        def replace(match):
            old_num = int(match.group(1))
            ref_text = match.group(2)
            new_num = number_mapping[old_num]
            return f"[{new_num}, {ref_text}]"

        updated_text = re.sub(pattern, replace, report)

        main, references_block = updated_text.split("## References")
        references_block = self.sort_references_by_identifier(references_block)
        updated_text = (main + f"\n\n## References\n\n{references_block}")


        # Create a helper Series with the same length as df and default to NaN
        sort_order = pd.Series([None]*len(ref_df))

        # Assign custom sort order to specified rows
        for pos, order in number_mapping.items():
            sort_order[pos-1] = order-1

        # Fill remaining with large numbers to push them to the end (while preserving their order)
        next_order = max(number_mapping.values()) + 1
        for i in range(len(ref_df)):
            if sort_order[i] is None:
                sort_order[i] = next_order
                next_order += 1

        # Add to DataFrame and sort
        ref_df['sort_order'] = sort_order
        df_sorted = ref_df.sort_values('sort_order').drop(columns='sort_order').reset_index(drop=True)


        return updated_text, df_sorted
    
    def sort_references_by_identifier(self, refs_block: str) -> str:
        """Sort the references by integer identifier.
        """
        pattern = r'\[(\d+),\s*([^\]]+)\]\s*\[([^\]]+)\]'

        # Macthes are extracted as tuples: (number, text, url)
        matches = re.findall(pattern, refs_block)
        sorted_matches = sorted(matches, key=lambda x: int(x[0]))
        sorted_ref_block = '\n\n'.join(f"[{num}, {text}] [{url}]({url})" for num, text, url in sorted_matches)
        return sorted_ref_block
