import logging
import os
from typing import Iterable, Optional

import backoff
import openai
import requests
import toml
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.general import remove_citations

total_input_tokens = 0
total_output_tokens = 0
logger = logging.getLogger(__name__)

def set_api_keys(toml_file_path: str = "secret.toml") -> None:
    """Set in the environment the API keys and endpoint from a TOML file.
    
    Parameters
    ----------
    - toml_file_path [`str`]: path to the TOML file
    """
    data = toml.load(toml_file_path)
    for key, value in data.items():
        for k, v in value.items():
            os.environ[k] = v


@backoff.on_exception(backoff.expo, 
                      (openai.RateLimitError, 
                       requests.exceptions.Timeout, 
                       openai.InternalServerError),
                       max_tries=3)
def queryLLM(
    llm: ChatOpenAI, 
    query: str
    ) -> str:
    """Query the model, with 3 maximum retries.
    
    Parameters
    ----------
    - llm [`langchain_openai.ChatOpenAI`]: model for the inference
    - query [`str`]: prompt

    Output
    ------
    - `str`: response
    """
    global total_input_tokens
    global total_output_tokens

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a scientific editor, helping me in crafting a technical document on a computer science topic."),
        ("user", "{input}")
        ])
    chain = prompt | llm

    with get_openai_callback() as cb:
        result = chain.invoke({"input": query})

    total_input_tokens += cb.prompt_tokens 
    total_output_tokens += cb.completion_tokens
    
    return result.content


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, 
    requests.exceptions.Timeout, 
    openai.InternalServerError),
    max_tries=3,
)
def _run_batch(
    llm: ChatOpenAI, 
    batch: list[str]
    ) -> tuple[list[AIMessage], OpenAICallbackHandler]:
    """Run the inference on a single batch, with 3 maximum retries allowed.
    
    Parameters
    ----------
    - llm [`langchain_openai.ChatOpenAI`]: model for the inference
    - batch [`list[str]`]: list of prompts

    Output
    ------
    - `list[AIMessage]`: responses of the model
    - `OpenAICallbackHandler`: callback to be used for the tokens count
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ])
    chain = prompt | llm

    with get_openai_callback() as cb:
        batch_results = chain.batch([{"input": q} for q in batch])
        return batch_results, cb


def queryLLM_batch(
        llm: ChatOpenAI,
        queries: list[str],
        batch_size: int = 15
        ) -> list[str]:
    """Query the model with batches, to avoid too many concurrent
    requests.
    
    Parameters
    ----------
    - llm [`langchain_openai.ChatOpenAI`]: model for the inference
    - queries [`list[str]`]: list of prompts
    - batch_size [`int`, default = 15]: batch size for the batches creation

    Output
    ------
    - `list[str]`: list of responses
    """
    global total_input_tokens
    global total_output_tokens

    results: list[Optional[AIMessage]] = [None] * len(queries)
    pending_batches = [
        (i, queries[i:i + batch_size])
        for i in range(0, len(queries), batch_size)
    ]
    max_tries = 3

    for attempt in range(1, max_tries + 1):
        failed_batches = []

        for start, batch in pending_batches:
            batch_number = start // batch_size + 1
            try:
                batch_results, cb = _run_batch(llm, batch)
                if len(batch_results) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} results, "
                        f"received {len(batch_results)}"
                    )

                results[start:start + len(batch)] = batch_results
                total_input_tokens += cb.prompt_tokens
                total_output_tokens += cb.completion_tokens
            except Exception as e:
                logger.warning(
                    "Batch %s failed on attempt %s/%s: %s",
                    batch_number,
                    attempt,
                    max_tries,
                    e,
                )
                failed_batches.append((start, batch))

        if not failed_batches:
            break

        pending_batches = failed_batches

    missing_indices = [
        index for index, result in enumerate(results) if result is None
    ]
    if missing_indices:
        message = (
            f"No results were returned for {len(missing_indices)} of "
            f"{len(queries)} queries (indices: {missing_indices})."
        )
        logger.warning(message)
        raise ValueError(message)

    return [result.content for result in results if result is not None]


class PromptOracle:
    """Centralized class for handling all prompts."""

    @staticmethod
    def _fmt(template: str, **kwargs) -> str:
        return template.strip().format(**kwargs)
    
    @staticmethod
    def get_completeness_prompt(content: str) -> str:
        return PromptOracle._fmt(COMPLETENESS_PROMPT, content=content)
    
    @staticmethod
    def get_additional_search_prompt(content: str) -> str:
        return PromptOracle._fmt(ADDITIONAL_SEARCH_PROMPT, content=content)
    
    @staticmethod
    def get_additional_search_query_gen_prompt(content: str) -> str:
        return PromptOracle._fmt(ADDITIONAL_SEARCH_QUERY_GEN_PROMPT, content=content)
    
    @staticmethod
    def get_embedded_links_extraction_prompt(content: str) -> str:
        return PromptOracle._fmt(EMBEDDED_LINKS_EXTRACTION_PROMPT, content=content)
    
    @staticmethod
    def get_embedded_links_relevance_prompt(content: str, links: str, topic: str) -> str:
        return PromptOracle._fmt(EMBEDDED_LINKS_RELEVANCE_PROMPT, content=content, links=links, topic=topic)
    
    @staticmethod
    def get_summarize_prompt(
            content: str, 
            topic: str, 
            topic_descr: Optional[str] = None, 
            url: Optional[str] = None, 
            generate_identifier: bool = False
            ) -> str:
        if (topic_descr is not None) and (topic_descr != ""):
            topic_w_descr_string = f"{topic}. A more detailed description of the topic is the following: {topic_descr}"
        else:
            topic_w_descr_string = topic
        if generate_identifier:
            return PromptOracle._fmt(SUMMARIZE_PROMPT_W_IDENTIFIER, 
                             content=content, 
                             topic=topic, 
                             topic_w_descr_string=topic_w_descr_string,
                             url=url
                             )
        else:
            return PromptOracle._fmt(SUMMARIZE_PROMPT_WO_IDENTIFIER, 
                             content=content, 
                             topic=topic, 
                             topic_w_descr_string=topic_w_descr_string
                             )

    @staticmethod 
    def get_report_prompt(topic: str, summary: Optional[str] = None) -> str:
        if summary is not None:
            return PromptOracle._fmt(REPORT_W_INFORMATION_PROMPT, topic=topic, summary=summary)
        else:
            return PromptOracle._fmt(REPORT_WO_INFORMATION_PROMPT, topic=topic)
    
    @staticmethod
    def get_outline_gen_prompt(topic: str, summary: str, description: str = "") -> str:
        if (description != "") and (description is not None):
            descr_string = f" A more detailed description of the topic is the following: {description}. "
        else:
            descr_string = ""
        return PromptOracle._fmt(OUTLINE_GEN_PROMPT, topic=topic, descr_string=descr_string, summary=summary)
    
    @staticmethod
    def get_section_gen_prompt(topic: str, outline: str, information: str, section: str) -> str:
        return PromptOracle._fmt(SECTION_GEN_PROMPT, topic=topic, outline=outline, information=information, section=section)
    
    @staticmethod
    def get_refinement_prompt(topic: str, info: str, section: str, description: str="") -> str:
        if description != "":
            descr_string = f" A more detailed description of the topic is the following: {description}. "
        else:
            descr_string = ""
        return PromptOracle._fmt(REFINEMENT_PROMPT, topic=topic, descr_string=descr_string, info=info, section=section)
    
    @staticmethod
    def get_relevance_prompt(
            documents: Iterable[str],
            topic: str,
            topic_descr: Optional[str] = None
            ) -> str:

        if (topic_descr is None) or (topic_descr == ""):
            description_string = ""
        else:
            description_string = f" A more detailed description of the topic is as follows: {topic_descr}."
        
        docs_string = ""
        for i in range(len(documents)):
            docs_string += f"[Document {i}]: {documents[i]}\n"
        
        return PromptOracle._fmt(
            RELEVANCE_PROMPT, 
            topic=topic, 
            description_string=description_string, 
            docs_string=docs_string
            )
   
    @staticmethod
    def get_description_gen_prompt(
            image_dict: dict[str, str], 
            topic: str
            ) -> str:
        """Get the prompt for the generation of the description.
        
        Parameters
        ----------
        - image_dict [`dict[str, str]`]: dictionary with information about the figure.
        In particular it should contain the following keys:  
        'caption', 'text_snippets', 'page_title'
        - topic [`str`]: topic defining the scope of the presented images
        
        Output
        ------
        - prompt [`str`]
        """
        return PromptOracle._fmt(
            DESCRIPTION_GEN_PROMPT,
            topic=topic, 
            caption=image_dict["caption"], 
            text_snippets=str(image_dict["text_snippets"]),
            page_title=image_dict["page_title"],
            summary=image_dict["summary"]
            )
    
    @staticmethod
    def get_figures_selection_prompt(
            descriptions: Iterable[str], 
            outline: str, 
            topic: str
            ) -> str:

        descriptions_text = ""
        for i, description in enumerate(descriptions):
            descriptions_text += f"\n\nFIGURE {i}\nDESCRIPTION: {description}\n"

        return PromptOracle._fmt(FIGURES_SELECTION_PROMPT, 
                         descriptions_text=descriptions_text, 
                         topic=topic, 
                         outline=outline)
    
    @staticmethod
    def get_figures_insertion_prompt(
            figures: Iterable[dict], 
            section: str, 
            topic: str
            ) -> str:
        images_text = ""
        for image in figures:
            images_text += f"\n IMAGE PATH: {image["placeholder_path"]}\nDESCRIPTION: {image["description"]}\nFIGURE IDENTIFIER: {image["figure_identifier"]}"
        return PromptOracle._fmt(FIGURES_INSERTION_PROMPT, images_text=images_text, section=section, topic=topic)
    
    @staticmethod
    def get_citations_check_prompt(
            topic: str, 
            section: str, 
            information: str
            ) -> str:
        return PromptOracle._fmt(CITATIONS_CHECK_PROMPT, topic=topic, section=section, information=information)
    
    @staticmethod
    def get_gen_atomic_claims_prompt(text: str) -> str:
        return PromptOracle._fmt(GEN_ATOMIC_CLAIMS_PROMPT, text=text)
    
    @staticmethod
    def get_entail_prompt(
            claim: str, 
            references_text: str, 
            partial: bool = False
            ) -> str:
        if partial:
            return PromptOracle._fmt(
                PARTIAL_ENTAILMENT_PROMPT, 
                references_text=references_text, 
                claim=remove_citations(claim)
                )
        else:
            return PromptOracle._fmt(
                FULL_ENTAILMENT_PROMPT, 
                references_text=references_text, 
                claim=remove_citations(claim)
                )

    @staticmethod    
    def get_stage1_outline_prompt(
            topic: str, 
            info: str, 
            description: str=""
            ) -> str:
        if description != "":
            descr_string = f"A more detailed description of the topic is the following: {description}. "
        else:
            descr_string = ""

        return PromptOracle._fmt(STAGE1_OUTLINE_PROMPT, topic=topic, info=info, descr_string=descr_string)

    @staticmethod
    def get_stage2_outline_prompt(
            topic: str, 
            info: str, 
            outlines: Iterable[str], 
            description: str = ""
            ) -> str:
        if description != "":
            descr_string = f"A more detailed description of the topic is the following: {description}. "
        else:
            descr_string = ""
        outlines_list = ""
        for i, outline in enumerate(outlines):
            outlines_list += f'''OUTLINE {i}: """{outline}\n"""'''
        
        return PromptOracle._fmt(
            STAGE2_OUTLINE_PROMPT,
            topic=topic,
            descr_string=descr_string,
            info=info,
            outlines_list=outlines_list
            )
    
    @staticmethod
    def get_build_table_prompt(
            topic: str, 
            information: str, 
            topic_descr: Optional[str] = None,
            ) -> str:
        if (topic_descr is None) or (topic_descr == ""):
            description_string = ""
        else:
            description_string = f". A more detailed description of the topic is the following: {topic_descr}"
        return PromptOracle._fmt(
            BUILD_TABLE_PROMPT,
            topic=topic,
            description_string=description_string,
            information=information)
    
    @staticmethod
    def get_insert_table_prompt(
            topic: str, 
            table: str, 
            explanation: str, 
            section: str
            ) -> str:
        return PromptOracle._fmt(
            INSERT_TABLE_PROMPT, 
            topic=topic, 
            table=table,
            explanation=explanation,
            section=section
            )
    
    @staticmethod
    def get_table_placement_prompt(
            report: str,
            table: str
            ) -> str:
        return PromptOracle._fmt(TABLE_PLACEMENT_PROMPT, report=report, table=table)

    @staticmethod
    def get_topic_definer_prompt(
            provided_info: str, 
            provided_info_fields: str
            ) -> str:
        return PromptOracle._fmt(TOPIC_DEFINER_PROMPT, provided_info=provided_info, provided_info_fields=provided_info_fields)

    @staticmethod
    def get_enrich_section_prompt(
            topic: str,
            old_section: str,
            information: str
            ) -> str:
        return PromptOracle._fmt(
            ENRICH_SECTION_PROMPT,
            topic=topic,
            old_section=old_section,
            information=information
        )
    
    @staticmethod
    def get_correct_claim_prompt(
        text_unit: str, 
        section: str, 
        claims: list[str], 
        explanations: list[str], 
        refs: str
        ) -> str:
        """It returns the prompt for the claims' correction.
        
        Parameters
        ----------
        - text_unit [`str`]: text snippet
        - section [`str`]: section where the text snippet is located
        - claims [`List[str]`]: list of claims
        - explanations [`List[str]`]: list of explanations about the claims
        - refs [`str`]: concatenation of the references' text
        
        Output
        ------
        - `str`: prompt
        """
        claims_info = ""
        for cl, ex in zip (claims, explanations):
            claims_info += f"""
        Claim: '{cl}'
        Explanation: '{ex}'
        """
            
        return PromptOracle._fmt(
            CORRECT_CLAIM_PROMPT,
            text_unit=text_unit,
            section=section,
            claims_info=claims_info,
            refs=refs
            )
    
    @staticmethod
    def get_headings_revision_prompt(
            title: str, 
            heading: str, 
            section_text: str
            ) -> str:
        return PromptOracle._fmt(
            HEADINGS_REVISION_PROMPT, 
            title=title, 
            heading=heading, 
            section_text=section_text
            )
    
    @staticmethod
    def get_queries_gen_prompt(
            n: int,
            topic: str, 
            description: Optional[str]
            ) -> str:
        if description is None:        
            return PromptOracle._fmt(
                QUERIES_GEN_WO_DESCRIPTION_PROMPT,
                n=n,
                topic=topic)
        else:
            return PromptOracle._fmt(
                QUERIES_GEN_W_DESCRIPTION_PROMPT,
                n=n,
                topic=topic,
                description=description)
    


COMPLETENESS_PROMPT = '''
You are a scientific editor, and you have to analyze a markdown text of a scraped webpage to gather useful information for your research. You have to assess whether the entire content has been successfully retrieved or whether there have been issues in the scraping phase (e.g., HTTP errors or sign up pages overrode the content). Technical documents, preprint papers, and social media content are all accepted. Provide as output an integer score in the range [0,9].
The score must be high if the full-text of the webpage was correctly scraped.
The score must be low if the text has missing sections, or if the parsing of the content of the webpage is not satisfying.

Here there is the text: """{content}"""

Output only the integer score with no explanation or extra text.
'''


ADDITIONAL_SEARCH_PROMPT = '''
You are a scientific editor, and you have to analyze a markdown text of a scraped webpage to assess whether the content is complete or a full-version should be searched on the web. This could be the case of paywalled papers, for which an open access arXiv version could be found on the Internet. Do not consider the embedded links or cited related works. Provide as output an integer score in the range [0,9].
The score must be high if the full-text could be retrieved with an additional search query (e.g., in the case of an IEEE page with paywall showing only the title of a paper).
The score must be low if an additional search would lead to the very same webpage (e.g., Medium articles, for which no subscription is available), or if the full-text is already available.             

Here there is the text: """{content}"""

Output only the integer score with no explanation or extra text.'''


ADDITIONAL_SEARCH_QUERY_GEN_PROMPT = '''
You are a scientific editor, and you have a markdown text of a scraped webpage for which a full-text should be found. You have to read the content of the webpage and define a search query that allows to retrieve a full-length version of it.
This can be the case, for instance, of paywalled paper, for which we want to retrieve an arXiv version.
Here there is the text: """{content}"""
Provide only the exact search query as output. If you cannot provide a specific query, return an empty string.'''


EMBEDDED_LINKS_EXTRACTION_PROMPT = '''
You are a scientific editor, and you have to analyze a markdown text of a scraped webpage to gather useful information for your research. You have to extract all the links, if any, in the input markdown text, mapping the relative urls to absolute urls. Ignore links to images or subsections of the webpage. 
Here there is the text: 
"""{content}"""

Provide as output one extracted URL per line. If a document does not contain any URL, return <NONE>, without any explanation. Do not hallucinate. When writing, do not add any triple backticks (```) or specify any language (e.g., markdown, python), just provide the raw textual content.'''


EMBEDDED_LINKS_RELEVANCE_PROMPT = '''
You are a scientific editor that has to assess whether some URLs could bring more insights into you research on the topic of {topic}.
Given the input markdown text and a list of embedded links extracted from it, score their relevance within the range [0,9].
The score must be low if the embedded link does not bring useful insights into the topic of {topic} (e.g., implementation code, author profiles or group pages), or if it is not focusing specifically on the core topic of {topic}.
The score must be high if the embedded link can potentially contain relevant follow-up discussions on the topic of {topic} specifically. If the embedded link is on a related topic, but not exactly on {topic} you must not assign high scores.
To assign the score you must consider the link's surrounding text, which conveys some explanation about it. 
Here is the text: """{content}"""
Here is the links list: """{links}"""

Provide as output a JSON object with the line index of the URL as key and the associated score as value, without any explanation. When writing, do not add any triple backticks (```) or specify any language (e.g., markdown, python), just provide the raw textual content.'''


SUMMARIZE_PROMPT_WO_IDENTIFIER = '''
As a scientific editor, you are given a markdown text containing information related to the topic of {topic_w_descr_string}. You have to summarize the content, extracting the most valuable information to write a report on {topic}. Focus in particular on technical aspects, mathematical equations, advantages and limitations with respect to other approaches, positive and negative feedback, and future directions. Do not organize the summary into sections. 

Here is the text: """{content}"""

Provide as output the summary, without any special font formatting. Be as complete as possible, do not discard content to make the summary short. Be complete and extensively cover all the important information. If there is not information available on {topic}, do not hallucinate.'''

SUMMARIZE_PROMPT_W_IDENTIFIER = '''
As a scientific editor, you are given a markdown text containing information related to the topic of {topic_w_descr_string}. You have two tasks:
1) Summarize the content, extracting the most valuable information to write a report on {topic}. In particular, if the resource is a research paper, cover this aspects:
- Methodology: write the main ideas and assumptions of the work. Be complete, integrating the ideas with their main explanations. Include all relevant technical details and mathematical equations.
- Key results: write the main results and conclusions of the work. When you write quantitative results, you must report the the comparison baseline, the models and datasets that the numbers refer to.
- Advantages/Limitations: report the main benefits or challenges discussed.
If the resource is a blog post, report the main discussion points and the arguments.
Generate the summary, without any special font formatting. Be as complete as possible, do not discard content to make the summary short. Be complete and extensively cover all the important information. If there is not information available on {topic}, do not hallucinate. Ensure to use a neutral tone, even when the resources are biased or emotionally charged: your summary must be scientifically objective.
2) After reading the content, you have to generate an identifier for this specific resource. If the content is a scientific paper, there are two cases: (1) if the proposed methodology is associated to an acronym within the text, use it as is as identifier. (2) If the proposed methodology has no specific name, create one by yourself, using also the name of the first author as starting point. Do not use the doi nor the arxiv identifier.
If the content does not propose a specific methodology but is rather a more general page, use as starting point the url. For instance, for a Wikipedia page about Neural Networks, you could generate the identifier "Wikipedia - Neural Networks". Do not add comments or explanations on the identifier.

Here is the text: """{content}"""
Here there is the URL: """{url}"""

Provide as output the summary and the identifier. Structure your output as follows:

### Summary
Summary of the content

### Identifier
The generated identifier
'''

REPORT_W_INFORMATION_PROMPT = '''
You are a scientific editor, and you have to generate a technical report about the topic of {topic} using your knowledge and the gathered information. 
Write the text using inline citations within square brackets. If there are multiple consecutive citations, use the following format: [x],[y],[z]. Try to maximize the usage of the gathered information and the associated citations. If multiple references support a claim, include all of them.
Do not include the references section at the end of the report.
Make the report as informative as possible, maximizing its length. Present the concepts with intuitive explanations, and possibly some analogies.
The report should include an easy to understand introduction to the topic of {topic}, a background section, a technical overview of the main concepts and components of {topic} and an extensive opinions section that includes both positive and negative feedbacks.

Here is the information: """{summary}"""

Provide as output the technical report in the markdown format. Provide as many details as possible and use the references as much as possible, without trying to reduce the length of the report. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content.'''
    
REPORT_WO_INFORMATION_PROMPT = '''
You are a scientific editor, and you have to generate a technical report about the topic of {topic} using your knowledge. If multiple references support a claim, include all of them.
Write the text using inline citations within square brackets. If there are multiple consecutive citations, use the following format: [x],[y],[z]. 
Make the report as informative as possible, maximizing its length. Present the concepts with intuitive explanations, and possibly some analogies.
The report should include an easy to understand introduction to the topic of {topic}, a background section, a technical overview of the main concepts and components of {topic} and an extensive opinions section that includes both positive and negative feedbacks.

Provide as output the technical report in the markdown format. Provide as many details as possible and use the references as much as possible, without trying to reduce the length of the report. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content.'''
    

OUTLINE_GEN_PROMPT = '''
You are a scientific editor and you have collected relevant information on {topic}.{descr_string} You have to write the outline of a technical report that presents what you have found out.
Write only the headings, sub-headings, and so on, with the markdown format. Do not write any section text.
The outline can include:
- an introductory/motivation section to the topic of {topic}
- a background section
- a technical overview of the main concepts and components of {topic} that covers also the mathematical foundations if any.
- a section presenting the main topology of the main approaches in the topic of {topic}. Do not detail specific works from the references.
- extensive benefits and limitations sections. Organize the various concepts in subsections.
- research opportunities.
- a section that details the main characteristics of specific key works. Name the subsections after the presented work.

You are not required to include all these sections, but they can serve as general structure.

Exclude the following:
- Acknowledgments
- Glossary
- Scope or objectives of the report
- References section

Ground the generation of the outline on the provided information.

Here there is the information: """{summary}"""

Generate an outline for a technical report, excluding the references section. Provide only the section headings, starting from level 2 (##) for the first section (e.g., ## 1. Section). Number the headings appropriately (e.g., ### 1.1 Subsection). Do not include any text within the sections, and avoid markdown syntax elements like triple backticks or specifying the language. Do not include the title of the document.'''

    

SECTION_GEN_PROMPT = '''
You are a scientific editor, and you have to write a technical report on the topic of {topic}. You have collected some information to be used to write the report. You are given an outline of the technical report that you have to adhere to.
You have to write the text of the section {section} only, including the provided subsections, using the collected information as much as possible. Ground your claims on the collected information, placing the inline citation within square brackets. If there are multiple consecutive citations, use the following format: [indentifier_x], [indentifier_y], [identifier_z], where identifier_x, identifier_y, and identizier_z are the integer + textual identifiers specified in between brackets in the collected information. For instance, if you use reference [4, ABC - DEF] of the collected information, you have to cite it in the text as follows: [4, ABC - DEF].
Present the concepts with extensive explanations, and possibly some analogies. Do not include overview tables.

Here there is the outline: """{outline}"""

Here there is the collected information: """{information}"""

Provide as output the text for the section {section} only, in the markdown format. Provide as many details as possible, grounding the content on the collected information. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content.
Ground the text on the provided references and cite them. Do not cite references which are not included in the information. Do not alter the levels of the section's headings by changing the number of "#" characters. E.g., if you are prompted to write the section corresponding to the heading """## 1. Section
### 1.1 Subsection""" do not modify it to """# 1. Section
## 1.1 Subsection""".
'''


REFINEMENT_PROMPT = '''
You are a scientific editor, and you are writing a technical report on the topic of {topic}.{descr_string} You have to refine and enrich a given text by incorporating relevant details from a set of provided references. Your goal is to ensure the revised text is comprehensive, coherent, and clear, while seamlessly integrating the additional information. Do not modify or remove any link or image placeholder, and do not remove any references to them in the text.

To accomplish the goal, follow these steps:
1) Analyze the original text: identify the key points, structure, and any gaps in information.
2) Review the references: extract all relevant details, examples, or explanations that are missing in the original text but would enhance its depth and completeness.
3) Enrich the text: incorporate the missing details into the original text in a way that maintains logical flow, clarity, and readability. If you mention new concepts or techniques, you must give enough details for the reader to understand the scope of the conveyed information. You must include details taken from the provided references only.
3) Check the citations: if you add citations, use the following format: [indentifier_x], [indentifier_y], [identifier_z], where identifier_x, identifier_y, and identizier_z are the integer + textual identifiers specified in between brackets in the collected references. For instance, if you use reference [4, ABC - DEF] of the collected references, you have to cite it in the text as follows: [4, ABC - DEF].
4) Revise the text: rewrite the old and new content to make it have a scientific and objective tone. Make sure the flow is coherent and cohesive. Ensure there are no repetitive sentences or claims. You are allowed to modify the order of the claims if that improves the flow of the presentation. Do not modify the headings' structure. Do not add new subsections or summaries. Do not modify links and source placeholders.

Here there is the original text: """{section}"""

Here there are the references: """{info}"""

Provide as output the revised text that includes all relevant details from the references, ensuring it is well-structured, coherent, and clear. Do not modify the headings' structure. Do not add new subsections or summaries. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content without any additional note or explanation. Do not modify or remove any link or image placeholder, and do not remove any references to them in the text.
'''

RELEVANCE_PROMPT = """
You are an anomaly detector. Analyze the following group of documents on the topic of "{topic}".{description_string} You have to infer their common content, then identify which ones are outliers.  

Steps:  
1. Read all documents below and acquire knowledge on the topic of {topic}.  
2. Infer the main content about {topic} (keywords, concepts etc.).  
3. Flag outliers: list documents that lack any alignment with the topic of {topic}. If you are not certain about a document, do not flag it as outlier.

Documents:
{docs_string}

Output format:  
Outliers:  
- Document X  
- Document Y

Remove only the clearly out-of-topic documents, keeping the documents that are connected to the topic."""



DESCRIPTION_GEN_PROMPT = '''
You are a scientific editor reviewing a technical report on the topic of {topic}. You are provided with the caption of a figure extracted from a paper, relevant excerpts mentioning it, the title of the article and its summary. Your task is to generate an accurate description of what the figure depicts, based on the image caption, the text snippets, the article title and its summary.  It is important that your description clarifies if the image refers to a specific methodology (mentioned in the title) or it is about a general concept. If it refers to a specific methodology, mention it explicitly in the description using the source indicated in the title.
Here is the caption: """{caption}"""

Below are the text excerpts that reference the figure in the paper: """{text_snippets}"""

Here is the article title (it can be empty): """{page_title}"""

Here there is the summary of the article: """{summary}"""

Provide as output the description, specifying whether the image refers to a general concept or a specific framework. Do not mention the figure number in the description, nor additional specific citations identifiers (e.g. "[13]"). Ground the description on the provided textual information only.'''



FIGURES_SELECTION_PROMPT = '''
You are a scientific editor tasked with writing a technical report on {topic}. You have been provided with an outline of the report and descriptions of several figures. Your goal is to select the figures that are most relevant to the report.

* A figure is considered highly relevant if it directly illustrates a concept from the outline.
* A figure is low relevance if it depicts concepts that are not included in the outline or if it is redundant with other figures.
* Experimental results and application-specific figures should only be included if they are absolutely necessary.

Here is the outline of the report:
{outline}

Figure descriptions:
{descriptions_text}

Instructions:

1) Select only figures that directly relate to the concepts in the outline.
2) Assign each selected figure to the most appropriate section based on the outline. Assignments should only be made to the lowest-level headings (e.g., in an outline structure like #2, #2.1, #2.1.1, #2.1.2, you should only assign sections to #2.1.1 and #2.1.2). Do not assign the same figure to multiple sections!
3) Classify figures into critical or low priority:
    - Critical figures are those essential for understanding the concepts in the report.
    - Low priority figures are those that are less relevant or redundant.

The output should be in JSON format, with the following structure:
    
{{
"critical_figures": [<list of dictionaries with figure identifier as key and assigned section as value>]}}where the list contains the identifiers of the figures.

An example output could be:

"""
{{"critical_figures": [{{"FIGURE 4": "1.1 Introduction and main concepts"}}, {{"FIGURE 12": "2.4.5 Mathematical aspects"}}]
}}
"""

Only flag figures as critical if they directly support the concepts outlined in the report, and not just their application in specific use cases. Keep the figure selection precise and minimal, being extremely selective.'''


FIGURES_INSERTION_PROMPT = '''
You are a scientific editor writing a technical report about {topic}. 
You are given a section of the report, and a list of figures, each with its description. 
Your task is to insert figures into the section only if they are semantically related to the text.

You have four tasks:

1. Assess relevance: Determine whether each image aligns with the section's content and context. Disregard any figure that is not relevant (e.g., do not insert a figure showing a specific methodology into a section that discusses other works or unrelated content).

2. Insert images: If a figure is relevant, insert it using the following markdown syntax, placing it between sentences where appropriate. Use the provided figure identifier as figure number.
    ```
    ![Image](path of the image)
    *Figure X. [Concise figure caption]*\n\n
    ```

3. Reference figures in text: For every inserted figure, add an explicit reference to it in the surrounding text, using standard scientific language. Use phrasing like "Figure X shows...", "As illustrated in Figure X...", or "Figure X provides an overview of...". The reference must:
    - Clearly explain the relevance of the figure.
    - Be based on the figure description provided.
    - Be located before or after the image, as appropriate for clarity.

4. Maintain scientific tone: Ensure the overall writing style remains formal, concise, and aligned with scientific reporting norms. Ensure that the images' mentions are integrated smoothly within the text.

Here there is the information about the figures: """
{images_text}
"""

Here is the text of the section: """
{section}
"""

Provide as output the modified markdown text of the section, with no additional comment.
When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content.
Do not modify the image paths.
Do not add headings, only text.'''


CITATIONS_CHECK_PROMPT = '''
You are a scientific editor tasked with verifying the factual accuracy of a technical report on {topic}. Your goal is to ensure that each citation in the report accurately supports the associated claims. You are provided with the content of the references. Perform the following tasks:
    1) Read the section carefully.
    2) For each citation, check whether the corresponding reference supports the claim in the text. If it does, leave the citation as is.
    3) If a citation does not support the claim, remove it.
    4) If you find additional reference that fully support the claim, add them in the following format: <sup><small>[integer identifier, textual identifier]</small></sup>. When adding multiple citations consecutively, use the following format: <small><sup>[integer identifier ref. 1, textual identifier ref. 1][integer identifier ref. 2, textual identifier ref. 2]</small></sup>, without separating them with a comma. Only add citations that clearly back the statement.

Here is the section text: """{section}"""

Here are the references and their contents: """{information}"""

Instructions:
- Do not change the content of the section, headings, or structure.
- Modify only the citations as needed.
- Keep the citation identifiers unchanged from the provided references.
- Provide the modified section text only (without explanation or markdown).'''

GEN_ATOMIC_CLAIMS_PROMPT = '''
You are an experienced researcher in the computer science domain. You are given a claim and your task is to represent it as a list of atomic sub-claims. Do not include in the list items that are very general or well-known facts. Each item in the list you create must have the format '[identifier]: [description]' where [identifier] is the number of the atomic sub-claim (such as 1, 2, 3, etc.), and [description] is the description of the atomic sub-claim. Each item must begin on a new line. 
If a part of the text refers to an image, do not generate claims that would require looking at the image, rather focus on textual information.
Generate self-contained claims, that can be understood without any additional information.
Here there is the text: """{text}"""

Provide only the list itself and be concise.'''


PARTIAL_ENTAILMENT_PROMPT = """
You are a scientific reviewer, who has to assess the factuality of claims in a technical report, given the references. You are given a claim and the text of the cited reference. You have to assess whether the text of the reference at least partially supports the claim. Do not use any outside or general knowledge in your evaluation: rely solely on the content provided in the source.\nSource: {references_text}\nClaim: {claim}\nStart your answer with 'Yes' if the claim is supported, 'No' if it is not supported by the reference text, followed by a brief justification. For the evaluation, you must rely on the content of the source text only and not on your knowledge."""


FULL_ENTAILMENT_PROMPT = """
You are a scientific reviewer, who has to assess the factuality of claims in a technical report, given the references. You are given a claim and the text of the cited reference. You have to assess whether the text of the reference fully supports the claim. Do not use any outside or general knowledge in your evaluation: rely solely on the content provided in the source.\nSource: {references_text}\nClaim: {claim}\nStart your answer with 'Yes' if the claim is supported, 'No' if it is not supported by the reference text, followed by a brief justification. For the evaluation, you must rely on the content of the source text only and not on your knowledge."""


STAGE1_OUTLINE_PROMPT = '''
You are a scientific editor tasked with organizing a technical report on the topic of {topic}.{descr_string} Based on the information you have gathered, your goal is to create an outline for the report. The outline should include only section and subsection headings in markdown format, with no section text.

Base the outline solely on the information provided below.

Here is the relevant information: """{info}"""

Generate an outline for a technical report, excluding the references section. Refer exclusively to the provided relevant information for the outline definition. Provide only the section headings, starting from level 2 (##) for the first section (e.g., ## 1. Section). Number the headings appropriately (e.g., ## 1.1 Subsection). Do not include any text within the sections, and avoid markdown syntax elements like triple backticks or specifying the language. Do not include the title of the document.'''


STAGE2_OUTLINE_PROMPT = '''
You are a scientific editor and you have to write the outline of a technical report about the topic of {topic}.{descr_string} Some collaborators have generated some outline drafts covering the key aspects of the topic that should be included in the report. You have to combine them all into a single, comprehensive, and logically structured outline, using the collected information as ground-truth, and avoiding redundancy across sections.

The outline may include:
- an introductory/motivation section to the topic of {topic}
- a background section
- a technical overview of the main concepts and components of {topic} that covers also the mathematical foundations if any.
- a section presenting the main topology of the main approaches in the topic of {topic}. Do not detail specific works from the references here!!
- extensive benefits and limitations sections. Organize the various concepts in subsections.
- research opportunities.
- a section that details the main characteristics of various specific key works. Name the subsections after the presented work.

Exclude the following:
- acknowledgments
- glossary
- scope or objectives of the report
- references section

Here there is the outlines list: """{outlines_list}"""

Here there is the collected information: """{info}"""

Generate your outline, trying to be concise and having a logical and not too verbose structure. You must avoid redundancy across different sections. You can modify the suggested structure and the provided outlines' ones to improve the quality of your outline. You can use only the content already mentioned in the given outlines. Provide only the section headings, starting from level 1 (#) for the report title. Number the headings appropriately (e.g., # Title, ## 1 Section, ### 1.1 Subsection). Do not include any text within the sections, and avoid markdown syntax elements like triple backticks or specifying the language. Do not add any comment to the answer.
'''

BUILD_TABLE_PROMPT = '''
You are a scientific editor preparing a technical report on {topic}.{description_string} You have collected a set of resources and need to create a detailed comparative analysis in table format. The table should compare key methodological aspects from the various sources. 

To accomplish this, you have to:
1) Read the documents carefully and identify the distinct relevant works for the comparative analysis. 
2) Extract informative common criteria to differentiate among the different works. Focus on methodological aspects that are different from one technique to the other. Do not include general criteria (e.g. avoid "Efficiency", "Example", "Hardware-friendly") that would lead to an ineffective and uninformative classification. Do not consider quantitative metrics, as they can refer to different datasets. 
3) Classify each work for each criteria. If there is not enough data to make a confident assessment leave the field blank - do not hallucinate.

Here is the information you have collected:
"""{information}"""

Your output should be:
1) A table, formatted clearly with rows and columns, showing the comparison. If the data is insufficient or not comparable, please return an empty string.
2) A textual explanation of the table. Ground your analysis on the provided information.

Use the following output format:

# TABLE
<Insert the table here>

# EXPLANATION
<Insert the textual explanation of the table here>

When you insert citations in the table and in the textual explanation of the table, you cannot use only the integer identifier for the citations, but you must include the textual identifier as well, as indicated in the collected information. Thus, you must cite references using the following format: [source integer identifier, source textual identifier]. The source textual identifier must be identical to the one defined in the collected information.
'''


INSERT_TABLE_PROMPT = '''
You are a scientific editor tasked with enhancing a report on the topic of {topic}. You have received a comparative analysis in table format and some explanatory notes that you must incorporate into the a specific section of the report. Your goal is to seamlessly integrate this table and the corresponding explanation into the section, ensuring that the content flows smoothly and logically.
    
Your tasks are:
1) Identify the appropriate placement for the table:
    - Determine the best position for the table within the section. Consider the flow of the content and where the table would best support the text.
    - If necessary, create a new subsection for the table and adjust the section numbering accordingly.

2) Write a caption to the table:
    - Write a clear and concise caption for the table, based on the explanation you have received. 
    - The caption must be in italic, with the following format:
*Table X. [Concise table caption]*

3) Insert the table and its caption into the section:
     - Add the table and its caption at the chosen location in the report, ensuring it is well-placed within the overall structure.
     - Use the following format:
[Table caption]\n\n[Table content]

4) Add an explicit reference to the table in the surrounding text, using standard scientific language. Use phrasing like "Table X shows...", or "Table X provides an overview of...". The reference must:
   - Be based on the explanatory notes provided, and must be followed by an extensive explanation of the key findings of the table.
   - Be located before or after the table, as appropriate for clarity.
   - Do not use citations as the grammatical subjects of sentences. For instance, instead of "Layer-centric approaches, like [5, SimLayerKV - Marktechpost] and [12, PyramidKV], exploit inter-layer redundancies, achieving significant memory reductions.", do write "Layer-centric approaches, like SimLayerKV [5, SimLayerKV - Marktechpost] and PyramidKV [12, PyramidKV], exploit inter-layer redundancies, achieving significant memory reductions."
   - Ensure that all modified and unchanged subsections of the section are included in your output.

Here there is the table content: """{table}"""

Here there is the explanatory text for the table: """{explanation}"""

Here there is the section of the report: """{section}"""

Provide the text in markdown format, ensuring that:
- You include the table, caption, and section text. Do not omit also the other subsections that remained unchanged, include them as well in your output.
- Do not include any markdown code blocks or language specifications; simply provide the raw markdown content.
- If you add new citations within the text, do not add hyperlinks. 
- Use the same citation format of the table, i.e. the following: [integer identifier of the source, textual identifier of the source]. Do not use thew following format: [integer identifier]. 
- Write the text using inline citations within square brackets. If there are multiple consecutive citations, use the following format: [integer, text],[integer, text],[integer, text]. Try to maximize the usage of the gathered information and the associated citations.

You must write completely also the parts of the sections that you did not modify. When you cite a source, use the following format: [integer identifier of the source, textual identifier of the source].
'''


TABLE_PLACEMENT_PROMPT = '''
You are a scientific editor tasked with incorporating key results into a technical report on the topic of KV cache. You are provided with the report text and a table containing relevant data.

Report text:
"""{report}"""

Table:
"""{table}"""

Your task is to determine the appropriate second-level section (indicated by the "##" symbol, that should not be omitted in your answer) where the table should be placed within the report. Provide only the exact section name, without any additional explanation.
    '''

TOPIC_DEFINER_PROMPT = '''
You are a scientific editor, tasked of crafting a report about a topic.
The user has provided you some information about the topic, in particular {provided_info}. You have to use this information to create a string that defines clearly the topic of interest. This string will be used afterwards to perform web searches on the topic, so it is important that it is very clear and reflects the user's interest.

This is the information that you have received:

"""{provided_info_fields}"""

Provide as output the topic definition. In particular, structure the output as follows:

Description: [Precise description of the topic]
Title: [the topic definition into a concise sentence]

Do not add any explanation and do not format the text in markdown formats.
'''


ENRICH_SECTION_PROMPT = '''
You are a scientific editor crafting a technical report about the topic of {topic}.You have collected some information about the topic, and now you have to revise a section of the report, enriching it as much as possible with the collected information. In particular, provide more details about the techniques mentioned in the section. Do not add information outside of the scope of the section. If you add new citations, use the following format: [indentifier_x], [indentifier_y], [identifier_z], where identifier_x, identifier_y, and identizier_z are the integer + textual identifiers specified in between brackets in the collected information. For instance, if you use reference [4, ABC - DEF] of the collected information, you have to cite it in the text as follows: [4, ABC - DEF].

Here there is the section text in markdown format: """{old_section}"""

Here there is the collected information: """{information}"""

Provide as output the enriched and refined version of the section text in markdown format. Ground the added details on the collected information, including as many as possible. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content.
'''


CORRECT_CLAIM_PROMPT = '''
You are a scientific editor tasked with revising a text snippet to ensure scientific accuracy and rigor. You will be provided with:

- A text snippet containing one or more potentially inaccurate claims.
- The full section from which the text snippet is extracted (for context).
- A list of problematic claims, each with an explanation of the issue and suggested corrections.
- Relevant references to support your revisions.

Your task is to:
1. Locate each claim in the text snippet.
2. Assess whether each claim is entirely incorrect and should be removed, or whether it is simply imprecise and can be revised.
3. Modify the text snippet to correct or remove each claim. Use the full section as context to preserve the intended meaning of the snippet. Base your edits on the explanations provided. If correcting the claim would change the meaning of the text in a way that is inconsistent with the surrounding section, remove the claim and/or its sentence entirely.
4. Preserve any in-text citations in the format [citation textual identifier].

Here is the text snippet:  
"""{text_unit}"""

Here is the full section for context:  
"""{section}"""

Here are the claims and their explanations:  
"""{claims_info}"""

Here are the references:  
"""{refs}"""

Output only the corrected version of the text snippet, without any explanation. When generating markdown, do not add any triple backticks (```) or specify the language (markdown), just provide the raw markdown content. Do not enclose the text snippet with any leading and tailing quotes characters (""").'''


HEADINGS_REVISION_PROMPT = '''
You are a scientific editor revising a report titled "{title}". Your task is to ensure that the section heading and all subsection headings accurately reflect the content. If a heading or subheading includes a method, technique, concept not discussed in the text of that specific section or subsection, do not include the reference in the new version.

Here is the current section heading: """{heading}"""  
Here is the section text: """{section_text}"""

Return a JSON object with the following structure:

{{
"## 1 Old section name": "## 1 New section name",
"### 1.1 Old subsection name": "### 1.1 New subsection name",
...
}}

- Include all subheadings, even if unchanged.
- Do not modify the heading levels (# symbols) and their numbering (e.g., 3.).
- Output only the JSON - no extra explanation or formatting.
- Do not name sections after the title of the report.
'''


QUERIES_GEN_WO_DESCRIPTION_PROMPT = '''
You are a researcher and have to write a technical report on the topic of "{topic}". The report should balance detailed technical insights with a significant discussion on challenges, opportunities, and limitations.

Generate {n} useful Google Search queries to gather relevant information on the topic. Focus on the methodology aspects rather than applications. Find both technical content and opinionated user discussions addressing the pros and cons. Present the queries in this format:

-<Q>: query 1
-<Q>: query 2
-<Q>: query {n}

Each query should start with '-<Q>:', and no explanations are needed, write only the queries themselves, without enclosing them in double quotes.'''


QUERIES_GEN_W_DESCRIPTION_PROMPT = '''
You are a researcher and have to write a technical report on the topic of "{topic}". The report should balance detailed technical insights with a significant discussion on challenges, opportunities, and limitations. You have also a more precise description of the topic:

Description: """{description}"""

Generate {n} useful Google Search queries to gather relevant information on the topic, given its title and description. Focus on the methodology aspects rather than applications. Find both technical content and opinionated user discussions addressing the pros and cons. Present the queries in this format:

-<Q>: query 1
-<Q>: query 2
-<Q>: query {n}

Each query should start with '-<Q>:', and no explanations are needed, write only the queries themselves, without enclosing them in double quotes.'''
