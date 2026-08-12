import json
import logging
import os
import re
import subprocess
import time
from collections import Counter
from enum import Enum, auto
from itertools import combinations
from typing import Optional

import backoff
import pandas as pd
import seaborn as sns
import tqdm
from langchain_openai import ChatOpenAI
from matplotlib import pyplot as plt

from utils.content_extraction import (ResourceType, assign_resource_type,
                                      convert, fix_url,
                                      remove_embedded_links,
                                      retrieve_results_from_queries)
from utils.general import (TextSimilarityEvaluator, filter_arxiv_url,
                           get_number_tokens, get_number_words,
                           normalize_arxiv_url, parse_llm_json)
from utils.prompting import (PromptOracle, queryLLM, queryLLM_batch,
                             set_api_keys)
from utils.scrape import url_to_filename

set_api_keys()

logger = logging.getLogger(__name__)

class QueryType(Enum):
    TEMPLATE = auto()
    GENERATE = auto()


class QueriesGenerator(): 
    def __init__(self, 
                 topic: str,
                 description: Optional[str] = None): 
        """Create a generator for the queries given an input topic as string."""
        self.topic = topic
        self.description = description

        # Queries template
        self.template = ("what is <TOPIC><SPLIT>"
                         "demystifying <TOPIC><SPLIT>"
                         "pros and cons of <TOPIC><SPLIT>"
                         "pros and cons of <TOPIC> filetype:pdf<SPLIT>"
                         "comparative analysis <TOPIC><SPLIT>"
                         "comparative analysis <TOPIC> filetype:pdf<SPLIT>"
                         "is <TOPIC> overrated? experts discuss pros and cons<SPLIT>"
                         "why <TOPIC> is not as great as people think<SPLIT>"
                         "critics review <TOPIC><SPLIT>"
                         "is <TOPIC> the future or just hype? pros and cons analysis<SPLIT>"
                         "pros and cons of <TOPIC> in the real world applications<SPLIT>"
                         "experts debate: <TOPIC> vs traditional approaches<SPLIT>"
                         "does <TOPIC> live up to the hype? in-depth analysis<SPLIT>"
                         "unpopular opinions about <TOPIC><SPLIT>"
                         "challenges and limitations of <TOPIC>")
        
        self.max_n_template = self.template.count("<SPLIT>") + 1         
        
    def _generate_queries_template(self, n: int):
        """Generate the queries for the Google Search from a template.
        """
        if n > self.max_n_template: 
            raise ValueError(f"{self.max_n_template} query templates are available, insert 'n' < {self.max_n_template}.")
        
        queries = self.template.replace("<TOPIC>", self.topic).split("<SPLIT>")[:n]
        return queries
    
    def _generate_queries_llm(self, 
                              n: int,
                              model: ChatOpenAI):
        """Generate a query list for the Google Search with a LLM.
        """
        # Model information in case of LLM-generated queries       
        start_time = time.time()
        prompt_llm = PromptOracle.get_queries_gen_prompt(n, self.topic, self.description)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        query_string = queryLLM(model, prompt_llm)
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens

        import os
        with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
            ff.write(f"(SEARCH) - A1: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
        queries = [q.strip() for q in query_string.split("-<Q>:")][1:]
        print("Queries generation took {:.2f} seconds".format(time.time() - start_time))
        for i, query in enumerate(queries):
            # Remove quotes at the beginning
            queries[i] = re.sub(r'^[\'"]+', '', query)

        self.prompt_llm = prompt_llm
        return queries

    def generate_queries(self, 
                         n: int,
                         type: QueryType,
                         model: Optional[ChatOpenAI]):
        """Generate the queries for the web search.

        Parameters
        ----------
        - n [`int`]: number of queries to be generated.
        - type [`QueryType`]: whether to use a static queries template or generate them with 
        an LLM
        - model_id [`str`, optional]: model to be used for the queries generation
        - temperature [`float`, optional]: temperature of the LLM

        Output
        ------
        - `list[str]`: list of 'n' query strings
        """
        if type == QueryType.TEMPLATE:
            logger.info("Generating the queries from the template.")
            queries = self._generate_queries_template(n)   
            return queries
        
        elif type == QueryType.GENERATE:
            if model is None:
                raise ValueError("Model not provided.")
            logger.info("Generating the queries with an agent")
            queries = self._generate_queries_llm(n, model)
            return queries

        else:
            raise ValueError("Unrecognized query type.")


class TopicDefiner():
    def __init__(self, model_name: str):
        self.model_name = model_name

    def define_topic(self, topic_path):
        if topic_path.endswith(".md"):
            print(f"Reading the md file: {topic_path} and generating the topic information")
            with open(topic_path, encoding="utf-8") as f:
                topic_prelim = f.read()

            def get_topic_info_md(topic_info):
                """Get the information that describes the topic.
                """
                fields = topic_info.split("##")[1:]

                tentative_title = [info for info in fields if "Tentative title" in info]
                if len(tentative_title) > 0:
                    tentative_title = tentative_title[0].replace("Tentative title", "").strip()
                    print(f"Topic tentative title: {tentative_title}")
                else:
                    tentative_title = None
                    print("No tentative title provided")

                interest = [info for info in fields if "Aspects of interest" in info]
                if len(interest) > 0:
                    interest = interest[0].replace("Aspects of interest", "").strip()
                    if interest.strip() == "":
                        interest = None
                        print("No aspects of interest provided")
                    else:
                        print(f"Aspects of interest: {interest}")
                else:
                    interest = None
                    print("No aspects of interest provided")

                no_interest = [info for info in fields if "Aspects not of interest" in info]
                if len(no_interest) > 0:
                    no_interest = no_interest[0].replace("Aspects not of interest", "").strip()
                    if no_interest.strip() == "":
                        no_interest = None
                        print("No aspects not of interest provided")
                    else:
                        print(f"Aspects not of interest: {no_interest}")
                else:
                    no_interest = None
                    print("No aspects not of interest provided")

                urls = [info for info in fields if "URLs" in info]
                if len(urls) > 0:
                    urls = urls[0].replace("URLs", "").strip()
                    urls = [url.strip() for url in urls.split("*") if url != ""]
                    urls = [normalize_arxiv_url(url) for url in urls]
                    print(f"URLs: {urls}")
                else:
                    urls = []
                    print("No URLs provided")

                return tentative_title, interest, no_interest, urls

            tentative_title, interest, no_interest, urls = get_topic_info_md(topic_prelim)
            info = [tentative_title, interest, no_interest]
            names = ["tentative title of the topic", "aspects of interest about the topic", 
                    "aspects not of interest about the topic"]
            provided_info = ", ".join([name for name, inf in zip(names, info) if inf is not None])
            provided_info_fields = "\n".join([f"# {name}\n{inf}" for name, inf in zip(names, info) if inf is not None])

            if (interest is None) and (no_interest is None):
                    print("No specific aspects of interest/no interest provided")
                    title = tentative_title
                    description = None
            else:
                print("Generating a description with an LLM")
                llm = ChatOpenAI(model_name=self.model_name,
                                 temperature=0,
                                 base_url=os.environ["BASE_URL"],
                                 api_key=os.environ["API_KEY"],
                                )
                response = queryLLM(llm, 
                                    PromptOracle.get_topic_definer_prompt(
                                        provided_info, 
                                        provided_info_fields
                                        )
                                    )
                description, title = response.split("Description: ")[1].split("Title: ")
                description = description.strip()
                title = title.strip()
        elif topic_path.endswith(".json"):
            print(f"Reading the json file: {topic_path}")
            with open(topic_path, encoding="utf-8") as f:
                json_file = json.load(f)
                print(json_file)
            description = json_file["topic_description"]
            title = json_file["topic"]
            urls = json_file["given_urls"]

        return description, title, urls



class SearchModule():
    def __init__(
            self,
            topic: str,
            topic_description: Optional[str],
            experiment_folder: str
            ):
        """Create a SearchModule object, that performs the search and 
        retrieves the most relevant documents on the input topic.
        
        Parameters
        ----------
        - topic [`str`]: title of the topic
        - topic_description [`str`|`None`]: description of the topic. 
        It can be `None` if only the title was provided.
        - experiment_folder [`str`]: path where to save the results 
        of the search
        """
        self.topic = topic
        self.topic_description = topic_description
        self.experiment_folder = experiment_folder

    def search(
            self,
            model: ChatOpenAI,
            reasoning_model: ChatOpenAI,
            num_queries: int,
            num_top_results: int,
            query_gen_type: QueryType,
            length_upper_filter: float,
            completeness_threshold: int,
            search_threshold: int,
            link_relevance_threshold: int,
            max_new_links: int,
            sim_threshold: float,
            tool: str,
            given_urls: list[str] = None,
            ) -> pd.DataFrame:
        """Perform the search.
        
        Parameters
        ----------
        - model [`ChatOpenAI`]: model for base agents
        - reasoning_model [`ChatOpenAI`]: model for reasoning agents
        - num_queries [`int`]: number of queries to be generated on the topic
        - num_top_results [`int`]: number of top results to extract per search
        - query_gen_type [`QueryType`]: query generation type. It can either follow a template, or use an agent to generate the queries.
        - length_upper_filter [`float`]: upper bound on the length of documents
        - completeness_threshold [`int`]: filtering threshold for the completeness score
        - search_threshold [`int`]: filtering threshold for the additional search score
        - link_relevance_threshold [`int`]: filtering threshold for the link relevance score
        - max_new_links [`int`]: maximum number of extracted links to be kept
        - sim_threshold [`float`]: filtering threshold for the documents' similarity
        - tool [`str`]: main parsing tool
        - given_urls [`list[str]`, default = None]: list of URLs to be inserted 
        into the initial data pool
        
        Output
        ------
        - `pd.DataFrame`: DataFrame with the collected resources
        """
        BASE_MODEL_ID = model.model_name
        BASE_MODEL_NAME = model.model_name.split("/")[-1]

        REASONING_MODEL_ID = reasoning_model.model_name
        REASONING_MODEL_NAME = reasoning_model.model_name.split("/")[-1]
        
        queries_generator = QueriesGenerator(
            topic=self.topic,
            description=self.topic_description
            )
        queries = queries_generator.generate_queries(
            n=num_queries,
            type=query_gen_type,
            model=model)
        
        # Save the queries to a file
        queries_path = os.path.join(self.experiment_folder, "queries")
        os.makedirs(queries_path, exist_ok=True)
        queries_file_path = os.path.join(queries_path, f"queries_list.txt")
        with open(queries_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(queries))

        # ** Conduct the search **
        # Retrieve the unique relevant URLs and assign the resource type.
        # If the user provided URLs, include them in the initial DataFrame.
        if len(given_urls) > 0:
            initial_df = pd.DataFrame({"url": given_urls, "query": "user-given"})
            logger.info(f"Including the {len(given_urls)} given URLs into the initial data pool {str(given_urls)}")
        else:
            initial_df = None

        df = retrieve_results_from_queries(
            queries=queries,
            n=num_top_results,
            df=initial_df
            )
        
        start_time = time.time()
        df["resource_type"] = df.apply(lambda row: assign_resource_type(row.url), axis=1)

        logger.info("Assigning the resource type took {:.2f} seconds".format(
            time.time() - start_time
        ))

        # -------------------------------------
        # Scraping and parsing to markdown
        # -------------------------------------
        # HTML download
        html_folder = os.path.join(self.experiment_folder, "html_documents")
        os.makedirs(html_folder, exist_ok=True)

        start_time = time.time()
        # Run the scraping script with the URLs as command-line arguments
        if len(df[df.resource_type == ResourceType.HTML]) > 0:

            logger.info(str(df[df.resource_type == ResourceType.HTML]))
            subprocess.run(["python", 
                            "utils/scrape.py", 
                            "--urls", *df[df.resource_type == ResourceType.HTML].url.values, 
                            "--output_dir", html_folder])
            logger.info("Finished scraping HTML websites in {:.2f} seconds".format(
                time.time() - start_time
            ))
        else:
            logger.info("No HTML websites to scrape")
        
        # Parsing with one or multiple tools
        start_time = time.time()
        df[f"content"] = df.apply(lambda row: convert(row.url, row.resource_type, html_folder, tool), axis=1)
        logger.info("Finished conversion to markdown in {:.2f} seconds".format(
            time.time() - start_time
        ))
        
        markdown_folder = os.path.join(self.experiment_folder, "markdown_documents")
        os.makedirs(markdown_folder, exist_ok=True)

        # Save the markdown content
        for ind, row in df.iterrows():
            with open(os.path.join(markdown_folder, 
                    f"{url_to_filename(row.url)}.md"), "w", encoding="utf-8") as g:
                g.write(row[f"content"])
        
        # Add some statistics about length and number of tokens of the texts
        start_time = time.time()


        df[f"number_tokens_{BASE_MODEL_NAME}"] = df.apply(lambda row: get_number_tokens(row[f"content"], BASE_MODEL_ID), axis=1)
        df["number_words"] = df.apply(lambda row: get_number_words(row["content"]), axis=1)
        df["number_chars"] = df.apply(lambda row: len(row["content"]), axis=1)
        logger.info("Length computation took {:.2f} seconds".format(
            time.time() - start_time
        ))

        df_folder = os.path.join(self.experiment_folder, "dataframes")
        os.makedirs(df_folder, exist_ok=True)
        df.to_csv(os.path.join(df_folder, f"1_initial_df.csv"))
        
        
        # Filter to avoid too long documents
        FILTERING_UPPER_QUANTILE = length_upper_filter

        if FILTERING_UPPER_QUANTILE <= 1:
            length_upper_bound =  df[f"number_tokens_{BASE_MODEL_NAME}"].quantile(FILTERING_UPPER_QUANTILE)
            logger.info(f"Filtering on the {FILTERING_UPPER_QUANTILE * 100}%: {length_upper_bound}")
        else:
            length_upper_bound = FILTERING_UPPER_QUANTILE
            logger.info(f"Filtering with a static threshold of {length_upper_bound}")

        logger.info(f"Removing {len(df[df[f"number_tokens_{BASE_MODEL_NAME}"] >= length_upper_bound])} resources out of {len(df)} because of excessive length")
        df_filtered = df[
            df[f"number_tokens_{BASE_MODEL_NAME}"] < length_upper_bound].copy()

        logger.info(f"Number of resources after length filtering: {len(df_filtered)}")

        
        # ** Completeness scoring **
        start_time = time.time()

        llm_queries = [PromptOracle.get_completeness_prompt(row.content) for i, row in df_filtered.iterrows()]

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens

        df_filtered['completeness_score'] = queryLLM_batch(model, llm_queries)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
            ff.write(f"(SEARCH) - A2: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        df_filtered.to_csv(os.path.join(df_folder, 
                                        "2_completeness_scoring.csv"))
        logger.info("Completeness scoring took {:.2f} seconds".format(time.time() - start_time))

        # ** Additional search scoring **
        start_time = time.time()

        llm_queries = [PromptOracle.get_additional_search_prompt(row.content) for i, row in df_filtered.iterrows()]

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens

        df_filtered['search_score'] = queryLLM_batch(model, llm_queries)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
            ff.write(f"(SEARCH) - A3: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        logger.info("Additional search scoring took {:.2f} seconds".format(time.time() - start_time))
        df_filtered.to_csv(os.path.join(df_folder,
                                        "3_additional_search.csv"))
        

        # Assign removal and search flags
        COMPLETENESS_THRESHOLD = completeness_threshold
        SEARCH_THRESHOLD = search_threshold
        df_filtered["to_be_removed"] = df_filtered.apply(lambda row: int(float(''.join(i for i in row.completeness_score if i.isdigit() or i == "."))) <= COMPLETENESS_THRESHOLD, axis=1)
        df_filtered["to_be_searched"] = df_filtered.apply(lambda row: int(float(''.join(i for i in row.search_score if i.isdigit() or i == "."))) > SEARCH_THRESHOLD, axis=1)
        
        # Additional search query generation
        indexes = df_filtered[df_filtered.to_be_searched == True].index
        start_time = time.time()

        df_filtered["additional_search_query"] = ""
        if len(indexes) > 0:
            llm_queries = [PromptOracle.get_additional_search_query_gen_prompt(row.content) for i, row in df_filtered.loc[indexes].iterrows()]


            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens

            df_filtered.loc[indexes, "additional_search_query"] = queryLLM_batch(model, llm_queries)

            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
                ff.write(f"(SEARCH) - A4: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

            logger.info("Query generation for the extra search for {} elements took {:.2f} seconds".format(len(indexes), time.time()-start_time))
        else:
            logger.info("No elements need an extra search step")
        
        df_filtered.to_csv(os.path.join(df_folder,
                                        "4_extra_queries_gen.csv"))
        
        
        # Remove the flagged URLs
        logger.info(f"Removing {len(df_filtered[df_filtered.to_be_removed == True])} resources out of {len(df_filtered)} because of completeness score < {COMPLETENESS_THRESHOLD}. Now there are {len(df_filtered[df_filtered.to_be_removed == False])} remaining ones.")
        df_filtered = df_filtered[df_filtered.to_be_removed == False]
        # Search according to the generated query
        queries = (
            df_filtered.loc[
                df_filtered["additional_search_query"].notna()
                & (df_filtered["additional_search_query"] != ""),
                "additional_search_query",
            ]
            .unique()
            .tolist()
        )
        logger.info(f"Extra search for {len(queries)} resources")
        if len(queries) > 0:
            df_tmp = retrieve_results_from_queries(
                queries=list(queries), 
                n=1
                )
            logger.info(f"Discarding {len(df_tmp[df_tmp.url.isin(df_filtered.url.values)])} searched resources out of {len(df_tmp)} because they are already in the DataFrame.")
            df_tmp = df_tmp[~df_tmp.url.isin(df_filtered.url.values)]
            if len(df_tmp) > 0:
                # Assign the resource type
                df_tmp["resource_type"] = df_tmp.apply(lambda row: assign_resource_type(row.url), axis=1)

                # Run the scraping script with the URLs as command-line arguments
                if len(df_tmp[df_tmp.resource_type == ResourceType.HTML]) > 0:
                    start_time = time.time()
                    logger.info(str(df_tmp[df_tmp.resource_type == ResourceType.HTML]))
                    subprocess.run(["python", 
                                    "utils/scrape.py", 
                                    "--urls", *df_tmp[df_tmp.resource_type == ResourceType.HTML].url.values, 
                                    "--output_dir", html_folder])
                    logger.info("Finished scraping HTML websites in {:.2f} seconds".format(
                        time.time() - start_time
                    ))
                else:
                    logger.info("No HTML websites to scrape")

                # Parsing
                start_time = time.time()
                df_tmp[f"content"] = df_tmp.apply(lambda row: convert(row.url, row.resource_type, html_folder, tool), axis=1)
                logger.info("Conversion of the new resources took {:.2f} seconds".format(
                    time.time() - start_time
                ))

                # Save the markdown content
                for ind, row in tqdm.tqdm(df_tmp.iterrows()):
                    with open(os.path.join(markdown_folder, 
                                        f"{url_to_filename(row.url)}.md"), "w", encoding="utf-8") as g:
                        g.write(row[f"content"])

                start_time = time.time()
                
                df_tmp[f"number_tokens_{BASE_MODEL_NAME}"] = df_tmp.apply(lambda row: get_number_tokens(row["content"], BASE_MODEL_ID), axis=1)
                df_tmp["number_words"] = df_tmp.apply(lambda row: get_number_words(row["content"]), axis=1)
                df_tmp["number_chars"] = df_tmp.apply(lambda row: len(row["content"]), axis=1)
                logger.info("Length computation for the new resources took {:.2f} seconds".format(
                    time.time() - start_time
                ))

                logger.info(f"Removed {len(df_tmp[df_tmp[f"number_tokens_{BASE_MODEL_NAME}"] >= length_upper_bound])} out of {len(df_tmp)} because of length higher than {length_upper_bound}")
                df_tmp = df_tmp[df_tmp[f"number_tokens_{BASE_MODEL_NAME}"] < length_upper_bound].copy()

            if len(df_tmp) > 0:
                # Completeness scoring
                start_time = time.time()
                llm_queries = [PromptOracle.get_completeness_prompt(row.content) for i, row in df_tmp.iterrows()]
                df_tmp['completeness_score'] = queryLLM_batch(model, llm_queries)
                logger.info("Completeness scoring of the {} new resources took {:.2f} seconds".format(len(df_tmp), time.time()-start_time))

                df_tmp["to_be_removed"] = df_tmp.apply(lambda row: int(float(''.join(i for i in row.completeness_score if i.isdigit() or i == "."))) <= COMPLETENESS_THRESHOLD, axis=1)
                
                # Keep only the complete documents, which will be added to the main dataframe
                logger.info(f"Removing {len(df_tmp[df_tmp.to_be_removed == True])} extra search resources because of uncompleteness")
                df_tmp_to_append = df_tmp[df_tmp.to_be_removed == False]
                df_complete = pd.concat((df_filtered, df_tmp_to_append), ignore_index=True)
            else:
                df_complete = df_filtered.copy()
                
        else:
            logger.info("No extra searches")
            df_complete = df_filtered.copy()
        

        logger.info(f"Number of resources after completeness and extra search: {len(df_complete)}")
        df_complete.to_csv(os.path.join(df_folder, 
                                        "5_expanded_search.csv"))
        
        # -------------------------------------
        # Embedded links extraction and scoring
        # -------------------------------------

        @backoff.on_exception(
                    backoff.constant,
                    Exception,
                    interval=0,               # can also use (openai.error.OpenAIError,) if more specific
                    max_tries=3,               # try at most 3 times
                )
        def embedded_links_extraction_scoring(llm, df_complete, topic):
            prompt_llm = [PromptOracle.get_embedded_links_extraction_prompt(row.content) for i, row in df_complete.iterrows()]
            start_time = time.time()
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            links_extracted = queryLLM_batch(llm, prompt_llm)
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
                ff.write(f"(SEARCH) - A5: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            logger.info("Links extraction from the {} resources took {:.2f} seconds".format(
                len(df_complete), time.time()-start_time))
            
            # `links_extracted` is a list, where each element is a string containing multiple links separated by '\n'
            prompt_llm = []  # List of prompts where each element is the index of a links_extracted position where there was at least one link
            prompt_indexes = []  # List of indexes of source URLs with at least one embedded link
            links_ids = []

            for index in range(len(links_extracted)):
                # If the page had no embedded links, do not create any prompt
                if links_extracted[index] == "<NONE>":
                    continue

                prompt_indexes.append(index)
                links_list = links_extracted[index].split("\n")

                existing_urls = set(df_complete.url.values)
                # If it was already found in the first retrieval step
                links_list = [link for link in links_list if link not in existing_urls]

                links_dict = {}
                # For each link extracted from a certain resource, fix the URL and add 
                # it to a dictionary referring to this resource only
                for j, link in enumerate(links_list):
                    links_dict[j] = fix_url(link, df_complete.iloc[index].url)
                    if link != links_dict[j]:
                        print(link, " -->", links_dict[j])
                links_ids.append(links_dict)  # add the dictionary to a list
                
                prompt_llm.append(
                    PromptOracle.get_embedded_links_relevance_prompt(
                        df_complete.iloc[index].content,
                        str(links_dict), 
                        topic))

            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            start_time = time.time()
            links_scores = queryLLM_batch(llm, prompt_llm)  
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
                ff.write(f"(SEARCH) - A6: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            logger.info("Links relevance scoring took {:.2f} seconds".format(time.time()-start_time))
            
            # `links_scores` is a list of strings, where each string is a dictionary with link id as key 
            # for that specific resource and the score as value: {<link_id>: <score>}
            i = 0
            final_dict = {}
            source_urls = [] 
            for d1, d2 in zip(links_ids, links_scores):
                d2 = parse_llm_json(d2)  # d2 {<link_id>: <score>}
                for key in d1: # d1: {<link_id>: <URL>}
                    # Check if link is already in the dictionary because extracted from another resource. 
                    # The check with respect to the original dataframe has already been done after the extraction
                    if (d1[key] not in final_dict) and (key in d2):  
                        final_dict.update({d1[key]: d2[key]})
                        source_urls.append(df_complete.iloc[prompt_indexes[i]].url)
                        assert len(final_dict) == len(source_urls)
                i += 1
            return final_dict, source_urls
        
        final_dict, source_urls = embedded_links_extraction_scoring(model, df_complete, self.topic)

        # Map to arXiv pdfs
        for key in list(final_dict.keys()):
            if "arxiv.org" in key:
                new_key = normalize_arxiv_url(key)
                final_dict[new_key] = final_dict.pop(key)
        links_df = pd.DataFrame(list(final_dict.items()), columns=['url', 'link_relevance_score'])

        links_df["source_url"] = source_urls
        links_df.to_csv(os.path.join(df_folder, f"__emblinks__.csv"))
        
        logger.info(f"Extracted {len(links_df)} links")
        logger.info(f"Scores distribution: {str(Counter(links_df.link_relevance_score))}")
        
        # Filter out the unrelevant links and consider a maximum number of 
        # extra resources
        LINKS_RELEVANCE_THRESHOLD = link_relevance_threshold
        MAX_NEW_RESOURCES = max_new_links
        logger.info(f"Keeping {len(links_df[links_df.link_relevance_score >= LINKS_RELEVANCE_THRESHOLD])} links out of {len(links_df)} with threshold {LINKS_RELEVANCE_THRESHOLD}")
        links_df = links_df[links_df.link_relevance_score >= LINKS_RELEVANCE_THRESHOLD].copy()
        logger.info(f"Keeping only the top {min(MAX_NEW_RESOURCES, len(links_df))} resources out of {len(links_df)}." )
        links_df = links_df.sort_values(by="link_relevance_score", ascending=False)[:MAX_NEW_RESOURCES]
        # Fix the urls
        # assert (links_df.url.duplicated().sum() == 0)
        links_df = links_df[~links_df.url.duplicated()].reset_index(drop=True)
        logger.info(f"Removing {len(links_df[links_df.url.isin(df_complete.url)])} as they are already in the DataFrame.")
        links_df = links_df[~links_df.url.isin(df_complete.url)]

        if len(links_df) > 0:
            # Assign the resource type
            links_df["resource_type"] = links_df.apply(lambda row: assign_resource_type(row.url), axis=1)

            # Run the scraping script with the URLs as command-line arguments
            if len(links_df[links_df.resource_type == ResourceType.HTML]) > 0:

                logger.info(str(links_df[links_df.resource_type == ResourceType.HTML]))
                start_time = time.time()
                subprocess.run(["python", 
                                "utils/scrape.py", 
                                "--urls", *links_df[links_df.resource_type == ResourceType.HTML].url.values, 
                                "--output_dir", html_folder])
                logger.info("Finished scraping HTML websites in {:.2f} seconds".format(
                    time.time() - start_time
                ))
            else:
                logger.info("No HTML websites to scrape")
            
            # Parsing with one or multiple tools
            start_time = time.time()
            links_df[f"content"] = links_df.apply(lambda row: convert(row.url, row.resource_type, html_folder, tool), axis=1)
            logger.info("Conversion of the new resources (from links) took {:.2f} seconds".format(
                time.time() - start_time
            ))
            
            # Save the markdown content
            for ind, row in tqdm.tqdm(links_df.iterrows()):
                with open(os.path.join(markdown_folder, f"{url_to_filename(row.url)}.md"), "w", encoding="utf-8") as g:
                    g.write(row["content"])
            
            # Add some statistics about length and number of tokens of the texts
            links_df[f"number_tokens_{BASE_MODEL_NAME}"] = links_df.apply(lambda row: get_number_tokens(row["content"], BASE_MODEL_ID), axis=1)
            links_df["number_words"] = links_df.apply(lambda row: get_number_words(row["content"]), axis=1)
            links_df["number_chars"] = links_df.apply(lambda row: len(row["content"]), axis=1)

            
            # Filter to avoid empty documents and huge prompts
            logger.info(f"Removing {len(links_df[links_df[f"number_tokens_{BASE_MODEL_NAME}"] >= length_upper_bound])} resources out of {len(links_df)} because of length higher than {length_upper_bound}")
            links_df = links_df[
                links_df[f"number_tokens_{BASE_MODEL_NAME}"] < length_upper_bound].copy()
            length_lower_bound = 200
            logger.info(f"Removing {len(links_df[links_df[f"number_tokens_{BASE_MODEL_NAME}"] <= length_lower_bound])} resources out of {len(links_df)} because of length lower than {length_lower_bound}")
            links_df = links_df[
                links_df[f"number_tokens_{BASE_MODEL_NAME}"] > length_lower_bound].copy()
            
            # Completeness scoring
            start_time = time.time()
            llm_queries = [PromptOracle.get_completeness_prompt(row.content) for i, row in links_df.iterrows()]
            links_df['completeness_score'] = queryLLM_batch(model, llm_queries)
            logger.info("Completeness scoring for the {} new resources took {:.2f} seconds".format(len(llm_queries), time.time()-start_time))
            
            links_df.to_csv(os.path.join(df_folder, 
                                        "6_with_embedded_links.csv"))
            

        if len(links_df) > 0:
            # Assign removal and search flags
            links_df["to_be_removed"] = links_df.apply(lambda row: int(float(''.join(i for i in row.completeness_score if i.isdigit() or i == "."))) <= COMPLETENESS_THRESHOLD, axis=1)

            # Remove the flagged URLs
            logger.info(f"Removing {len(links_df[links_df.to_be_removed == True])} out of {len(links_df)} because of uncompleteness")
            links_df = links_df[links_df.to_be_removed == False]

        
        final_df = pd.concat((df_complete, links_df), ignore_index=True)
        final_df.to_csv(os.path.join(df_folder, 
                                    "7_collected_resources.csv"))
        
        
        # Remove the embedded links from the markdown contents
        final_df["content"] = final_df.apply(lambda row: remove_embedded_links(row.content), axis=1)
        logger.info(f"Finished data collection: {len(final_df)} resources, average length of {final_df[f"number_tokens_{BASE_MODEL_NAME}"].mean()}, total length of {final_df[f"number_tokens_{BASE_MODEL_NAME}"].sum()}")

        # -------------------------------------
        # Duplicates removal
        # -------------------------------------
        # Filter the arxiv links to keep only the latest version of the same paper
        final_df["url"] = final_df["url"].apply(normalize_arxiv_url)
        removed_counter = len(final_df)
        final_df = final_df[final_df.url.isin(filter_arxiv_url(final_df.url))]
        removed_counter -= len(final_df)
        logger.info((f"Removed {removed_counter} "
                    "resources as they are not the most recent arXiv version"))
        
        # Compute the similarity 
        sim_dict = {}
        sim_evaluator = TextSimilarityEvaluator()

        start_time = time.time()
        for pair in combinations(final_df.index, 2):
            pair = sorted(pair)
            # We operate on the index of the dataframe, not on the position
            similarity_value = sim_evaluator.jaccard_similarity(final_df.loc[pair[0]].content,
                                                                final_df.loc[pair[1]].content)
            if pair[0] not in sim_dict:
                sim_dict[pair[0]] = {}
            sim_dict[pair[0]][pair[1]] = similarity_value

        logger.info("Similarity computation for duplicates removal took {:.2f} seconds".format(
            time.time() - start_time
        ))

        # Plot the heatmap of the similarity scores
        sim_df = pd.DataFrame(sim_dict)
        fig, ax = plt.subplots(1, 1, figsize=(30,30))
        sns.heatmap(sim_df, ax=ax, annot=True, fmt=".2f", annot_kws={"fontsize":6})
        fig.tight_layout()
        plot_folder = os.path.join(self.experiment_folder, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        fig.savefig(os.path.join(plot_folder, "Heatmap_similarity_duplicates.svg"), 
                    format='svg', 
                    transparent=True)
        sim_df.to_csv(os.path.join(df_folder, "_similarity_duplicates.csv"))
        similar_elems = [
            (row_idx, col_idx, sim_df.loc[row_idx, col_idx])
            for (row_idx, col_idx), value in sim_df.stack().items()
            if value > sim_threshold
        ]
        pairs = [(row, col) for row, col, _ in similar_elems]

        indexes_to_remove = []
        for pair in pairs:
            index_doc1 = pair[0]
            index_doc2 = pair[1]
            len_doc1 = len(final_df.loc[index_doc1].content)
            len_doc2 = len(final_df.loc[index_doc2].content)
            
            if len_doc1 > len_doc2:
                indexes_to_remove.append(index_doc2)
                logger.info(f"Keeping resource {final_df.loc[index_doc1].url} of length {len_doc1} instead of {final_df.loc[index_doc2].url} of length {len_doc2}")
            else:
                indexes_to_remove.append(index_doc1)
                logger.info(f"Keeping resource {final_df.loc[index_doc2].url} of length {len_doc2} instead of {final_df.loc[index_doc1].url} of length {len_doc1}")
        
        logger.info(f"Removing {len(pairs)} resources as they are flagged as duplicates")
        final_df = final_df.drop(indexes_to_remove)

        final_df.to_csv(os.path.join(df_folder, 
                                    "8_duplicates_removed.csv"))
        
        
        # -------------------------------------
        # Summarize the resources
        # -------------------------------------
        generate_identifier=True
        prompt_llm = [PromptOracle.get_summarize_prompt(
            row.content, 
            self.topic,
            self.topic_description, 
            generate_identifier=generate_identifier,
            url=row.url) for i, row in final_df.iterrows()]
        start_time = time.time()

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens

        summaries_llm = queryLLM_batch(model, prompt_llm)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
            ff.write(f"(SEARCH) - A7: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        logger.info("Summarization took {:.2f} seconds".format(time.time()-start_time))

        if generate_identifier == False:
            final_df["summary"] = summaries_llm
        else:
            try:
                summaries = [summ.replace("**", "").split("### Summary")[1].split("###")[0].strip() for summ in summaries_llm]
                identifiers = [summ.replace("**", "").split("### Identifier")[1].strip() for summ in summaries_llm]
                final_df["summary"] = summaries
                final_df["identifier"] = identifiers
            except Exception as e:
                print(e)

        summaries_folder = os.path.join(self.experiment_folder, "summaries")
        os.makedirs(summaries_folder, exist_ok=True)
        for ind, row in final_df.iterrows():
            with open(os.path.join(summaries_folder,
                                f"{url_to_filename(row.url)}.md"), "w", encoding="utf-8") as g:
                g.write(row.summary)
                g.write(f"\n\nIdentifier: {row.identifier}")

        # Add some statistics about length and number of tokens of the texts
        final_df[f"number_tokens_summary_{BASE_MODEL_NAME}"] = final_df.apply(lambda row: get_number_tokens(row[f"summary"], BASE_MODEL_ID), axis=1)

        final_df.to_csv(os.path.join(df_folder, 
                                    "9_summaries.csv"))
        
    
        # -------------------------------------
        # LLM selection for relevance filtering
        # -------------------------------------
        logger.info(f"Using model {REASONING_MODEL_NAME} to evaluate the relevance")

        final_df = self._filter_by_relevance(reasoning_model, final_df, self.topic, self.topic_description)
        final_df.to_csv(os.path.join(df_folder,
                                    "10_relevant.csv"))

        final_df.reset_index(inplace=True)
        return final_df
    
    def load_research(
            self,
            research_path: str
            ) -> pd.DataFrame:
        """Load the DataFrame of an already conducted research.
        
        Parameters
        ----------
        - research_path [`str`]: path to the DataFrame
        
        Output
        ------
        - `pd.DataFrame`: the loaded research DataFrame
        """
        logger.info(f"Loading the research from '{research_path}'")
        final_df = pd.read_csv(research_path, index_col=0)
        df_folder = os.path.join(self.experiment_folder, "dataframes")
        os.makedirs(df_folder, exist_ok=True)
        final_df.to_csv(os.path.join(df_folder,
                                    "final_df.csv"))
        final_df.reset_index(inplace=True)
        return final_df
    
    def _filter_by_relevance(self, llm, df, topic, topic_descr=None):  
        prompt = PromptOracle.get_relevance_prompt(df.summary.values, topic, topic_descr)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens

        output = queryLLM(llm, prompt)

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a", encoding="utf-8") as ff:
            ff.write(f"(SEARCH) - A8: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        # Extract the indexes of the documents to be removed
        # NOTE: the integer positions are considered, not the indexes themselves
        # Using a set in case before casting to deal with possible LLM output
        # format inconsistencies
        logger.info(f"Output relevance filtering:\n{output}")

        if "</think>" in output:
            output = output.split("</think>")[1]
        if "Outliers" in output:
            output = output.split("Outliers")[-1]
        indexes_to_remove = re.findall(r'Document (\d+)', output)
        indexes_to_remove = [int(index) for index in set(indexes_to_remove)]  
        logger.info(f"Indexes to remove: {indexes_to_remove}")
        logger.info(f"Removing {len(indexes_to_remove)} resources out of {len(df)} as they are not relevant for the topic")

        return df.drop(df.index[indexes_to_remove])
