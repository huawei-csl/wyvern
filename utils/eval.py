import copy
import logging
import os
import re
from enum import Enum, auto
from typing import Optional

import langchain_openai
import numpy as np
import pandas as pd
from nltk import sent_tokenize

from utils.general import (MarkdownTableHandler, find_last_report_version,
                           generate_subtables, get_number_tokens,
                           remove_citations)
from utils.prompting import PromptOracle, queryLLM, queryLLM_batch

logger = logging.getLogger(__name__)


DEFAULT_GROUNDING_MAX_INPUT_TOKENS = 60000
DEFAULT_GROUNDING_CHUNK_OVERLAP_TOKENS = 1000


def _parse_entailment_result(result: str) -> int:
    """Convert the grounding model's Yes/No response to a binary integer result.
    """
    answer = result.split("</think>", 1)[1].strip() if "</think>" in result else result
    answer = re.sub(r'[_*~`[\](){}#+!\-\\]', '', answer.strip())
    return int(answer.startswith("Yes"))


def _find_supporting_result(results: list[str]) -> Optional[str]:
    """Return the first fully supporting result, implementing logical OR.
    """
    return next(
        (result for result in results if _parse_entailment_result(result) == 1),
        None,
    )


def _split_text_with_overlap(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split text into bounded chunks, trying to select paragraph boundaries.
    """
    if max_chars <= 0:
        raise ValueError("`max_chars` must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("`overlap_chars` must be non-negative and smaller than `max_chars`")
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end

        if hard_end < len(text):
            # Avoid producing a very small chunk just to respect an early newline
            minimum_break = start + int(max_chars * 0.8)
            paragraph_end = text.rfind("\n\n", minimum_break, hard_end)
            line_end = text.rfind("\n", minimum_break, hard_end)
            end = max(paragraph_end + 2, line_end + 1)  # choose the last available boundary
                                                        # and include its newline characters
                                                        # in the chunk
            if end <= start:
                end = hard_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _build_grounding_chunks(
    claim: str,
    references_text: str,
    model_id: str,
    max_input_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Create chunks for the grouding check
    """
    empty_reference_prompt = PromptOracle.get_entail_prompt(claim, "", False)
    prompt_overhead = get_number_tokens(
        empty_reference_prompt,
        model_id,
    )
    available_tokens = max_input_tokens - prompt_overhead
    if available_tokens <= 0:
        raise ValueError(
            "`GROUNDING_MAX_INPUT_TOKENS` is too small for the grounding prompt and claim"
        )
    if overlap_tokens >= available_tokens:
        raise ValueError(
            "`GROUNDING_CHUNK_OVERLAP_TOKENS` must be smaller than the available "
            "grounding reference budget"
        )

    # Use a six-characters-per-token approximation for the reference and overlap.
    return _split_text_with_overlap(
        references_text,
        max_chars=available_tokens * 6,
        overlap_chars=overlap_tokens * 6,
    )


class ClaimGranularityLevel(Enum):
    SENTENCE = auto()
    PARAGRAPH_UP_TO_CITATIONS = auto()
    PARAGRAPH = auto()


# The following code has been adapted from:
# https://github.com/stanford-oval/storm/blob/NAACL-2024-code-backup
class CitationsEvaluator():
    def __init__(self,
                 llm_nli: langchain_openai.ChatOpenAI,
                 llm_claims: langchain_openai.ChatOpenAI):
        """Evaluator of the citations recall and precision of a report.
        
        Parameters
        ----------
        - llm_nli [`langchain_openai.ChatOpenAI`]: model to use for the NLI
        - llm_claims [`langchain_openai.ChatOpenAI`]: model to use for the 
        decomposition into atomic claims
        """
        self.llm_nli = llm_nli
        self.llm_claims = llm_claims

    def _run_nli_autoais(self,
                         references_text: str,
                         claim: str,
                         partial: bool):
        """ 
        Assess whether the claim is attributable to identified sources (AIS). 

        Parameters
        ----------
        - references_text [`str`]: text of the references. If multiple ones are 
                                considered, a concatenation can be done
        - claim [`str`]: the claim which is under assessment
        - partial [`bool`]: if True, evaluate if the claim is at least partially
                            supported by the sources. If False, the core part of the
                            claim must be supported
        Output
        ------
        - `int`: it takes value 0 if the claim is not supported, 1 otherwise
        """
        inference = 0
        prompt = PromptOracle.get_entail_prompt(claim, references_text, partial)
        res = queryLLM(self.llm_nli, prompt)

        if "</think>" in res:
            if re.sub(r'[_*~`[\](){}#+!\-\\]', '', res.split("</think>")[1].strip()).startswith('Yes'):
                inference = 1
        else:
            if re.sub(r'[_*~`[\](){}#+!\-\\]', '', res).startswith('Yes'):
                inference = 1
                
        return inference


    def evaluate(self,
                 report_text: str,
                 references_df: pd.DataFrame,
                 granularity_level: ClaimGranularityLevel):
        """Compute the citation precision and recall for a given report.
        
        Parameters
        ----------
        - report_text [`str`]: report to be evaluated
        - references_df [`str`]: dataframe containing the references
        - granularity_level [`ClaimGranularityLevel`]: granularity criterion for
        the tokenization into claims
        
        Output
        ------
        - [`float`]: citation recall
        - [`float`]: citation precision
        - [`int`]: number of supported claims
        - [`int`]: number of claims without any reference
        - [`int`]: number of claims for which at least one reference
                   was not found in the DB
        - [`int`]: number of logically unsupported claims
        """
        logger.info(f"Evaluating the citations quality with {self.llm_nli.model_name}")

        # Keep track of the total number of input and output tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        sent_no_citations = 0
        sent_mcite = 0
        sent_mcite_support = 0
        sent_mcite_overcite = 0
        sent_ref_not_found = 0
        sent_failed_support = 0

        eval_log = []
        entail = 0
        entail_prec = 0
        total_citations = 0
        citations_no_recall = 0

        logger.info(f"Selected claim granularity level: {str(granularity_level)}")

        if granularity_level == ClaimGranularityLevel.SENTENCE:
            # Use NLTK to get the sentences
            sentences = sent_tokenize(report_text)

        # Consider group of sentences divided by either citations or end of paragraph
        elif granularity_level == ClaimGranularityLevel.PARAGRAPH_UP_TO_CITATIONS:
            paragraphs_list = list(filter(bool, report_text.splitlines()))
            sentences = []
            # Consider each paragraph separately. Keep a flag and a memory variable
            # to keep accumulating sentences if they do not contain any citation
            for paragraph in paragraphs_list:
                CONTINUATION = False
                MEMORY_CLAIMS = ""
                sentences_par = sent_tokenize(paragraph)
                for sentence_par_id, sentence_par in enumerate(sentences_par):
                    refs = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sentence_par)]
                    if len(refs) > 0:
                        if CONTINUATION:
                            sentence_par = MEMORY_CLAIMS + sentence_par
                            CONTINUATION = False
                            MEMORY_CLAIMS = ""
                        sentences.append(sentence_par)
                    else:
                        if sentence_par_id == (len(sentences_par) - 1):
                            if CONTINUATION:
                                sentence_par = MEMORY_CLAIMS + sentence_par
                                CONTINUATION = False
                                MEMORY_CLAIMS = ""
                            sentences.append(sentence_par)
                        else:
                            CONTINUATION = True
                            MEMORY_CLAIMS += (sentence_par + " ")

        # Consider each claim to be a paragraph
        elif granularity_level == ClaimGranularityLevel.PARAGRAPH:
            sentences = list(filter(bool, report_text.splitlines()))

        else:
            raise ValueError(f"Unsupported ({str(granularity_level)}) claim tokenization level!")


        # Iterate over all the sentences in the report        
        for sentence_id, sentence in enumerate(sentences):
            joint_entail = -1  # Undecided

            # Extract the references from the sentence. NOTE: the '-1' is beacuse in 
            # the report the citation count starts from 1 and not 0
            refs = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sentence)]  
            logger.info(f"For `{sentence}`, find citations {refs}")
            # For the claim to be supported the citations set must be of at least
            # cardinality 1
            if len(refs) == 0:
                joint_entail = 0
                sent_no_citations += 1
                logger.info(f"[Unsupported sentence][No citations] {sentence}")
            # The citation does not refer to any reference. This check is not sufficient
            # to ensure the citation is correct.
            elif any([ref_id not in references_df.index for ref_id in refs]):
                joint_entail = 0
                nonexistent_refs = [str(ref_id) for ref_id in set(refs) if ref_id not in references_df.index]
                nonexistent_refs_str = "[" + ",".join(nonexistent_refs) + "]"
                logger.info(f"[Unsupported sentence][Reference id not found] {nonexistent_refs_str}")
                sent_ref_not_found += 1
            else:
                total_citations += len(refs)
                # Concatenate the texts to have the whole references content
                joint_ref_passages = '\n'.join(
                    [references_df.loc[ref_id]['summary'] for ref_id in set(refs)]
                )

            # If not directly rejected by citation format error, 
            # calculate the recall binary score: it is 1 if the claim is entirely 
            # supported by the cited references, 0 otherwise
            if joint_entail == -1:
                joint_entail = self._run_nli_autoais(references_text=joint_ref_passages, 
                                                     claim=sentence, 
                                                     partial=False)
                if joint_entail == 0:
                    logger.info(f'[Unsupported sentence][Claim not supported] {sentence}')
                    sent_failed_support += 1

            entail += joint_entail
            if len(refs) > 1:
                sent_mcite += 1

            unnecessary_citations = []

            # Calculate the citation precision 
            # (the claim must be supported by the references to continue)
            if joint_entail and len(refs) > 1:
                sent_mcite_support += 1
                flag = 0  # Keep track of the number of sentences with overcitations

                # Consider each reference, and check whether it is irrelevant
                for ref_id in refs:
                    # condition A: the reference must support the claim, otherwise 
                    #              it can be instantly classified as irrelevant
                    ref_text = references_df.loc[ref_id]["summary"]
                    nli_result = self._run_nli_autoais(references_text=ref_text, 
                                                claim=sentence, 
                                                partial=True)

                    # condition B: evaluated only if the claim is not supported by the
                    #              reference 'ref_id' alone. Otherwise, the citation for
                    #              sure is relevant.
                    if not nli_result:
                        # Consider all the cited references except for 'ref_id'
                        subset_exclude = set(copy.deepcopy(refs))
                        subset_exclude.remove(ref_id)
                        subset_exclude_text = '\n'.join(
                            [references_df.loc[ref_id_excluded]["summary"] for ref_id_excluded in subset_exclude]
                        )
                        nli_result = self._run_nli_autoais(references_text=subset_exclude_text,
                                                           claim=sentence,
                                                           partial=False)
                        # If the claim is supported, it means that 'ref_id' was not
                        # fundamental to support the claim, thus it is an irrelevant
                        # citation
                        if nli_result:
                            if flag == 0:
                                sent_mcite_overcite += 1
                            else: 
                                flag = 1
                            logger.info(f'[Unnecessary citation] sent: {sentence} citation: [{ref_id}]')
                            unnecessary_citations.append(ref_id)
                        # The claim was not supported, thus the reference is relevant
                        else:
                            entail_prec += 1
                    # The claim is supported by the reference 'ref_id' itself. Thus
                    # the reference is relevant.
                    else:
                        entail_prec += 1

            # Two options to enter here:
            # - the claim is not supported by all the references ('joint_entail' = 0).
            #   In this case, we are sure the precision cannot be 1, so we have a sum 
            #   with 0
            # - the claim is supported:
            #       * If there is only one reference, then 'joint_entail' is 1, so we
            #         have precision equal to 1
            #       * If there are zero references, then 'joint_entail' is 0, so we 
            #         have precision equal to 0
            else:
                entail_prec += joint_entail
                if joint_entail == 0:
                    citations_no_recall += len(refs)

            eval_log.append({
                "sent": sentence,
                "sentence_no_refs": remove_citations(sentence),
                "refs": refs,
                "joint_entail": joint_entail,
                "unnecessary_citations": unnecessary_citations,
            })

        number_sentences = len(sentences)
        logger.info("Of all the {} sentences, {:.2f}% have no citations".format(
            number_sentences, 100 * float(sent_no_citations)/number_sentences
        ))
        logger.info(f"Of all the {number_sentences} sentences, {sent_mcite} have at least two citations and are supported by the references (i.e., support for the computation of the citation precision) ")
        # If there are sentences with multiple citations, and if there is at least one
        # sentence with multiple citations which is supported by them
        if sent_mcite > 0 and sent_mcite_support > 0:
            logger.info(
                "Among all sentences, {:.2f}% has multiple citations:\n\t-{:.2f}% of them are supported by the joint set\n\t-{:.2f}% of them are overcited.".format(
                    100 * sent_mcite / number_sentences,
                    100 * sent_mcite_support / sent_mcite,
                    100 * sent_mcite_overcite / sent_mcite_support
                )
            )

        logger.info(f"Total citations: {total_citations}, citations with no recall: {citations_no_recall}, irrelevant citations: {sent_mcite_overcite}, relevant citations: {entail_prec}")

        citation_rec = 100 * entail / number_sentences
        citation_prec = 100 * (entail_prec / total_citations if total_citations > 0 else 0)

        logger.info("Citation recall: {:.2f}".format(citation_rec))
        logger.info("Citation precision: {:.2f}".format(citation_prec))

        logger.info("The citation recall computed with the number of supported claims {} and excluding the sentences with no citations ({}) is {:.2f}%".format(
            entail, sent_no_citations, float(entail)/(number_sentences - sent_no_citations)* 100
        ))
 
        return citation_rec, citation_prec, entail, sent_no_citations, sent_ref_not_found, sent_failed_support


    def evaluate_atomic_claims(self,
                               report_text: str,
                               references_df: pd.DataFrame,
                               granularity_level: ClaimGranularityLevel,
                               load_claims: Optional[str] = None,
                               eval_folder: Optional[str] = None,
                               evaluate_precision: Optional[bool] = False):
        """Compute the citation recall for a given report doing a decomposition
        of text units into atomic claims.
        
        Parameters
        ----------
        - report_text [`str`]: report to be evaluated
        - references_df [`str`]: dataframe containing the references
        - granularity_level [`ClaimGranularityLevel`]: granularity criterion for
        the tokenization into claims
        - load_claims [`str`]: path of the `pd.DataFrame` containing the claims
        - eval_folder [`str`]: path of the folder where to save the results
        - evaluate_precision [`bool`]: whether to evaluate also the citations 
        precision, in addition to the recall
        
        Output
        ------
        - [`float`]: citation recall
        - [`int`]: number of supported claims
        - [`int`]: number of claims without any reference
        - [`int`]: number of claims for which at least one reference
                   was not found in the DB
        - [`int`]: number of logically unsupported claims
        - [`pd.DataFrame`]: claims dataframe
        - [`pd.DataFrame`]: precision dataframe
        """
        logger.info(f"Evaluating the citations quality with {self.llm_nli.model_name}")
        
        sent_no_citations = 0
        sent_ref_not_found = 0
        sent_failed_support = 0

        entail = 0
        total_citations = 0

        logger.info(f"Selected claim granularity level: {str(granularity_level)}")

        tab_handler = MarkdownTableHandler()
        tab = tab_handler.extract_markdown_tables(report_text)
        # NOTE: we assume it is at most one
        if len(tab) > 0:
            tab = tab[0]
        else:
            tab = None
        report_text = tab_handler.hide_tables(report_text)

        if granularity_level == ClaimGranularityLevel.SENTENCE:
            # Use NLTK to get the sentences
            sentences = sent_tokenize(report_text)
            sentences = [s.strip() for s in sentences if s.strip()!= ""]

        # Consider group of sentences divided by either citations or end of paragraph
        elif granularity_level == ClaimGranularityLevel.PARAGRAPH_UP_TO_CITATIONS:
            paragraphs_list = list(filter(bool, report_text.splitlines()))
            sentences = []
            # Consider each paragraph separately. Keep a flag and a memory variable
            # to keep accumulating sentences if they do not contain any citation
            for paragraph in paragraphs_list:
                CONTINUATION = False
                MEMORY_CLAIMS = ""
                sentences_par = sent_tokenize(paragraph)
                for sentence_par_id, sentence_par in enumerate(sentences_par):
                    refs = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sentence_par)]
                    if len(refs) > 0:
                        if CONTINUATION:
                            sentence_par = MEMORY_CLAIMS + sentence_par
                            CONTINUATION = False
                            MEMORY_CLAIMS = ""
                        sentences.append(sentence_par)
                    else:
                        if sentence_par_id == (len(sentences_par) - 1):
                            if CONTINUATION:
                                sentence_par = MEMORY_CLAIMS + sentence_par
                                CONTINUATION = False
                                MEMORY_CLAIMS = ""
                            sentences.append(sentence_par)
                        else:
                            CONTINUATION = True
                            MEMORY_CLAIMS += (sentence_par + " ")

            sentences = [s.strip() for s in sentences if s.strip()!= ""]

        # Consider each claim to be a paragraph
        elif granularity_level == ClaimGranularityLevel.PARAGRAPH:
            sentences = list(filter(bool, report_text.splitlines()))
            sentences_tmp = [s.strip() for s in sentences if s.strip()!= ""]
            sentences = []
            for i, sent in enumerate(sentences_tmp):
                if len(list(set([int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]))) > 5:
                    logger.info(f"The text unit '{sent}' has too many citations, breaking it down on sentences")
                    sentences.extend(sent_tokenize(sent))
                else:
                    sentences.append(sent)

        else:
            raise ValueError(f"Unsupported ({str(granularity_level)}) claim tokenization level!")
             
        if tab is not None:
            sentences.extend(generate_subtables(tab))
        
        if load_claims is None:
            logger.info("""Claims are not provided, decomposing the text unit 
                        into atomic claims...""")
            # Create a list of atomic claims
            prompts = [PromptOracle.get_gen_atomic_claims_prompt(sentence) for sentence in sentences]
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens
            _tmp_output_tokens = total_output_tokens
            claims = queryLLM_batch(self.llm_claims, prompts)
            from utils.prompting import (total_input_tokens,
                                         total_output_tokens)
            _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
            _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
            with open(os.environ["TOKEN_LOG"], "a") as ff:
                ff.write(f"(GROUNDING) - A19: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")
            claims_expanded = []
            claims_expanded_index = []
            refs = []
            claims_list = []
            for i, sublist in enumerate(claims):
                refs.append(list(set([int(r[1:]) - 1 for r in re.findall(r"\[\d+", sentences[i])])))
                logger.info(f"For `{sentences[i]}`, found citations {refs[-1]} and claims {sublist}")

                for item in sublist.split("\n"):
                    claims_expanded.append(re.sub(r"^\d+: ", "", item).strip())
                    claims_expanded_index.append(i)
                    claims_list.append({"claim": re.sub(r"^\d+: ", "", item).strip(), 
                                        "text_unit_id": i, 
                                        "refs": refs[-1],
                                        "text_unit": sentences[i]})
                    
            logger.info(f"""Created {len(claims_expanded)} sub-claims from the 
                        original {len(sentences)} text units with 
                        {self.llm_claims.model_name}""")

        else:
            logger.info(f"Loading the claims from {load_claims}...")
            claims_list = pd.read_csv(load_claims, index_col=0)

            refs = []
            for sentence in sentences:
                refs.append(list(set([int(r[1:]) - 1 for r in re.findall(r"\[\d+", sentence)])))

            for i in range(len(sentences)):
                indexes = claims_list[claims_list.text_unit_id == i].index
                for index_tmp in indexes:
                    claims_list.at[index_tmp, "refs"] = refs[i]
                
            claims_list = claims_list.to_dict("records")
            logger.info(f"Loaded {len(claims_list)} sub-claims from the original {len(sentences)} text units")

        pd.DataFrame(claims_list).to_csv(os.path.join(eval_folder,
                                                      f"claims_list_v{find_last_report_version(eval_folder, "claims_list") + 1}.csv"))

        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens
        _tmp_output_tokens = total_output_tokens
        prompts_entail = []
        unassigned = []
        for jj, claim_dict in enumerate(claims_list):
            joint_entail = -1  # Undecided
            claim_refs = claim_dict["refs"]
            claim = claim_dict["claim"]

            # Extract the references from the sentence. NOTE: the '-1' is beacuse in 
            # the report the citation count starts from 1 and not 0
            # For the claim to be supported the citations set must be of at least
            # cardinality 1
            if len(claim_refs) == 0:
                joint_entail = 0
                sent_no_citations += 1
                logger.info(f"[Unsupported claim][No citations] {claim}")
                entail += joint_entail
                claims_list[jj]["recall"] = joint_entail
                claims_list[jj]["no_recall_type"] = "No citations"
            # The citation does not refer to any reference. This check is not sufficient
            # to ensure the citation is correct.
            elif any([ref_id not in references_df.index for ref_id in claim_refs]):
                joint_entail = 0
                nonexistent_refs = [str(ref_id) for ref_id in set(claim_refs) if ref_id not in references_df.index]
                nonexistent_refs_str = "[" + ",".join(nonexistent_refs) + "]"
                logger.info(f"[Unsupported claim][Reference id not found] {nonexistent_refs_str}")
                sent_ref_not_found += 1
                entail += joint_entail
                claims_list[jj]["recall"] = joint_entail
                claims_list[jj]["ref_not_found"] = "Reference id not found"
            else:
                total_citations += len(claim_refs)
                # Concatenate the texts to have the whole references content
                joint_ref_passages = '\n'.join(
                    [references_df.loc[ref_id]['summary'] for ref_id in set(claim_refs)]
                )

                # If not directly rejected by citation format error, 
                # calculate the recall binary score: it is 1 if the claim is entirely 
                # supported by the cited references, 0 otherwise
                prompts_entail.append(PromptOracle.get_entail_prompt(claim, joint_ref_passages, False))
                claims_list[jj]["recall"] = -1
                claims_list[jj]["joint_ref_passages"] = joint_ref_passages
                unassigned.append(jj)
                
        results = queryLLM_batch(self.llm_nli, prompts_entail)
        
        fallback_jobs = []
        new_prompts = []
        for _, result in enumerate(results):
            numerical_result = _parse_entailment_result(result)

            if numerical_result == 0:
                sent_failed_support += 1
                
            
            entail += int(numerical_result)
            ind = unassigned.pop(0)
            claims_list[ind]["recall"] = numerical_result
            claims_list[ind]["recall_explained"] = result
            if numerical_result == 0:
                claims_list[ind]["no_recall_type"] = "No logical support"
                joint_entire_passages = '\n'.join(
                    [references_df.loc[ref_id]['content'] for ref_id in set(claims_list[ind]["refs"])]
                )
                claims_list[ind]["joint_entire_passages"] = joint_entire_passages
                chunks = _build_grounding_chunks(
                    claims_list[ind]["claim"],
                    joint_entire_passages,
                    self.llm_nli.model_name,
                    DEFAULT_GROUNDING_MAX_INPUT_TOKENS,
                    DEFAULT_GROUNDING_CHUNK_OVERLAP_TOKENS,
                )
                prompt_start = len(new_prompts)
                new_prompts.extend([
                    PromptOracle.get_entail_prompt(
                        claims_list[ind]["claim"], chunk, False
                    )
                    for chunk in chunks
                ])
                fallback_jobs.append({
                    "claim_index": ind,
                    "prompt_start": prompt_start,
                    "prompt_end": len(new_prompts),
                })
                claims_list[ind]["grounding_chunk_count"] = len(chunks)
                logger.info(
                    "Split the full references for claim %s into %s grounding chunk(s)",
                    ind,
                    len(chunks),
                )

        if fallback_jobs:
            try:
                new_results = queryLLM_batch(self.llm_nli, new_prompts)
            except Exception as e:
                raise RuntimeError("Failed to run NLI batch for unassigned claims") from e
            for job in fallback_jobs:
                chunk_results = new_results[
                    job["prompt_start"]:job["prompt_end"]
                ]
                supporting_result = _find_supporting_result(chunk_results)

                # The summary pass already counted this claim as unsupported.
                # Full-text chunk results therefore use logical OR and only add
                # to the totals when at least one chunk fully supports the claim.
                if supporting_result is not None:
                    sent_failed_support -= 1
                    entail += 1
                    ind = job["claim_index"]
                    claims_list[ind]["recall"] = 1
                    claims_list[ind]["recall_explained"] = supporting_result
                    claims_list[ind].pop("no_recall_type", None)

        number_sentences = len(claims_list)
        logger.info("Of all the {} claims, {} ({:.2f})% have no citations".format(
            number_sentences, sent_no_citations, 100 * float(sent_no_citations)/number_sentences
        ))
        logger.info(f"Total citations: {total_citations}")

        citation_rec = 100 * entail / number_sentences
        logger.info("Citation recall: {:.2f}%".format(citation_rec))

        logger.info("The citation recall computed with the number of supported claims {} (not supported: {}, logically: {}) and excluding the claims with no citations ({}) is {:.2f}%".format(
            entail, number_sentences - entail, sent_failed_support, sent_no_citations, float(entail)/(number_sentences - sent_no_citations) * 100
        ))
        from utils.prompting import (total_input_tokens, total_output_tokens)
        _tmp_input_tokens = total_input_tokens - _tmp_input_tokens
        _tmp_output_tokens = total_output_tokens - _tmp_output_tokens
        with open(os.environ["TOKEN_LOG"], "a") as ff:
            ff.write(f"(GROUNDING) - A20: [{_tmp_input_tokens},{_tmp_output_tokens}]\n")

        claims_df = pd.DataFrame(claims_list)
        claims_df.to_csv(os.path.join(eval_folder, f"recall_df_v{find_last_report_version(eval_folder, "recall") + 1}.csv"))
        # -------------------------------
        # Compute citation precision
        # -------------------------------
        if evaluate_precision:
            def _compute_relevant_citations(df: pd.DataFrame):
                
                relevant_citations = 0
                total_citations = 0
                for _, row in df.iterrows():
                    # Iteration over the claims
                    # A citation of a set can be deemed irrelevant if it is not relevant in any claim
                    refs = row.refs
                    # import ast
                    # refs = ast.literal_eval(refs)
                    refs_relevance_tmp = np.zeros(len(refs))
                    if len(refs) > 1 and row.recall == 1:
                        for i, ref_id in enumerate(refs):
                            ref_text = references_df.loc[ref_id]["summary"]
                            nli_result = self._run_nli_autoais(
                                references_text=ref_text, 
                                claim=row.claim, 
                                partial=True)
                            
                            if nli_result == 1:
                                refs_relevance_tmp[i] += 1
                            else:
                                # If the ref does not support this claim by itself,
                                # check if the concat of the others does
                                subset_exclude = set(copy.deepcopy(refs))
                                subset_exclude.remove(ref_id)
                                subset_exclude_text = '\n'.join(
                                    [references_df.loc[ref_id_excluded]["summary"] \
                                    for ref_id_excluded in subset_exclude]
                                )
                                nli_result = self._run_nli_autoais(
                                    references_text=subset_exclude_text,
                                    claim=row.claim,
                                    partial=False)
                                # If the claim is supported, it means that 'ref_id' was not
                                # fundamental to support the claim, thus it is an 
                                # irrelevant citation
                                if nli_result:
                                    # Not relevant here, no metrics to be updated
                                    pass
                                else:
                                    # The claim was supported and without the ref not 
                                    # anymore: citation is relevant
                                    refs_relevance_tmp[i] += 1
                    elif len(refs) == 1 and row.recall == 1:
                        refs_relevance_tmp[0] += 1
                    else:  # Recall 0 or no refs
                        pass

                    relevant_citations += int(np.sum(refs_relevance_tmp > 0))
                    total_citations += len(refs_relevance_tmp)
                return pd.Series([relevant_citations, total_citations], 
                                index=['relevant_citations', 'total_citations'])
            
            prec_df = claims_df.groupby('text_unit_id').apply(_compute_relevant_citations).reset_index()
            relevant_citations = float(prec_df["relevant_citations"].sum())
            total_citations = float(prec_df["total_citations"].sum())
            logger.info((f"Over all text units, there are {relevant_citations} "
                        f"relevant citations and {total_citations} total "
                        "citations respectively."))
            logger.info(f"Citation precision: {relevant_citations/total_citations * 100}%")

            # Save to pandas DataFrame
            prec_df.to_csv(os.path.join(eval_folder, f"prec_df_v{find_last_report_version(eval_folder, "prec") + 1}.csv"))
        else:
            logger.info("Skipping the evaluation of the citation precision")
            prec_df = None

        return citation_rec, entail, sent_no_citations, sent_ref_not_found, sent_failed_support, claims_df, prec_df
