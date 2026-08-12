import json
import logging
import os
import re
import urllib
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Optional

import html2text
import pandas as pd
import requests
import tqdm
import tqdm.auto
from bs4 import BeautifulSoup
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (AcceleratorDevice,
                                                AcceleratorOptions,
                                                PdfPipelineOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem
from markdownify import markdownify
from markitdown import MarkItDown
from trafilatura import extract

from utils.general import filter_arxiv_url, normalize_arxiv_url
from utils.scrape import guard_url, safe_request, url_to_filename

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    WIKIPEDIA = auto()
    PDF = auto()
    HTML = auto()


HEADERS_LIST = [
    {"User-Agent": "Wyvern/1.0"},
    ]


@lru_cache(maxsize=1)
def get_docling_converter() -> DocumentConverter:
    """Build one reusable Docling converter for all document conversions."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO,
    )
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def get_search_results(query, filters=["youtube.com"], try_load=True):
    """Retrieve the results from Google Search for the input query.
    If the search has already been performed, re-load the saved results to avoid
    an unnecessary call to the API. Filter out websites from a given list
    (e.g. YouTube, that does not contain textual data)."""
    
    # Add the websites filters in the query string
    for filter in filters:
        query = query + " -inurl:" + filter

    json_path = os.path.join("tmp", "search")
    os.makedirs(json_path, exist_ok=True)
    json_path = os.path.join(json_path, f"{query[:150]}-us-en-true-1-10-search--.json".replace("/", "-"))

    # If the path exists, then load the search results from the existing json file
    if os.path.exists(json_path) and try_load:
        print(f"Search results already exist, loading from '{json_path}'")
        with open(json_path) as fp:
            text = json.load(fp)
    # If the path does not exist, make the API call    
    else:
        print("Searching...")
        url = "https://google.serper.dev/search"

        payload = json.dumps({
        "q": query
        })
        headers = {
        'X-API-KEY': os.environ.get("SERPER_API_KEY"),
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
        text = json.loads(response.text)

        # Save the result to a json file for later reuse if needed
        with open(json_path, "w", encoding='utf-8') as fp:
            json.dump(text, fp, ensure_ascii=False, indent=4)
    
    return text


def retrieve_results_from_queries(queries: list[str], 
                                  n: int=10, 
                                  df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Given a list of queries, perform the search and extract the top-n URLs of interest.
    Return the result as a `pd.DataFrame`.
    """
    for query in queries:
        text = get_search_results(query)
        links = extract_links(text)[:n]
        if df is None:
            df = pd.DataFrame({"url": links, "query": query})
        else:
            df = pd.concat(
                (df, pd.DataFrame({"url": links, "query": query})), 
                ignore_index=True)

    df["url"] = df["url"].apply(normalize_arxiv_url)

    # Remove duplicated URLs, obtained with different queries
    logger.info(f"Retrieved {len(df)} URLs from the search, removing {df.url.duplicated().sum()} entries because of duplicated URLs")
    df = df[~df.url.duplicated()].reset_index(drop=True)

    removed_counter = len(df)
    df = df[df.url.isin(filter_arxiv_url(df.url))]
    removed_counter -= len(df)
    logger.info((f"Removed {removed_counter} resources as they are not the most recent arXiv version"))

    logger.info(f"After the search, {len(df)} URLs are in the DataFrame")
    return df


def extract_links(response_text):
    """Extract the webpages urls from the search results. 
    Avoid specific websites (such as YouTube, that contains non-textual data)
    and replace any arxiv url with the link to the pdf version of the paper"""

    links = []
    for result in response_text["organic"]:
        # If a retrived webpage is an arxiv article,
        # replace the abstract webpage or html version with the pdf version of the article
        if "arxiv.org" in result["link"]:
            links.append(normalize_arxiv_url(result["link"]))
        else:  # Add the retrieved link to the links list without any change
            links.append(result["link"])

    return links


def fetch_webpage(link: str):
    """Fetch the webpage given the url as input.
    """
    for headers in HEADERS_LIST:
        try:
            response = safe_request("GET", link, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"Request failed with error: {e}. Trying next headers...")

    print(f"\n\n***FAILED to fetch {link}")
    return None


def extract_contents(links, tool="docling"):
    """Extract the content of webpages from the given url to the markdown format.
    If possible, it directly uses the docling converter. In case of an HTTPError,
    the urllib and markdownify libraries are used to extract the content."""
    
    texts = []
    for link in tqdm.auto.tqdm(links, total=len(links),
                               desc=f"Extracting webpages content (trying with {tool})"):
        texts.append(extract_content(link, tool))
    
    return texts


def extract_content(link: str, tool: str="docling"):
    """Extract the content of fetched webpages. With the `tool`parameter it is possible to select
    which library to use for the extraction to markdown (or txt in the case of Beautiful Soup). 
    Default is "docling", supported options are "trafilatura", "beautifulsoup", "markdownify", "markitdown".
    """

    try:
        link = guard_url(link)
    except Exception as e:
        logger.warning("Skipping %s: %s", link, e)
        return ""

    # Check whether the link points to a website with available API.
    if "wikipedia.org" in link:
        return parse_wikipedia(link)
    
    if tool.lower() == "markitdown":
        try:
            md = MarkItDown()
            text = md.convert(link)
            text = text.text_content
        except Exception as e:
            tool = "docling"
            print(f"\tFailed to parse website '{link}' with Markitdown, trying with '{tool}'.")
    
    if tool.lower() == "docling":
        try: 
            converter = get_docling_converter()
            result = converter.convert(link)
            text = result.document.export_to_markdown()
        except Exception as e:
            tool = "trafilatura"
            print(f"\tFailed to parse website '{link}' with Docling, trying with '{tool}'.")
            
    response = fetch_webpage(link)

    # If it was impossible to fetch the webpage return an empty string as content
    if response is None:
        return ""
    
    if tool.lower() == "trafilatura":
        text = extract(response,
                       include_links=True,
                       include_tables=True,
                       include_images=True, 
                       include_comments=False, 
                       output_format="markdown")
    
    if tool.lower() == "beautifulsoup":
        soup = BeautifulSoup(response, features="lxml")
        text = soup.text

    if tool.lower() == "markdownify":
        text = markdownify(response, heading_style="ATX")

    return text.strip()


def parse_wikipedia(link: str):
    """Extract the content of a Wikipedia webpage by means of its API.
    """
    try:
        link = guard_url(link)
    except Exception as e:
        logger.warning("Skipping %s: %s", link, e)
        return ""
    if "wikipedia.org" not in link:
        raise ValueError(f"The link '{link}' does not point to a Wikipedia webpage!")
    
    page_name = link.split("/")[-1]
    link = f"https://en.wikipedia.org/w/api.php?action=parse&page={page_name}&format=json"

    # Fetch the HTML content
    response = safe_request("GET", link, timeout=10)
    html_content = response.json()['parse']['text']['*']

    # Convert HTML to markdown using the html2text library
    markdown_content = html2text.html2text(html_content)

    # Manually refactor the links to add the domain "https://en.wikipedia.org/" 
    # if they start with "/wiki/"
    pattern = r'(/wiki/[^"\s]*)'
    markdown_content = re.sub(pattern, r'https://wikipedia.org\1', markdown_content)

    return markdown_content


def assign_resource_type(url: str) -> ResourceType:
    """Assign the resource type given the URL.
    """
    if "wikipedia.org" in url:
        return ResourceType.WIKIPEDIA
    else:
        for headers in HEADERS_LIST:
            try:
                r = safe_request("HEAD", url, headers=headers, timeout=30)
                if r.status_code == 200 and r.headers.get("Content-Type") in ("application/pdf", "application/octet-stream"):
                    return ResourceType.PDF
                return ResourceType.HTML
            except requests.RequestException as e:
                print(f"Request failed with error: {e}. Trying next headers...")
            except Exception as e:
                print(f"Request failed with error: {e}. Trying next headers...")

        print(f"All attempts with different headers have failed. Returning HTML for {url}.")
        return ResourceType.HTML
        

def convert(url: str, source: ResourceType, html_folder: str, tool: str="markitdown") -> str:
    """Convert the content of the resource with the input URL to markdown.
    The source parameter allows to discriminate between pdf, html and URLs of specific domains.
    """
    def get_figures_folder() -> Path:
        figures_folder = (Path(html_folder).parent / "figures").resolve()
        figures_folder.mkdir(parents=True, exist_ok=True)
        return figures_folder

    try:
        url = guard_url(url)
        if source == ResourceType.WIKIPEDIA:
            return parse_wikipedia(url)
        else:
            if tool == "markitdown":
                if source == ResourceType.PDF:
                    print("Parsing the pdf with docling: ", url)
                    converter = get_docling_converter()
                    result = converter.convert(url)
                    figures_folder = get_figures_folder()
                    FLAG_PICTURES = False
                    for element, _level in result.document.iterate_items():
                        if isinstance(element, PictureItem):
                            FLAG_PICTURES = True
                            break
                    doc_name = url_to_filename(url)
                    if FLAG_PICTURES:
                        result.document.save_as_markdown(figures_folder / f"{doc_name}-with-image-refs.md", 
                                                        image_mode=ImageRefMode.REFERENCED)
                    doc_with_refs = result.document._make_copy_with_refmode(
                        figures_folder / f"{doc_name}-with-image-refs_artifacts", 
                        ImageRefMode.REFERENCED)

                    md_out = doc_with_refs.export_to_markdown(
                        image_mode=ImageRefMode.REFERENCED
                    )
                    return md_out.strip()
                elif source == ResourceType.HTML:
                    md = MarkItDown()
                    if os.path.exists(f"{html_folder}/{url_to_filename(url)}.html"):
                        return md.convert(f"{html_folder}/{url_to_filename(url)}.html").text_content.strip()
                    else:
                        print(f"Parsing failed for {url}, returning empty string")
                        return ""

            elif tool == "docling":
                converter = get_docling_converter()
                result = converter.convert(url)
                figures_folder = get_figures_folder()
                FLAG_PICTURES = False
                for element, _level in result.document.iterate_items():
                    if isinstance(element, PictureItem):
                        FLAG_PICTURES = True
                        break
                doc_name = url_to_filename(url)
                if FLAG_PICTURES:
                    result.document.save_as_markdown(figures_folder / f"{doc_name}-with-image-refs.md", 
                                                     image_mode=ImageRefMode.REFERENCED)
                doc_with_refs = result.document._make_copy_with_refmode(
                    figures_folder / f"{doc_name}-with-image-refs_artifacts", 
                    ImageRefMode.REFERENCED)

                md_out = doc_with_refs.export_to_markdown(
                    image_mode=ImageRefMode.REFERENCED
                )
                return md_out.strip()
            
            elif tool == "trafilatura":
                if source == ResourceType.PDF:
                    print(f"Trafilatura supports HTML only, using Docling to parse the pdf at {url}")
                    converter = get_docling_converter()
                    result = converter.convert(url)
                    figures_folder = get_figures_folder()
                    FLAG_PICTURES = False
                    for element, _level in result.document.iterate_items():
                        if isinstance(element, PictureItem):
                            FLAG_PICTURES = True
                            break
                    doc_name = url_to_filename(url)
                    if FLAG_PICTURES:
                        result.document.save_as_markdown(figures_folder / f"{doc_name}-with-image-refs.md", 
                                                        image_mode=ImageRefMode.REFERENCED)
                    doc_with_refs = result.document._make_copy_with_refmode(
                        figures_folder / f"{doc_name}-with-image-refs_artifacts", 
                        ImageRefMode.REFERENCED)

                    md_out = doc_with_refs.export_to_markdown(
                        image_mode=ImageRefMode.REFERENCED
                    )
                    return md_out.strip()
                else:
                    with open(f"{html_folder}/{url_to_filename(url)}.html", "r") as f:
                        html_content = f.read()

                    text = extract(
                        html_content,
                        include_links=True,
                        include_tables=True,
                        include_images=True, 
                        include_comments=False, 
                        output_format="markdown"
                        )
                    return text.strip()
            else:
                raise ValueError("Only Docling, Markitdown and Trafilatura are accepted")
    except Exception as e:
        logger.warning("Skipping %s: %s", url, e)
        print(f"Parsing failed for {url}, returning empty string ({e})")
        return ""


def fix_url(url, base):
    """Fix the relative URLs mapping them to their absolute version. In the case of arXiv links,
    ensure the pdf version of the paper is considered.
    
    Parameters
    ----------
    - url [`string`]: the url to be analyzed
    - base [`str`]: the base domain of the webpage from which the URL has been extracted
    
    Output
    ------
    - `str`: the fixed URL
    """
    if urllib.parse.urlsplit(url).netloc == "":
        new_url = "https://" + urllib.parse.urlsplit(base).netloc + url
    else: 
        new_url = url
    if "arxiv.org" in new_url:
        new_url = normalize_arxiv_url(new_url)
    return new_url


def remove_embedded_links(text):
    """Remove the image and textual embedded links from the text. For the latter, the description is kept.
    
    Parameters
    ----------
    - text [`str`]: input markdown text
    
    Output
    - `str`: text without embedded links
    """

    # Remove images links
    text_no_links = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove purely textual links
    text_no_links = re.sub(r"\[(.+)\]\(.+\)", r"\1", text_no_links)
    
    return text_no_links
