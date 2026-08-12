import logging
import re
from typing import List, Union

logger = logging.getLogger(__name__)


def extract_image_paragraphs(markdown_text: str) -> List[tuple]:
    """Extract the image paths from the markdown text and the paragraphs
    above and below each figure insertion.
    
    Parameters
    ----------
    - markdown_text [`str`]: the source text
    
    Output
    ------
    - `List[tuple]`: a list of tuples, where each tuple is associated to a 
    figure. The tuple contains three elements: the paragraph above the figure,
    the image path, and the paragraph below the figure.
    """
    image_pattern = re.compile(r'!\[Image]\((.*?)\)')
    lines = markdown_text.split('\n\n')
    results = []
    
    for i, line in enumerate(lines):
        # Check if the line contains an image
        match = image_pattern.search(line)
        if match:
            # Extract the image path
            image_path = match.group(1)
            
            # Check if there is a non-empty line above the image. If so, there
            # is a candidate caption above the figure to be extracted.
            if i > 0 and lines[i-1].strip():
                paragraph_before = lines[i-1].strip()
            else:
                paragraph_before = None
            
            # Check if there is a non-empty line after the image. If so, there
            # is a candidate caption below the figure to be extracted.
            if i + 1 < len(lines) and lines[i+1].strip():
                paragraph_after = lines[i+1].strip()
            else:
                paragraph_after = None
                
            results.append((paragraph_before, image_path, paragraph_after))
    
    return results


def extract_figure_number(caption: str) -> Union[int, None]:
    """Given a caption, extract the figure number.
    
    Parameters
    ----------
    - caption [`str`]: caption of the figure

    Output
    ------
    - `Union[int, None]`: figure number
    """
    # Match "Figure", "fig", "Fig:", "fig." followed by a number
    figure_pattern = re.compile(r'(?i)(fig(?:ure)?|fig\.)\s*[:\.]?\s*(\d+)')
    if caption is None:
        logger.info(f"None caption: {caption}")
        return None
    match = figure_pattern.search(caption)
    
    if match:
        return int(match.group(2))
    else:
        logger.info(f"Failed number extraction from caption: {caption}")
        return None 


def find_paragraphs_mentioning_figure(markdown_text: str, 
                                      figure_number: int) -> List[str]:
    """Return a list of the paragraphs which mention the figure identified by
    the provided integer number.
    
    Parameters
    ----------
    - markdown_text [`str`]: the text in markdown format
    - figure_number [`int`]: the figure number to look for references
    
    Output
    ------
    - `List[str]`: list of paragraphs that mention the figure
    """
    # Match a mention to the specific figure number
    figure_pattern = re.compile(r'(?i)(fig(?:ure)?|fig\.)\s+' + re.escape(str(figure_number)) + r'\b')
    paragraphs = markdown_text.split('\n\n')
    
    matching_paragraphs = []
    
    # Iterate through the paragraphs and check if the figure number is mentioned
    for paragraph in paragraphs:
        if figure_pattern.search(paragraph):
            matching_paragraphs.append(paragraph.strip())
    
    return matching_paragraphs


def extract_image_text(markdown_text: str) -> List[dict]:
    """Extract the images from the input markdown text and the textual 
    information (captions and paragraphs) referring to them.
    
    Parameters
    ----------
    - markdown_text [`str`]: markdown text of the document
    
    Output
    ------
    - `List[dict]`: list of dictionaries, one per image. The dictionaries 
    contain the following keys: 'image_path', 'figure_number', 'caption', 
    'text_snippets'
    """
    image_info = extract_image_paragraphs(markdown_text)
    figure_last_id = 0
    images_info = []

    for figure_paragraph in image_info:
        image_path = figure_paragraph[1]
        # Assess whether the first paragraph, i.e. the paragraph above the figure
        # contains the caption. Check whether the caption starts with "Fig" or "fig"
        # and if the figure identifier in such case is different from the last
        # encountered one. This may happen if we have captions placed below figures
        # and there are multiple consecutive ones. 
        fig_number_tmp = extract_figure_number(figure_paragraph[0])
        if (figure_paragraph[0] is not None) and (figure_paragraph[0].lower().startswith("fig")) and \
            (fig_number_tmp != figure_last_id):
            figure_current_id = fig_number_tmp
            figure_last_id = figure_current_id
            caption = figure_paragraph[0]
            # Find the paragraphs mentioning the figure, but exclude the caption 
            # from those
            text_snippets = find_paragraphs_mentioning_figure(markdown_text, figure_current_id)
            text_snippets = [snippet for snippet in text_snippets if snippet != caption]

        # If the caption was not in the paragraph above the figure, assess whether
        # it is in the paragraph below
        elif (figure_paragraph[2] is not None) and (figure_paragraph[2].lower().startswith("fig")):
            figure_current_id = extract_figure_number(figure_paragraph[2])
            figure_last_id = figure_current_id
            caption = figure_paragraph[2]
            text_snippets = find_paragraphs_mentioning_figure(markdown_text, figure_current_id)
            text_snippets = [snippet for snippet in text_snippets if snippet != caption]
            
        # If the caption is not either in the paragraph above or below, then there
        # may have been some issues in the parsing. In that case, skip the figure
        # to avoid a wrong caption assignment
        else:
            continue

        images_info.append({
            "image_path": image_path,
            "figure_number": figure_current_id,
            "caption": caption,
            "text_snippets": text_snippets
        })

    return images_info


class FiguresMasker():
    def __init__(self):
        self.base64_to_link_map = {}
        self.link_to_base64_map = {}
        self.source_map = {}
        self.counter = 1  # Counter of the image
        self.source_counter = 1  # Counter for sources

    def replace_base64_with_links(self, markdown_text):
        """
        Replaces all images encoded in base64 with 'link_*' placeholders.
        While doint that, it stores the mappings from base64 to 'link_*' 
        identifiers.
        """
        image_pattern = r'!\[Image\]\(data:image\/png;base64,([a-zA-Z0-9+/=]+)\)'
        
        def replace_image(match):
            base64_encoding = match.group(1)
            
            # If this base64 string has not been mapped yet, 
            # assign it a new link_* with the counter
            if base64_encoding not in self.base64_to_link_map:
                link_id = f'link_{self.counter}'
                self.base64_to_link_map[base64_encoding] = link_id
                self.link_to_base64_map[link_id] = base64_encoding
                self.counter += 1
            
            # Return the corresponding link_* identifier
            return f'![Image]({self.base64_to_link_map[base64_encoding]})'

        markdown_text = re.sub(image_pattern, replace_image, markdown_text)

        # Replace "*Source: [...]" lines with placeholders
        source_pattern = r'^.*Source:\s*\[.*\n'

        
        def replace_source(match):
            source_text = match.group(0)
            placeholder = f'\nsource_placeholder_{self.source_counter}\n'
            self.source_map[placeholder.strip()] = source_text.strip()
            self.source_counter += 1
            return placeholder

        markdown_text = re.sub(source_pattern, replace_source, markdown_text, flags=re.MULTILINE)

        return markdown_text

    def reconvert_links_to_base64(self, markdown_text):
        """Reverts all 'link_*' and 'source*' placeholders back to their 
        corresponding image (base64 encoding) and source strings.
        """
        image_pattern = r'!\[Image\]\((link_\d+)\)'

        def revert_image(match):
            link_id = match.group(1)
            if link_id in self.link_to_base64_map:
                return f'\n\n![Image](data:image/png;base64,{self.link_to_base64_map[link_id]})'
            return match.group(0)

        markdown_text = re.sub(image_pattern, revert_image, markdown_text)

        # Revert also the source placeholders
        for placeholder, original in self.source_map.items():
            markdown_text = markdown_text.replace(f'{placeholder}\n', 
                                                  f'{original}\n')

        return markdown_text

    def get_mapping(self):
        """Returns the mapping of base64 to link_* identifiers.
        """
        return self.base64_to_link_map

    def get_reversed_mapping(self):
        """Returns the reversed mapping of link_* to base64-encoded strings.
        """
        return self.link_to_base64_map

    def get_source_mapping(self):
        """Returns the mapping of sources identifiers.
        """
        return self.source_map


def remove_duplicated_figures(figures: list) -> list:
    """Force the removal of possible duplicates.
    """
    seen = set()
    deduplicated = []

    for item in figures:
        figure_id = next(iter(item))  # e.g., "FIGURE X"

        if figure_id not in seen:
            deduplicated.append(item)
            seen.add(figure_id)
        else:
            logger.info(f"Figure {figure_id} is a duplicate, removing the entry {item}")
    return deduplicated