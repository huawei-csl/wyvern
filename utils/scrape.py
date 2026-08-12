import argparse
import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import requests
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

def url_to_filename(url: str, 
                    max_length: int = 50):
    """Convert the URL to a filename, using hashing to not have duplicates.
    
    Parameters
    ----------
    - url [`str`]: the input URL
    - max_length [`int`, default = 50]: maximum length for the filename 
    (extension excluded)
    
    Output
    ------
    - `str`: the filename (without extension)
    """
    # Remove protocol and www
    cleaned = re.sub(r'^https?://(www\.)?', '', url)
    safe = cleaned.replace("/", "_")
    # Truncate with hash to preserve uniqueness of the filename
    if len(safe) > max_length:
        # Hash the full URL and append to the truncated base
        hash_suffix = hashlib.md5(url.encode()).hexdigest()[:8]
        safe = safe[:max_length - 9] + "_" + hash_suffix
    return safe


def guard_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Blocked URL: {url}")
    # Resolve the hostname before fetching so internal targets are not considered
    for *_, addr in socket.getaddrinfo(parsed.hostname, None):
        ip = ipaddress.ip_address(addr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"Blocked URL: {url}")
    return url


def safe_request(method: str, url: str, **kwargs):
    for _ in range(10):
        guard_url(url)
        response = requests.request(method, url, allow_redirects=False, **kwargs)
        # Follow redirects manually so that each hop is validated again
        if response.is_redirect and response.headers.get("Location"):
            url = urljoin(response.url, response.headers["Location"])
            response.close()
            continue
        return response
    raise ValueError(f"Too many redirects: {url}")


async def scrape_website(url):
    """Scrape a single website and return HTML content (async)
    """
    try:
        # Resolve server-side redirects safely before handing the URL to the browser
        response = safe_request("GET", url, headers={"User-Agent": "Wyvern/1.0"}, timeout=10, stream=True)
        url = response.url
        response.close()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Wyvern/1.0")
            page = await context.new_page()

            # Wait for the page to load
            await page.goto(url)
            # Re-check in case the browser was redirected to a different destination.
            guard_url(page.url)
            await page.wait_for_load_state('domcontentloaded')
            content = await page.content()
            await browser.close()

            return content
    except Exception as e:
        logger.info("Skipping %s: %s", url, e)
        return None


async def scrape_multiple_urls(urls, output_dir):
    """Scrape multiple URLs and save them as HTML files in the provided output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        try:
            html_content = await scrape_website(url)
            if html_content is None:
                continue

            filename = os.path.join(output_dir, url_to_filename(url) + ".html")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(html_content)

        except Exception as e:
            print("Error in saving HTML: ", e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urls", 
        nargs="+", 
        help="List of URLs to scrape", 
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        help="Directory where to save HTML files", 
        default="tmp_scraped_websites"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not asyncio.get_event_loop().is_running():  # script
        asyncio.run(scrape_multiple_urls(args.urls, args.output_dir))
    else:  # Jupyter
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scrape_multiple_urls(args.urls, args.output_dir))
