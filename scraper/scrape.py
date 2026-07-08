import logging
import os
import re
import time

import html2text
import requests
import yaml
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = "articles"
BASE_URL = "https://support.optisigns.com/api/v2/help_center/articles.json"
REQUEST_DELAY_SECONDS = 0.3
LOW_VALUE_WORD_THRESHOLD = 40

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def slugify(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", "-", title)
    return title.strip("-")


def fetch_articles(session):
    articles = []
    url = BASE_URL
    while url:
        res = session.get(url, timeout=30)
        res.raise_for_status()
        data = res.json()
        articles.extend(data["articles"])
        url = data.get("next_page")
        logger.info("Fetched %d articles so far...", len(articles))
        if url:
            time.sleep(REQUEST_DELAY_SECONDS)
    return articles


def inline_image_placeholders(soup):
    """Replace <img> with a text placeholder so image-only table cells/steps
    don't collapse into empty markdown once the image itself is dropped."""
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        img.replace_with(f"[Screenshot: {alt}]" if alt else "[Screenshot]")
    return soup


EMPTY_TABLE_ROW_RE = re.compile(r"^\s*(\|\s*)+$")
EMPTY_HEADING_RE = re.compile(r"^#{1,6}\s*$")


def clean_markdown(markdown):
    lines = [
        line
        for line in markdown.split("\n")
        if not EMPTY_TABLE_ROW_RE.match(line) and not EMPTY_HEADING_RE.match(line)
    ]
    markdown = "\n".join(lines)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)  # trailing whitespace left by html2text
    markdown = re.sub(r"\*{4}", "", markdown)  # stray '****' from malformed nested bold tags
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def convert_body_to_markdown(body):
    soup = BeautifulSoup(body or "", "html.parser")
    inline_image_placeholders(soup)

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return clean_markdown(h.handle(str(soup)))


def build_frontmatter(article, markdown_body, slug):
    word_count = len(markdown_body.split())
    frontmatter = {
        "id": article["id"],
        "title": article["title"],
        "slug": slug,
        "url": article["html_url"],
        "section_id": article.get("section_id"),
        "locale": article.get("locale"),
        "created_at": article.get("created_at"),
        "updated_at": article.get("updated_at"),
        "label_names": article.get("label_names") or [],
        "low_value": word_count < LOW_VALUE_WORD_THRESHOLD,
    }
    return "---\n" + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True) + "---\n"


def save_article(article):
    title = article["title"]
    slug = slugify(title)
    markdown_body = convert_body_to_markdown(article.get("body"))
    frontmatter = build_frontmatter(article, markdown_body, slug)

    content = f"{frontmatter}\n# {title}\n\nArticle URL: {article['html_url']}\n\n{markdown_body}\n"
    filename = f"{article['id']}-{slug}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def cleanup_orphans(expected_filenames):
    """Remove local files for articles that no longer exist on the Help Center."""
    existing = {name for name in os.listdir(OUTPUT_DIR) if name.endswith(".md")}
    orphans = existing - expected_filenames
    for name in orphans:
        os.remove(os.path.join(OUTPUT_DIR, name))
        logger.info("Removed orphan article: %s", name)
    if orphans:
        logger.info("Removed %d orphan article(s) no longer present in Help Center", len(orphans))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = build_session()
    articles = fetch_articles(session)
    logger.info("Total: %d articles", len(articles))

    expected_filenames = set()
    for article in articles:
        filename = save_article(article)
        expected_filenames.add(filename)
        logger.info("Saved: %s", article["title"])

    cleanup_orphans(expected_filenames)


if __name__ == "__main__":
    main()
