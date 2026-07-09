import logging
import sys
import time

from scraper.scrape import main as scrape_main
from uploader.upload import main as upload_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    start = time.monotonic()
    logger.info("Job started")

    try:
        scrape_main()
        logger.info("Scrape completed")

        upload_main()
        logger.info("Upload completed")
    except Exception:
        logger.exception("Job failed")
        return 1

    elapsed = time.monotonic() - start
    logger.info("Job finished successfully in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
