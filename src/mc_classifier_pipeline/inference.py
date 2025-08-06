from argparse import ArgumentParser
from pathlib import Path
import mediacloud.api
from dotenv import load_dotenv
import os
from importlib.metadata import version
import datetime as dt
import json
import pandas as pd

from .utils import configure_logging

logger = configure_logging()

load_dotenv()
api_key = os.getenv("MC_API_KEY")
search_api = mediacloud.api.SearchApi(api_key)
directory_api = mediacloud.api.DirectoryApi(api_key)
f"Using Media Cloud python client v{version('mediacloud')}"



def build_inference_parser():
    parser = ArgumentParser(

    )

    parser.add_argument(
        "--url-file",
        type=Path,
        help = "Path to the file with list of urls"
    )

    parser.add_argument(
        "--model",
        type = Path,
        help = "Path to the trained model"
    )

    return parser

def parse_args():
    parser = build_inference_parser()
    return parser.parse_args()


def process_urls(urls):
    articles = []
    #urls = urls[:2]
    for url in urls:
        my_query = f'url:"{url}"'
        start_date = dt.date(2025, 1, 1)
        end_date = dt.date(2025, 6, 10)
        results = search_api.story_list(my_query, start_date, end_date)
        if(results and len(results[0]) > 0):
            story_id = results[0][0]['id']
            articles.append(search_api.story(story_id))

    df = pd.DataFrame(articles)[["id", "url", "text"]]
    df.to_csv("predictions.csv", index=False)



def main():
    args = parse_args()

    url_file = args.url_file
    urls = []
    with open(url_file, "r") as f:
        urls = [url.strip() for url in f]

    process_urls(urls)
    





if __name__ == "__main__":
    main()
