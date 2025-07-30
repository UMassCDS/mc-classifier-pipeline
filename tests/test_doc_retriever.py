import sys
import os
import pytest
import tempfile
import shutil
import json
from pathlib import Path
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from mc_classifier_pipeline import doc_retriever


class DummySearchApi:
    def __init__(self, stories=None, story_data=None):
        self.stories = stories or []
        self.story_data = story_data or {}

    def story_list(self, query, start, end, collection_ids=None):
        return [self.stories]

    def story(self, story_id):
        return self.story_data.get(story_id, {})


@pytest.fixture
def temp_dirs():
    raw_dir = Path(tempfile.mkdtemp())
    failed_log = raw_dir / "failed_urls.txt"
    index_file = raw_dir / "articles_index.json"
    yield raw_dir, failed_log, index_file
    shutil.rmtree(raw_dir)


@pytest.fixture
def dummy_articles():
    return [
        {
            "url": "http://example.com/1",
            "title": "Title 1",
            "text": "Text 1",
            "source": "mediacloud_query",
            "retrieved_at": dt.datetime.now().isoformat(),
            "status": doc_retriever.ArticleStatus.SUCCESS,
            "story_id": "1",
            "publish_date": "2024-12-01",
            "media_id": "m1",
            "language": "en",
            "query": "test",
        },
        {
            "url": "http://example.com/2",
            "title": "Title 2",
            "text": "",
            "source": "mediacloud_query",
            "retrieved_at": dt.datetime.now().isoformat(),
            "status": doc_retriever.ArticleStatus.FAILED_NO_TEXT,
            "story_id": "2",
            "publish_date": "2024-12-02",
            "media_id": "m2",
            "language": "fr",
            "query": "test",
        },
    ]


def test_load_and_save_articles_index(temp_dirs):
    _, _, index_file = temp_dirs
    index = {"1": {"status": str(doc_retriever.ArticleStatus.SUCCESS)}}
    doc_retriever.save_articles_index(index, index_file)
    loaded = doc_retriever.load_articles_index(index_file)
    assert loaded == index


def test_is_article_retrieved():
    index = {"1": {"status": str(doc_retriever.ArticleStatus.SUCCESS), "text_length": 10}}
    assert doc_retriever.is_article_retrieved("1", index)
    assert not doc_retriever.is_article_retrieved("2", index)
    index_fail = {"1": {"status": str(doc_retriever.ArticleStatus.FAILED_NO_TEXT), "text_length": 0}}
    assert not doc_retriever.is_article_retrieved("1", index_fail)


def test_search_mediacloud_by_query(monkeypatch, temp_dirs):
    raw_dir, _, _ = temp_dirs
    stories = [{"id": "1"}, {"id": "2"}]
    story_data = {
        "1": {
            "url": "http://example.com/1",
            "title": "Title 1",
            "text": "Text 1",
            "publish_date": "2024-12-01",
            "media_id": "m1",
            "language": "en",
        },
        "2": {
            "url": "http://example.com/2",
            "title": "Title 2",
            "text": "",
            "publish_date": "2024-12-02",
            "media_id": "m2",
            "language": "fr",
        },
    }
    dummy_api = DummySearchApi(stories, story_data)
    monkeypatch.setattr(doc_retriever, "SEARCH_API", dummy_api)
    articles = doc_retriever.search_mediacloud_by_query(
        query="test",
        start_date=dt.date(2024, 12, 1),
        end_date=dt.date(2024, 12, 2),
        limit=2,
        articles_index={},
        raw_articles_dir=raw_dir,
    )
    assert len(articles) == 2
    assert articles[0]["url"] == "http://example.com/1"
    assert articles[1]["url"] == "http://example.com/2"


def test_save_articles_from_query(temp_dirs, dummy_articles):
    raw_dir, failed_log, index_file = temp_dirs
    articles_index = {}
    doc_retriever.save_articles_from_query(dummy_articles, raw_dir, failed_log, articles_index)
    # Check files written
    for article in dummy_articles:
        story_id = article["story_id"]
        path = raw_dir / f"{story_id}.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
            assert data["story_id"] == story_id
    # Check failed log
    assert failed_log.exists()
    with open(failed_log) as f:
        lines = f.read().splitlines()
        assert "http://example.com/2" in lines
    # Check index
    assert "1" in articles_index
    assert "2" in articles_index


def test_analyze_search_results(caplog, dummy_articles):
    caplog.set_level("INFO")
    doc_retriever.analyze_search_results(dummy_articles)
    assert "Total articles found: 2" in caplog.text
    assert "Successful retrievals: 1" in caplog.text
    assert "Failed retrievals: 1" in caplog.text
    assert "en: 1" in caplog.text
    assert "fr: 1" in caplog.text


def test_build_arg_parser():
    parser = doc_retriever.build_arg_parser()
    args = parser.parse_args(["--query", "test", "--start-date", "2024-12-01", "--end-date", "2024-12-31"])
    assert args.query == "test"
    assert args.start_date == "2024-12-01"
    assert args.end_date == "2024-12-31"


def test_main(monkeypatch, temp_dirs, dummy_articles):
    raw_dir, failed_log, index_file = temp_dirs

    class Args:
        def __init__(self, raw_dir, failed_log, index_file):
            self.query = "test"
            self.output = None
            self.no_save_json = False
            self.force_reprocess = False
            self.limit = 2
            self.start_date = "2024-12-01"
            self.end_date = "2024-12-31"
            self.collection_ids = None
            self.raw_dir = raw_dir
            self.failed_log = failed_log
            self.index_file = index_file
            self.label_studio_tasks = raw_dir / "labelstudio_tasks.json"

    args = Args(raw_dir, failed_log, index_file)
    monkeypatch.setattr(doc_retriever, "search_mediacloud_by_query", lambda *a, **kw: dummy_articles)
    monkeypatch.setattr(doc_retriever, "analyze_search_results", lambda articles: None)
    doc_retriever.main(args)
    # Check label studio file written
    assert (raw_dir / "labelstudio_tasks.json").exists()
    with open(raw_dir / "labelstudio_tasks.json") as f:
        tasks = json.load(f)
        assert len(tasks) == 1  # Only one with text
        assert tasks[0]["data"]["story_id"] == "1"
