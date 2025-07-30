import logging
import sys
import os
import pytest
import tempfile
import shutil
import json
from pathlib import Path
import datetime as dt
from unittest.mock import patch, mock_open

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
    caplog.set_level("DEBUG")
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


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_load_articles_index_corrupted_file(self, temp_dirs):
        """Test loading corrupted index file"""
        _, _, index_file = temp_dirs
        # Create corrupted JSON file
        with open(index_file, "w") as f:
            f.write("invalid json {")

        result = doc_retriever.load_articles_index(index_file)
        assert result == {}

    def test_save_articles_index_permission_error(self, temp_dirs, monkeypatch):
        """Test saving index when file is not writable"""
        _, _, index_file = temp_dirs

        # Mock open to raise PermissionError
        mock_file = mock_open()
        mock_file.side_effect = PermissionError("Access denied")

        with patch("builtins.open", mock_file):
            # Should not raise exception, just log error
            doc_retriever.save_articles_index({"test": "data"}, index_file)

    def test_search_mediacloud_api_error(self, monkeypatch, temp_dirs):
        """Test API error handling"""
        raw_dir, _, _ = temp_dirs

        class FailingSearchApi:
            def story_list(self, *args, **kwargs):
                raise Exception("API Error")

        monkeypatch.setattr(doc_retriever, "SEARCH_API", FailingSearchApi())

        articles = doc_retriever.search_mediacloud_by_query(
            query="test",
            articles_index={},
            raw_articles_dir=raw_dir,
        )
        assert articles == []


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_is_article_retrieved_edge_cases(self):
        """Test various edge cases for article retrieval checking"""
        # Empty index
        assert not doc_retriever.is_article_retrieved("1", {})

        # Article with unknown status
        index = {"1": {"status": "unknown", "text_length": 10}}
        assert not doc_retriever.is_article_retrieved("1", index)

        # Article with success status but zero length
        index = {"1": {"status": str(doc_retriever.ArticleStatus.SUCCESS), "text_length": 0}}
        assert not doc_retriever.is_article_retrieved("1", index)

        # Missing text_length field
        index = {"1": {"status": str(doc_retriever.ArticleStatus.SUCCESS)}}
        assert not doc_retriever.is_article_retrieved("1", index)

    def test_search_mediacloud_empty_results(self, monkeypatch, temp_dirs):
        """Test handling of empty search results"""
        raw_dir, _, _ = temp_dirs

        class EmptySearchApi:
            def story_list(self, *args, **kwargs):
                return [[]]  # Empty results

        monkeypatch.setattr(doc_retriever, "SEARCH_API", EmptySearchApi())

        articles = doc_retriever.search_mediacloud_by_query(
            query="test",
            articles_index={},
            raw_articles_dir=raw_dir,
        )
        assert articles == []

    def test_search_mediacloud_no_url(self, monkeypatch, temp_dirs):
        """Test handling of stories without URLs"""
        raw_dir, _, _ = temp_dirs

        stories = [{"id": "1"}]
        story_data = {"1": {"title": "Title", "text": "Text"}}  # No URL

        dummy_api = DummySearchApi(stories, story_data)
        monkeypatch.setattr(doc_retriever, "SEARCH_API", dummy_api)

        articles = doc_retriever.search_mediacloud_by_query(
            query="test",
            articles_index={},
            raw_articles_dir=raw_dir,
        )
        assert len(articles) == 0


class TestCollectionIds:
    """Test collection IDs functionality"""

    def test_search_with_collection_ids(self, monkeypatch, temp_dirs, caplog):
        """Test searching with collection IDs"""
        raw_dir, _, _ = temp_dirs

        class CollectionAwareSearchApi:
            def story_list(self, query, start, end, collection_ids=None):
                if collection_ids == [123, 456]:
                    return [[{"id": "1"}]]
                return [[]]

            def story(self, story_id):
                return {
                    "url": "http://example.com/1",
                    "title": "Title 1",
                    "text": "Text 1",
                    "publish_date": "2024-12-01",
                    "media_id": "m1",
                    "language": "en",
                }

        monkeypatch.setattr(doc_retriever, "SEARCH_API", CollectionAwareSearchApi())

        caplog.set_level(logging.DEBUG)

        articles = doc_retriever.search_mediacloud_by_query(
            query="test",
            collection_ids=[123, 456],
            articles_index={},
            raw_articles_dir=raw_dir,
        )

        assert len(articles) == 1
        print(caplog.text)
        assert "Query restricted to 2(s) collections" in caplog.text


class TestAnalyzeSearchResults:
    """Test search results analysis"""

    def test_analyze_empty_results(self, caplog):
        """Test analysis with no articles"""
        caplog.set_level("DEBUG")
        doc_retriever.analyze_search_results([])
        assert "No articles to analyze" in caplog.text

    def test_analyze_detailed_results(self, caplog):
        """Test analysis with various article types"""
        articles = [
            {"status": doc_retriever.ArticleStatus.SUCCESS, "text": "a" * 100, "language": "en"},
            {"status": doc_retriever.ArticleStatus.SUCCESS, "text": "b" * 200, "language": "en"},
            {"status": doc_retriever.ArticleStatus.FAILED_NO_TEXT, "text": "", "language": "fr"},
            {"status": doc_retriever.ArticleStatus.SUCCESS, "text": "c" * 50, "language": "es"},
        ]

        caplog.set_level(logging.DEBUG)
        doc_retriever.analyze_search_results(articles)

        assert "Total articles found: 4" in caplog.text
        assert "Successful retrievals: 3" in caplog.text
        assert "Failed retrievals: 1" in caplog.text
        assert "Success rate: 75.0%" in caplog.text
        assert "Mean: 117" in caplog.text  # (100+200+50)/3
        assert "en: 2" in caplog.text
        assert "fr: 1" in caplog.text
        assert "es: 1" in caplog.text


class TestArgumentParser:
    """Test argument parser edge cases"""

    def test_parser_missing_required_args(self):
        """Test parser with missing required arguments"""
        parser = doc_retriever.build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required --query

    def test_parser_invalid_collection_ids(self):
        """Test parser with invalid collection IDs"""
        parser = doc_retriever.build_arg_parser()

        # This should work - collection-ids accepts multiple integers
        args = parser.parse_args(
            [
                "--query",
                "test",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--collection-ids",
                "123",
                "456",
                "789",
            ]
        )
        assert args.collection_ids == [123, 456, 789]

    def test_parser_default_values(self):
        """Test parser default values"""
        parser = doc_retriever.build_arg_parser()
        args = parser.parse_args(["--query", "test", "--start-date", "2024-01-01", "--end-date", "2024-12-31"])

        assert args.limit == 100
        assert args.raw_dir == doc_retriever.RAW_ARTICLES_DIR
        assert args.failed_log == doc_retriever.FAILED_URLS_LOG
        assert not args.no_save_json
        assert not args.force_reprocess


class TestMainFunction:
    """Test main function edge cases"""

    def test_main_invalid_dates(self, temp_dirs, caplog):
        """Test main function with invalid date formats"""
        raw_dir, failed_log, index_file = temp_dirs

        class Args:
            def __init__(self):
                self.query = "test"
                self.output = None
                self.no_save_json = False
                self.force_reprocess = False
                self.limit = 2
                self.start_date = "invalid-date"
                self.end_date = "2024-12-31"
                self.collection_ids = None
                self.raw_dir = raw_dir
                self.failed_log = failed_log
                self.index_file = index_file
                self.label_studio_tasks = None

        args = Args()

        caplog.set_level("ERROR")
        doc_retriever.main(args)

        assert "Invalid date format" in caplog.text

    def test_main_no_output_specified(self, temp_dirs, caplog, monkeypatch):
        """Test main function when no output format is specified"""
        raw_dir, failed_log, index_file = temp_dirs

        class Args:
            def __init__(self):
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
                self.label_studio_tasks = None

        args = Args()
        monkeypatch.setattr(doc_retriever, "search_mediacloud_by_query", lambda *a, **kw: [])

        caplog.set_level("WARNING")
        doc_retriever.main(args)

        assert "No output format specified" in caplog.text
        assert "Defaulting to Label Studio JSON" in caplog.text


# Performance and Integration Tests
class TestIntegration:
    """Integration tests"""

    def test_full_pipeline_with_csv_output(self, temp_dirs, monkeypatch):
        """Test complete pipeline with CSV output"""
        raw_dir, failed_log, index_file = temp_dirs
        csv_output = raw_dir / "output.csv"

        # Mock articles
        test_articles = [
            {
                "url": "http://example.com/1",
                "title": "Title 1",
                "text": "Text 1",
                "story_id": "1",
                "status": doc_retriever.ArticleStatus.SUCCESS,
                "publish_date": "2024-12-01",
                "media_id": "m1",
                "language": "en",
                "query": "test",
                "retrieved_at": "2024-12-01T12:00:00",
            }
        ]

        class Args:
            def __init__(self):
                self.query = "test"
                self.output = csv_output
                self.no_save_json = False
                self.force_reprocess = False
                self.limit = 2
                self.start_date = "2024-12-01"
                self.end_date = "2024-12-31"
                self.collection_ids = None
                self.raw_dir = raw_dir
                self.failed_log = failed_log
                self.index_file = index_file
                self.label_studio_tasks = None

        args = Args()
        monkeypatch.setattr(doc_retriever, "search_mediacloud_by_query", lambda *a, **kw: test_articles)

        doc_retriever.main(args)

        # Check CSV was created
        assert csv_output.exists()

        # Check content
        import pandas as pd

        df = pd.read_csv(csv_output)
        assert len(df) == 1
        assert df.iloc[0]["story_id"] == 1


# Add parameterized tests for better coverage
@pytest.mark.parametrize(
    "status,text_length,expected",
    [
        (doc_retriever.ArticleStatus.SUCCESS, 10, True),
        (doc_retriever.ArticleStatus.SUCCESS, 0, False),
        (doc_retriever.ArticleStatus.FAILED_NO_TEXT, 0, False),
        (doc_retriever.ArticleStatus.UNKNOWN, 10, False),
    ],
)
def test_is_article_retrieved_parametrized(status, text_length, expected):
    """Parametrized test for is_article_retrieved"""
    index = {"1": {"status": status, "text_length": text_length}}
    assert doc_retriever.is_article_retrieved("1", index) == expected
