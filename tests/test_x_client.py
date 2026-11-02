"""Source contract tests: no X credentials or external network calls."""

from urllib.error import HTTPError

import pytest

from src.ingestion import x_client
from src.ingestion.x_client import SearchPage, XCollectionError, XReplyCollector, parse_post_id


@pytest.mark.parametrize("host", ["x.com", "www.x.com", "twitter.com"])
def test_post_urls(host: str) -> None:
    assert parse_post_id(f"https://{host}/example/status/123?s=20") == "123"


@pytest.mark.parametrize(
    "url",
    [
        "http://x.com/u/status/123",
        "https://x.com.evil/u/status/123",
        "https://x.com@evil/u/status/123",
        "https://x.com/u/status/abc",
        "https://x.com/home",
    ],
)
def test_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValueError):
        parse_post_id(url)


def test_pagination_deduplicates_and_filters_direct_english_replies() -> None:
    requests: list[dict[str, str]] = []

    def fetch(token: str, params: dict[str, str]) -> SearchPage:
        assert token == "test-token"
        requests.append(params)
        if len(requests) == 1:
            return SearchPage.model_validate(
                {
                    "data": [
                        {
                            "id": "1",
                            "text": "hello",
                            "lang": "en",
                            "referenced_tweets": [{"type": "replied_to", "id": "123"}],
                        },
                        {
                            "id": "2",
                            "text": "merhaba",
                            "lang": "tr",
                            "referenced_tweets": [{"type": "replied_to", "id": "123"}],
                        },
                        {
                            "id": "3",
                            "text": "nested",
                            "lang": "en",
                            "referenced_tweets": [{"type": "replied_to", "id": "1"}],
                        },
                    ],
                    "meta": {"next_token": "page2"},
                }
            )
        return SearchPage.model_validate(
            {
                "data": [
                    {
                        "id": "1",
                        "text": "hello",
                        "lang": "en",
                        "referenced_tweets": [{"type": "replied_to", "id": "123"}],
                    },
                    {
                        "id": "4",
                        "text": "thanks",
                        "lang": "en",
                        "referenced_tweets": [{"type": "replied_to", "id": "123"}],
                    },
                ]
            }
        )

    result = XReplyCollector("test-token", fetch).collect("https://x.com/u/status/123")
    assert [reply.id for reply in result.replies] == ["1", "4"]
    assert result.pages_requested == 2
    assert result.stopped_by_limit is False
    assert requests[1]["next_token"] == "page2"
    assert "lang:en" in requests[0]["query"]


def test_page_limit_bounds_requests_and_marks_truncation() -> None:
    def fetch(token: str, params: dict[str, str]) -> SearchPage:
        return SearchPage.model_validate({"meta": {"next_token": "more"}})

    result = XReplyCollector("token", fetch).collect("https://x.com/u/status/123", max_pages=1)
    assert result.pages_requested == 1
    assert result.stopped_by_limit is True
    assert result.replies == []


def test_partial_api_errors_are_not_silently_saved() -> None:
    def fetch(token: str, params: dict[str, str]) -> SearchPage:
        return SearchPage.model_validate({"errors": [{"detail": "unavailable"}]})

    with pytest.raises(XCollectionError, match="partial"):
        XReplyCollector("token", fetch).collect("https://x.com/u/status/123")


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500])
def test_http_errors_do_not_expose_token(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise HTTPError("https://api.x.com", status, "secret-token", {}, None)

    monkeypatch.setattr(x_client, "urlopen", fail)
    with pytest.raises(XCollectionError) as error:
        XReplyCollector("secret-token").collect("https://x.com/u/status/123")
    assert "secret-token" not in str(error.value)


def test_missing_token_fails_before_network() -> None:
    with pytest.raises(ValueError, match="X_BEARER_TOKEN"):
        XReplyCollector("")


def test_comment_limit_stops_without_fetching_another_page() -> None:
    calls = 0

    def fetch(token: str, params: dict[str, str]) -> SearchPage:
        nonlocal calls
        calls += 1
        return SearchPage.model_validate(
            {
                "data": [
                    {
                        "id": str(index),
                        "text": "hello",
                        "lang": "en",
                        "referenced_tweets": [{"type": "replied_to", "id": "123"}],
                    }
                    for index in range(5)
                ],
                "meta": {"next_token": "more"},
            }
        )

    result = XReplyCollector("token", fetch).collect("https://x.com/u/status/123", max_comments=2)
    assert len(result.replies) == 2
    assert result.stopped_by_limit is True
    assert calls == 1


def test_repeated_pagination_token_fails() -> None:
    def fetch(token: str, params: dict[str, str]) -> SearchPage:
        return SearchPage.model_validate({"meta": {"next_token": "same"}})

    with pytest.raises(XCollectionError, match="repeated"):
        XReplyCollector("token", fetch).collect("https://x.com/u/status/123")
