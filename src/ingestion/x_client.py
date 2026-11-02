"""Bounded, read-only collection of English replies using the official X API."""

import re
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError


class XCollectionError(RuntimeError):
    """A source request failed; no partial collection should be published."""


class Reference(BaseModel):
    type: str
    id: str


class XPost(BaseModel):
    id: str
    text: str
    lang: str | None = None
    referenced_tweets: list[Reference] = Field(default_factory=list)


class SearchMeta(BaseModel):
    next_token: str | None = None


class SearchPage(BaseModel):
    data: list[XPost] = Field(default_factory=list)
    meta: SearchMeta = Field(default_factory=SearchMeta)
    errors: list[dict[str, object]] = Field(default_factory=list)


class CollectedReply(BaseModel):
    id: str
    text: str
    source_url: str


class ReplyCollection(BaseModel):
    post_id: str
    collected_at: datetime
    language: str = "en"
    search_window: str = "recent_7_days"
    pages_requested: int
    stopped_by_limit: bool
    replies: list[CollectedReply]


def parse_post_id(post_url: str) -> str:
    """Accept X/Twitter status URLs, never arbitrary fetch destinations."""
    parts = urlsplit(post_url.strip())
    if parts.scheme != "https" or parts.netloc.lower() not in {
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
    }:
        raise ValueError("Provide an HTTPS x.com or twitter.com post URL.")
    match = re.fullmatch(r"/(?:[A-Za-z0-9_]+/status|i/web/status)/([0-9]{1,25})/?", parts.path)
    if match is None:
        raise ValueError("The URL must contain /status/<numeric post ID>.")
    return match.group(1)


def _request_page(token: str, params: dict[str, str]) -> SearchPage:
    request = Request(
        "https://api.x.com/2/tweets/search/recent?" + urlencode(params),
        headers={"Authorization": f"Bearer {token}", "User-Agent": "mlops-toxic/0.1"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return SearchPage.model_validate_json(response.read())
    except HTTPError as exc:
        messages = {
            401: "X rejected the access token.",
            403: "X search access is not enabled for this application.",
            402: "X API credits are required.",
            429: "X API rate limit reached. Try again later.",
        }
        raise XCollectionError(messages.get(exc.code, f"X API returned HTTP {exc.code}.")) from None
    except (URLError, TimeoutError, OSError, ValidationError):
        raise XCollectionError("X API could not be reached or returned invalid data.") from None


class XReplyCollector:
    def __init__(
        self,
        token: str,
        fetch_page: Callable[[str, dict[str, str]], SearchPage] = _request_page,
    ) -> None:
        if not token.strip():
            raise ValueError("Set X_BEARER_TOKEN before collecting replies.")
        self._token = token.strip()
        self._fetch_page = fetch_page

    def collect(
        self, post_url: str, *, max_comments: int = 50, max_pages: int = 3
    ) -> ReplyCollection:
        if not 1 <= max_comments <= 100 or not 1 <= max_pages <= 10:
            raise ValueError("Use 1..100 comments and 1..10 pages.")
        post_id = parse_post_id(post_url)
        # Select direct replies, not the entire descendant thread.
        params = {
            "query": f"in_reply_to_tweet_id:{post_id} lang:en -is:retweet",
            "tweet.fields": "lang,referenced_tweets",
            "max_results": str(max(10, max_comments)),
            "sort_order": "recency",
        }
        replies: dict[str, CollectedReply] = {}
        seen_tokens: set[str] = set()
        stopped = False
        pages = 0
        for _ in range(max_pages):
            page = self._fetch_page(self._token, params.copy())
            pages += 1
            if page.errors:
                raise XCollectionError("X returned a partial error; collection was not saved.")
            for post in page.data:
                is_direct = any(
                    ref.type == "replied_to" and ref.id == post_id for ref in post.referenced_tweets
                )
                if post.lang != "en" or not is_direct or not post.text.strip():
                    continue
                replies.setdefault(
                    post.id,
                    CollectedReply(
                        id=post.id,
                        text=post.text,
                        source_url=f"https://x.com/i/web/status/{post.id}",
                    ),
                )
                if len(replies) >= max_comments:
                    stopped = True
                    break
            if stopped or page.meta.next_token is None:
                break
            next_token = page.meta.next_token
            if next_token in seen_tokens:
                raise XCollectionError("X returned a repeated pagination token.")
            seen_tokens.add(next_token)
            params["next_token"] = next_token
            params["max_results"] = str(max(10, max_comments - len(replies)))
        else:
            stopped = True
        return ReplyCollection(
            post_id=post_id,
            collected_at=datetime.now(timezone.utc),
            pages_requested=pages,
            stopped_by_limit=stopped,
            replies=list(replies.values()),
        )
