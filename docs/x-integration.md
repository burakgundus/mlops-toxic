# X replies and batch inference

This increment uses the official X API rather than HTML scraping. It keeps the
existing English model. No new dependencies or lockfile changes are needed.

## Collect replies

You need a developer application with recent-search access, a bearer token and
any applicable credits. Set a spending limit in the X developer console. The
collector has request limits, not a monetary spending guarantee.

From the repository root, replace the example with a real post URL:

```powershell
$xSecret = Read-Host 'X Bearer Token' -AsSecureString
$env:X_BEARER_TOKEN = [System.Net.NetworkCredential]::new('', $xSecret).Password
poetry run python -m entrypoint.collect_x 'https://x.com/USERNAME/status/POST_ID' --max-comments 20 --max-pages 1 --output data/01-raw/x-replies-demo.json
Remove-Item Env:X_BEARER_TOKEN
```

Keep the token local; do not commit or share it. This command makes network
requests only when explicitly run. No automatic retries or posting actions occur.
Defaults: 50 comments, three pages. Hard ceilings: 100 comments, ten pages.

Only direct replies classified as English by X are retained. Nested replies are
excluded. Recent search covers seven days, not a complete archive. A zero-result
search can reflect no matches, old content or unavailable content. Even when
`stopped_by_limit` is false, completeness is not guaranteed.

The JSON stores the post ID, UTC collection time, search window, pages requested,
limit status and reply IDs/text/source URLs. Profile details are not requested.
Failed pages abort without saving a partial file. Existing files are not replaced.
`data/*/*` is already Git-ignored; do not publish real replies as test fixtures.

## Analyse comments

Rebuild the API after applying this increment:

```powershell
docker compose -f docker/docker-compose.yml up -d --build api
Invoke-RestMethod http://localhost:8000/ready
```

Test without X credentials:

```powershell
$payload = @{ comments = @(@{text='Thank you for your help.'}, @{text='You are a terrible idiot.'}) } | ConvertTo-Json -Depth 4
Invoke-RestMethod http://localhost:8000/predict/batch -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($payload)) | ConvertTo-Json -Depth 6
```

Analyse collected replies:

```powershell
$collection = Get-Content data/01-raw/x-replies-demo.json -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($collection.replies).Count -eq 0) { throw 'No replies collected. Check the post or access.' }
$payload = @{ comments = @($collection.replies | ForEach-Object { @{text=$_.text} }) } | ConvertTo-Json -Depth 4
$analysis = Invoke-RestMethod http://localhost:8000/predict/batch -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($payload))
$analysis | ConvertTo-Json -Depth 6
```

Result `i` corresponds to reply `i`. Each result uses the existing `/predict`
schema and 0.5 cutoff. Batches allow 1–100 comments, each 1–20,000 characters.
Whitespace-only content rejects the whole batch with 400, invalid schema/size
returns 422, missing model returns 503, and invalid model output returns 500 with
no partial predictions. Collected text is untrusted input, never instructions.

## Validate

```powershell
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy --strict src/api/main.py src/ingestion entrypoint/collect_x.py
poetry run pytest -q
```

Tests use synthetic source responses without live X access or charges. Live
credentials/access and model quality on real X replies require separate checks.
Next: a UI with a source/score table and filters, followed by evaluation on a
manually labelled sample of real English replies.

References:
- https://docs.x.com/x-api/posts/search-recent-posts
- https://docs.x.com/x-api/posts/search/integrate/operators
- https://docs.x.com/x-api/getting-started/pricing
