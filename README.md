# Flickr GPT Autotagger

Flickr GPT Autotagger generates titles, descriptions, and keywords for Flickr photos with an OpenAI-compatible vision model. It supports safe dry-runs, resumable checkpoints, bounded concurrent analysis, existing-tag preservation, cost limits, and explicit apply operations.

## Safety model

Running the command without `--apply` never changes Flickr. It writes a JSON run report and a durable JSON Lines checkpoint that can be reviewed or applied later.

```bash
# Analyse and create a plan only
python flickr-autotagger.py

# Analyse and update Flickr
python flickr-autotagger.py --apply

# Apply an existing report or checkpoint without calling OpenAI
python flickr-autotagger.py --apply-plan updated_metadata.json
```

Flickr tags are merged with existing tags by default. Set `TAG_MODE=replace` only when the generated keywords should replace every existing tag.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install --no-build-isolation --no-deps -e .
cp .env.example .env
```

Set these credentials in `.env`:

```dotenv
FLICKR_API_KEY=your_flickr_api_key
FLICKR_API_SECRET=your_flickr_api_secret
OPENAI_API_KEY=your_openai_api_key
```

If `FLICKR_OAUTH_TOKEN` is not supplied, the first local run performs interactive Flickr OAuth and writes a private token file. `FLICKR_TOKEN_FILE` changes its location.

## Custom OpenAI endpoint

Set `OPENAI_BASE_URL` to the base URL of OpenAI or an OpenAI-compatible service. Include the API version prefix expected by that service, but not `/chat/completions`.

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1

# Example local compatible endpoint
# OPENAI_BASE_URL=http://localhost:11434/v1
```

The configured endpoint receives the API key and Flickr image URLs, so only use a trusted service. Compatible services must support image URL content through Chat Completions. The following settings handle common compatibility differences:

- `OPENAI_INSTRUCTION_ROLE=developer|system`
- `OPENAI_TOKENS_PARAMETER=max_completion_tokens|max_tokens`
- `OPENAI_JSON_MODE=true|false`

Azure OpenAI is supported separately:

```dotenv
OPENAI_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
OPENAI_API_VERSION=your-supported-api-version
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_MODEL=your-deployment-name
```

## Configuration

All options can be placed in `.env` or supplied as environment variables. CLI arguments override the corresponding environment setting where available.

### OpenAI and generation

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `OPENAI_MODEL` | `gpt-5-mini` | Model or Azure deployment name |
| `OPENAI_TIMEOUT` | `60` | Request timeout in seconds |
| `OPENAI_MAX_RETRIES` | `2` | SDK retries for transient failures |
| `OPENAI_MAX_OUTPUT_TOKENS` | `800` | Output ceiling per analysis |
| `OPENAI_IMAGE_DETAIL` | `low` | `low`, `high`, or `auto` |
| `ANALYSIS_CONCURRENCY` | `3` | Maximum simultaneous model requests |
| `MAX_KEYWORDS` | `10` | Maximum generated keywords |
| `MAX_TITLE_CHARS` | `120` | Maximum stored title length |
| `MAX_DESCRIPTION_CHARS` | `1200` | Maximum stored description length |
| `METADATA_LANGUAGE` | `English` | Requested metadata language |
| `DESCRIPTION_STYLE` | `detailed, factual, and concise` | Description-writing guidance |
| `PROMPT_FILE` | unset | Custom prompt template path |

Custom prompts may use `{{MAX_KEYWORDS}}`, `{{MAX_TITLE_CHARS}}`, `{{MAX_DESCRIPTION_CHARS}}`, `{{LANGUAGE}}`, and `{{DESCRIPTION_STYLE}}`. See `prompt.example.txt`.

### Flickr selection and updates

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLICKR_PRIVACY_FILTER` | `1` | `all` or Flickr privacy level `1`-`5` |
| `FLICKR_IMAGE_URL` | `url_m` | Flickr image-size extra sent to the model |
| `FLICKR_PHOTOSET_ID` | unset | Legacy single photoset selector |
| `FLICKR_PHOTOSET_IDS` | unset | JSON or comma-separated photoset IDs |
| `PHOTOSET_ALLOWLIST` | unset | Allowed photoset IDs or exact titles |
| `PHOTOSET_DENYLIST` | unset | Denied photoset IDs or exact titles |
| `SKIP_PREFIX` | `["#", "@"]` | Photoset-title prefixes to skip |
| `DESCRIPTIONS_TO_ANALYZE` | camera filename defaults | Existing-description prefixes eligible for replacement |
| `DESCRIPTION_POLICY` | `missing-or-placeholder` | `missing-or-placeholder`, `always`, or `never` |
| `UPDATE_FIELDS` | `title,description,tags` | Flickr fields to update |
| `TAG_MODE` | `merge` | Preserve existing tags or `replace` them |

JSON-list configuration is parsed as JSON, never executable Python.

### Limits, checkpoints, and cost reporting

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_PHOTOS` | unset | Maximum number of new analyses |
| `MAX_TOTAL_COST` | unset | Stop after recorded cost reaches this amount |
| `RESUME` | `true` | Reuse matching cached analysis and skip updated photos |
| `CHECKPOINT_FILE` | `.autotagger-checkpoint.jsonl` | Append-only durable event log |
| `UPDATED_METADATA_FILE` | `updated_metadata.json` | Atomic report for the current run |
| `FAIL_ON_ERROR` | `false` | Return a non-zero exit code when any photo fails |

Cost reporting uses `OPENAI_COST_PER_1M_PROMPT_TOKEN`, `OPENAI_COST_PER_1M_COMPLETION_TOKEN`, and `OPENAI_VISION_COST_PER_IMAGE`. These values are manual estimates because models and compatible providers have different prices. A concurrent batch can exceed `MAX_TOTAL_COST` slightly before all in-flight usage is known.

## CLI overrides

```text
--apply                  Update Flickr after analysis
--apply-plan PATH        Apply an existing JSON/JSONL plan without OpenAI
--photoset-id ID         Select a photoset; may be repeated
--limit N                Limit new analyses
--max-cost AMOUNT        Set a recorded-cost limit
--concurrency N          Set OpenAI concurrency
--prompt-file PATH       Use a custom prompt
--no-resume              Ignore previous checkpoint state
--strict                 Fail the process if any photo fails
```

## GitHub Actions

The **Autotag Photos** workflow remains manually dispatched and applies changes by default. Clear **Apply generated metadata to Flickr** to perform a dry-run. The workflow:

- prevents overlapping autotag runs;
- verifies lint and tests before accessing Flickr;
- restores and saves the latest checkpoint;
- uploads the run report and checkpoint even after partial failure;
- installs the exact versions in `requirements.lock`.

Configure these GitHub secrets:

- `FLICKR_API_KEY`
- `FLICKR_API_SECRET`
- `FLICKR_OAUTH_TOKEN`
- `OPENAI_API_KEY`

Those four existing secrets are sufficient for the standard OpenAI setup: no repository variables or Azure secrets are required. Open **Autotag Photos**, choose **Run workflow**, and accept the defaults to run and apply the generated metadata as before.

Set the repository variable `OPENAI_BASE_URL` only when using a custom compatible endpoint. It is deliberately not a workflow-dispatch text input, preventing an operator from redirecting secrets to an arbitrary host. Optional provider variables are `OPENAI_PROVIDER`, `OPENAI_INSTRUCTION_ROLE`, `OPENAI_TOKENS_PARAMETER`, and `OPENAI_JSON_MODE`. Azure additionally uses the `AZURE_OPENAI_API_KEY` secret plus the `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_VERSION` variables.

## Development

```bash
pip install -r requirements.lock
pip install --no-build-isolation --no-deps -e .
ruff check .
ruff format --check .
pytest
```

The CI workflow tests Python 3.11 and 3.12. Regenerate the lock after changing dependency constraints with:

```bash
pip-compile --allow-unsafe --output-file=requirements.lock --strip-extras requirements-dev.txt
```
