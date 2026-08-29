# What this module does

This module produces AI-generated summaries of portions of TRAPI response messages. At present it only targets OpenAI's Responses API.

It takes as input a TRAPI response message and a set of indices that indicate which elements of the `results` array to include in the summary.

It produces a markdown-based preliminary file that collects the relevant data from the entire response message (particularly from the knowledge graph); and passes that preliminary summary to OpenAI along with instructions on how to produce a good summary.

To use this module, you must provide your own OpenAI API key.

## How to install

- Create the dirs `dependencies` and `data` by hand in the project's top level dir
- Obtain the `graphwerks` and `openai_lib` packages by asking the author (these will be published on PyPI at some point)
  and place them in `dependencies/`
- `$ make install`

## Configuration

Runtime settings for the Vertex server are loaded from JSON files under `configurations/`,
one per environment (`dev`, `ci`, `test`, `production`), following the same scheme as the
`ui-be` repo.

- The base file is either given explicitly (`--config configurations/dev.json`) or derived
  from the `APP_ENVIRONMENT` environment variable.
- An optional second file is deep-merged on top of the base. `configurations/local-overrides.json`
  (gitignored) is the place for per-developer settings; at minimum it should set `document_root`
  to your checkout, since the committed files carry the container path `/app`.
- A `_load_<name>` key is replaced by the contents of the JSON file it names — resolved against
  `document_root` — under the key `<name>`.

### Secrets

No module reads a credential from the environment on its own. `configurations/secrets/secrets.json`
(gitignored, injected at deploy time) is pulled in via `_load_secrets` and is available to the app as
`config['secrets']`. Any value in the tree may be written as `{"_env": "VAR_NAME"}`, which is
resolved from the environment at startup. Add `"required": false` to tolerate a missing variable.

The Vertex service account is assembled in memory from two sources that are never combined on
disk: the non-secret record in `configurations/gcp-sa.json` and the private key from secrets.

## How to use

- `$ export OPENAI_KEY='<your openai key>'`
  - Note that to use o3, you must have verified your account
- `$ python driver.py -i <pk | path to file> --list N1,N2,N3 --template <path to yaml config> [--loop | --run]`
  - `-i <pk | path to file>`: If the PK of a pre-run query is provided, it will first be downloaded to a file
    in `data/`. If a file name is provided, it must be valid JSON conforming to a TRAPI Response message.
  - `--list n1[,n2,n3]`: 0-based indices of elements of the TRAPI `response` array to summarize.
  - `--template <yaml file>`: A file providing parameters for the OpenAI Responses API call. See `yaml/` for examples
  - `--run`: Provide this when your API call doesn't use tools and will produce a single response
  - `--loop`: Provide this when your API call contains a tool-calling loop (non-streaming)
  - `--stream`: Provide this to retrieve streaming results from a tool-calling loop
  - `--chunk N`: Number of tokens of streaming textual output to collect before yielding output to the caller (default 10)

If you provide none of `--loop`, `--run`, or `--stream`, the program will print the query request payload and exit without
submitting it to OpenAI.

### The Vertex server

`vertex_ui_server.py` serves the same endpoints as `ui_server.py` but against Vertex-hosted
models. Provider, model, generation parameters, templates, and listen address all come from
the configuration file, so the only launch argument is the config itself:

- `$ python vertex_ui_server.py --config configurations/dev.json configurations/local-overrides.json`

It has no module-level `app`, so it must be launched this way rather than via `uvicorn`.
