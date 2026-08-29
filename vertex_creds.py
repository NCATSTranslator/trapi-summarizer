## This module builds the Vertex acct creds.
## We expect non-secret creds (acct name, etc.) to be present in the repo in an on-disk
## file in the config directory.
## Secret values (the private key and the private key id) are injected either via
## a secrets file or an env var--the config loader is able to handle both options.
##
## Here are the expected env vars
##     GCP_SA_PRIVATE_KEY      the PEM private key
##     GCP_SA_PRIVATE_KEY_ID   the key id
##
## The private key contains newlines. When provided in a file, express as:
##     { ...
##         "private_key": "-----BEGIN PRIVATE KEY-----\nMAsDfjr...\n34SXwlek....",
##       ...
##     }
## When provided as an env var, at least in bash, continue to specify exactly as
## above, not as "...\\n", because "\n" is not a special sequence in bash double
## quotes; therefore the \ is self-quoting.
##


import os
import json
from google.oauth2 import service_account

LOCATION = "us-east5"
_SA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "configurations", "gcp-sa.json")
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def credentials_from(sa_info: dict, private_key: str, private_key_id: str | None = None):
    """Return ``(credentials, project_id)`` from SA metadata plus a private key.

    ``sa_info`` is the non-secret service-account record; it is copied rather
    than mutated so the caller's (frozen) config is never written into.
    """
    if not private_key:
        raise RuntimeError(
            "No service-account private key supplied. Provide it via the secrets "
            "configuration (or GCP_SA_PRIVATE_KEY); it is merged in-process and "
            "never written to disk or logged.")

    # Env vars and JSON-escaped values typically carry the PEM with literal
    # backslash-n; restore real newlines.
    info = dict(sa_info)
    info["private_key"] = private_key.replace("\\n", "\n")
    if private_key_id:
        info["private_key_id"] = private_key_id

    creds = service_account.Credentials.from_service_account_info(info, scopes=list(_SCOPES))
    return creds, info["project_id"]


def load_credentials(sa_path: str = _SA_PATH):
    """Return ``(credentials, project_id)`` for tools that have no config loaded.

    Reads the non-secret SA fields from ``sa_path`` and the key material from the
    environment. Raises RuntimeError if the private-key env var is absent, so the
    failure is obvious rather than surfacing later as an opaque auth error.
    """
    with open(sa_path) as f:
        sa_info = json.load(f)  # non-secret fields only

    private_key = os.environ.get("GCP_SA_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError(
            "GCP_SA_PRIVATE_KEY is not set. Export the service-account private key "
            "(and ideally GCP_SA_PRIVATE_KEY_ID) in the environment; they are merged "
            "in-process and never written to disk or logged.")

    return credentials_from(sa_info, private_key, os.environ.get("GCP_SA_PRIVATE_KEY_ID"))
