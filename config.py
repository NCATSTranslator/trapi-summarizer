"""File-based configuration, modelled on the ui-be repo's ``lib/config.mjs``.

A run is configured by a base JSON file (one per environment) plus an optional
override file for per-developer deviation. Bootstrapping proceeds in four steps:

1. ``document_root`` is reconciled between base and override. It anchors every
   relative path in the config, so a committed ``dev.json`` can say ``/app``
   (the container path) while a developer's override points at their checkout.
2. ``_load_<name>`` keys are replaced by the JSON they point at, under the key
   ``<name>``. This is what composes the tree out of per-environment and
   env-invariant fragments. Unlike ui-be's, expansion here is recursive, so a
   fragment may itself contain ``_load_`` keys.
3. Base and override are each expanded independently, then deep-merged.
4. ``{"_env": "VAR"}`` leaves are replaced by the environment variable's value.
   This is the secrets-injection seam: deployments inject a secrets file with
   literal values, while a developer's file names env vars instead, and the
   consuming code reads the same ``config['secrets'][...]`` either way.

The result is frozen: config is read-only after bootstrap.
"""
import copy
import json
import os

APP_ENV_VAR = "APP_ENVIRONMENT"
CONFIG_DIR = "configurations"
_LOAD_PREFIX = "_load_"

# Dotted paths that must be present and non-None after bootstrap. Checked once,
# loudly, at startup rather than surfacing later as an opaque failure.
REQUIRED_PATHS = (
    "document_root",
    "server.host",
    "server.port",
    "llm.provider",
    "llm.model",
    "llm.templates.general",
    "llm.templates.gene",
    "vertex.location",
    "gcp_sa.project_id",
    "secrets.gcp.private_key",
)


class ConfigError(Exception):
    """Raised for any malformed or incomplete configuration."""


class FrozenDict(dict):
    """A dict that refuses mutation, so nothing can patch config at runtime."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("configuration is read-only after bootstrap")

    __setitem__ = __delitem__ = _immutable
    update = setdefault = pop = popitem = clear = _immutable

    # Copying yields plain, mutable dicts: copy it if you mean to change it.
    def __copy__(self):
        return {k: v for k, v in self.items()}

    def __deepcopy__(self, memo):
        return {k: copy.deepcopy(v, memo) for k, v in self.items()}


# ---------- public API ----------

def bootstrap(base_path: str, override_path: str | None = None) -> FrozenDict:
    """Load, compose, and validate the configuration tree."""
    config = _read_json(base_path)
    if override_path:
        overrides = _read_json(override_path)
        # The override's document_root wins if it has one; otherwise it inherits
        # the base's, so both trees resolve their relative paths identically.
        if overrides.get("document_root") is not None:
            config["document_root"] = overrides["document_root"]
        else:
            overrides["document_root"] = config.get("document_root")
        config = _merge(_expand(config), _expand(overrides))
    else:
        config = _expand(config)

    _resolve_env_refs(config)
    _validate(config)
    return _freeze(config)


def resolve_config_paths(paths: list[str] | None) -> tuple[str, str | None]:
    """Return ``(base, override)`` from explicit CLI paths, else from the env.

    Explicit paths always win. Absent them we fall back to the single env var
    the deployment platform gives us, mapping it to ``configurations/<env>.json``
    (the same mapping ui-be's entrypoint.sh performs in shell).
    """
    if paths:
        if len(paths) > 2:
            raise ConfigError(f"expected at most 2 config paths, got {len(paths)}")
        return paths[0], (paths[1] if len(paths) == 2 else None)

    env = os.environ.get(APP_ENV_VAR)
    if not env:
        raise ConfigError(
            f"No config file given and ${APP_ENV_VAR} is unset. Pass "
            f"--config <base.json> [<override.json>], or set {APP_ENV_VAR} to one "
            f"of the environments in {CONFIG_DIR}/."
        )
    return os.path.join(CONFIG_DIR, f"{env}.json"), None


def resolve_path(config, path: str) -> str:
    """Resolve a config-supplied relative path against ``document_root``."""
    if os.path.isabs(path):
        return path
    return os.path.join(config["document_root"], path)


def get(config, dotted: str, default=None):
    """Look up a dotted path, returning ``default`` if any segment is missing."""
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def redacted(config) -> dict:
    """A copy safe to log: every leaf under ``secrets`` becomes a presence marker."""
    out = copy.deepcopy(dict(config))
    if isinstance(out.get("secrets"), dict):
        out["secrets"] = _mark_presence(out["secrets"])
    return out


# ---------- internals ----------

def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file {path} is not valid JSON: {e}") from None


def _expand(config, _chain=()):
    """Replace ``_load_<name>`` keys with the parsed contents of the file named."""
    document_root = config.get("document_root") or os.getcwd()
    return _expand_node(config, document_root, _chain)


def _expand_node(node, document_root, chain):
    if isinstance(node, list):
        return [_expand_node(v, document_root, chain) for v in node]
    if not isinstance(node, dict):
        return node

    out = {}
    for key, value in node.items():
        if not key.startswith(_LOAD_PREFIX):
            out[key] = _expand_node(value, document_root, chain)
            continue
        if not value:
            continue
        path = os.path.abspath(
            value if os.path.isabs(value) else os.path.join(document_root, value))
        if path in chain:
            raise ConfigError(f"circular {_LOAD_PREFIX} reference: {path}")
        loaded = _expand_node(_read_json(path), document_root, chain + (path,))
        out[key[len(_LOAD_PREFIX):]] = loaded
    return out


def _merge(orig, overwrite):
    """Deep-merge ``overwrite`` onto ``orig``: dicts recurse, everything else replaces."""
    for key, value in overwrite.items():
        if isinstance(value, dict) and isinstance(orig.get(key), dict):
            _merge(orig[key], value)
        else:
            orig[key] = value
    return orig


def _resolve_env_refs(node):
    """Replace ``{"_env": "VAR"}`` leaves in place with the env var's value.

    Missing variables are an error unless the leaf sets ``"required": false``,
    in which case the value resolves to None.
    """
    if isinstance(node, list):
        for item in node:
            _resolve_env_refs(item)
        return
    if not isinstance(node, dict):
        return

    for key, value in node.items():
        if isinstance(value, dict) and "_env" in value:
            var = value["_env"]
            resolved = os.environ.get(var)
            if resolved is None and value.get("required", True):
                raise ConfigError(
                    f"config value '{key}' requires environment variable ${var}, "
                    f"which is unset")
            node[key] = resolved
        else:
            _resolve_env_refs(value)


def _validate(config):
    missing = [p for p in REQUIRED_PATHS if get(config, p) is None]
    if missing:
        raise ConfigError("missing required config values: " + ", ".join(missing))


def _freeze(node):
    if isinstance(node, dict):
        return FrozenDict({k: _freeze(v) for k, v in node.items()})
    if isinstance(node, list):
        return tuple(_freeze(v) for v in node)
    return node


def _mark_presence(node):
    if isinstance(node, dict):
        return {k: _mark_presence(v) for k, v in node.items()}
    return "<set>" if node else "<unset>"
