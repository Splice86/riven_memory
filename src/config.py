"""Config system for riven-memory - layered config with env override support.

Precedence (highest wins):
    1. Env var (RV_* prefix, __ for nesting: RV_LLM__API_KEY)
    2. Non-template YAML (secrets.yaml, config.yaml)  (user overrides)
    3. Template YAML (secrets_template.yaml)         (fallback if no user file)
    4. Hardcoded default passed to get()

Template convention:
    If <name>_template.yaml exists, it serves as fallback.
    If <name>.yaml exists (without _template), it wins.
    Example: secrets_template.yaml → check secrets.yaml first, fallback to template.
"""

import os
import copy
import yaml


def _find_memory_root() -> str:
    """Find the riven-memory project root (two levels up from src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_yaml(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val) if isinstance(val, dict) else val
    return result


def _coerce(value: str):
    """Coerce string env var to likely Python type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _env_override(data: dict, prefix: str = "RV") -> dict:
    """Override values with matching env vars (RV_SECTION__KEY)."""
    result = copy.deepcopy(data)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}_"):
            continue
        rest = env_key[len(prefix) + 1:]
        parts = rest.split("__")
        if not parts:
            continue
        current = result
        for part in parts[:-1]:
            part = part.lower()
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1].lower()] = _coerce(env_val)
    return result


def _resolve_template(template_path: str) -> str:
    """Return secrets_path if it exists, otherwise template."""
    if not template_path.endswith("_template.yaml"):
        return template_path
    base, ext = os.path.splitext(template_path)
    non_template = base + ext
    return non_template if os.path.exists(non_template) else template_path


# Lazy-loaded merged config
_merged: dict = {}
_loaded = False


def _load():
    global _merged, _loaded
    if _loaded:
        return
    root = _find_memory_root()

    merged = {}
    # Template YAMLs first (lowest precedence - fallback only)
    secrets_template = os.path.join(root, "secrets_template.yaml")
    merged = _deep_merge(merged, _load_yaml(secrets_template))
    
    # Non-template YAMLs override templates
    config_path = os.path.join(root, "config.yaml")
    merged = _deep_merge(merged, _load_yaml(config_path))
    
    # User secrets.yaml overrides both (if it exists)
    secrets_path = os.path.join(root, "secrets.yaml")
    merged = _deep_merge(merged, _load_yaml(secrets_path))
    
    # Env vars override everything (highest precedence)
    merged = _env_override(merged)

    _merged = merged
    _loaded = True


def get(key: str, default=None):
    """Get a config value by dotted key (e.g. 'llm.url', 'database.db_dir')."""
    _load()
    parts = key.split(".")
    current = _merged
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def reload():
    """Reload all config files (useful after changing env vars or yaml files)."""
    global _merged, _loaded
    _loaded = False
    _merged = {}
    _load()
