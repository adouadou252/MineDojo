"""Participant configuration and private runtime copies; no contestant code runs here."""
from copy import deepcopy
import json
from pathlib import PurePosixPath
import stat
import zipfile


MODEL_FIELDS = {"name", "api_url", "api_key"}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"model", "agent"}:
        raise ValueError('config.json must contain exactly "model" and "agent"; no version field')
    model = config["model"]
    if not isinstance(model, dict) or set(model) != MODEL_FIELDS:
        raise ValueError("config.model must contain name, api_url and api_key")
    if any(not isinstance(model[key], str) for key in MODEL_FIELDS):
        raise ValueError("config.model fields must be strings")
    if len(model["name"]) > 160 or len(model["api_url"]) > 2048 or len(model["api_key"]) > 4096:
        raise ValueError("config.model fields exceed the size limit")
    if not isinstance(config["agent"], dict):
        raise ValueError("config.agent must be an object")
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("config.json must contain finite JSON values") from None
    return deepcopy(config)


def package_config(archive):
    entries = archive.infolist()
    names = set()
    if len(entries) > 256 or sum(entry.file_size for entry in entries) > 64 * 1024 * 1024:
        raise ValueError("agent package exceeds the extracted size limit")
    for entry in entries:
        name = PurePosixPath(entry.filename)
        if (name.is_absolute() or not name.parts or ".." in name.parts or "\\" in entry.filename
                or stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK
                or str(name) in names):
            raise ValueError("agent package contains an unsafe or duplicate path")
        names.add(str(name))
    if "agent.py" not in archive.namelist() or "config.json" not in archive.namelist():
        raise ValueError("agent.py and config.json must be at the ZIP root")
    if archive.getinfo("config.json").file_size > 64 * 1024:
        raise ValueError("agent config.json exceeds 64 KB")
    try:
        return validate_config(json.loads(archive.read("config.json").decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("agent config.json must be valid UTF-8 JSON") from None


def runtime_package(source, destination, model):
    """Replace only model settings in an ephemeral ZIP; source is never modified."""
    with zipfile.ZipFile(source) as original:
        config = package_config(original)
        config["model"] = deepcopy(model)
        validate_config(config)
        if not all(model[key].strip() for key in MODEL_FIELDS):
            raise ValueError("organizer must configure the evaluation model, proxy URL and token")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as runtime:
            for entry in original.infolist():
                if entry.filename == "config.json":
                    runtime.writestr(entry, json.dumps(config, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                else:
                    with original.open(entry) as src, runtime.open(entry, "w") as dst:
                        while chunk := src.read(65536):
                            dst.write(chunk)
