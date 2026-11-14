from pathlib import Path

from .errors import SubmissionError
from .features import SimilarityConfig
from .parsing import decode_json, require_fields, require_object, require_string


def load_similarity_config(path: Path | None) -> SimilarityConfig:
    if path is None:
        return SimilarityConfig()
    raw = require_object(decode_json(path.read_text(encoding="utf-8")), "config")
    require_fields(raw, frozenset({"version", "extra_templates"}))
    templates = raw["extra_templates"]
    if not isinstance(templates, list):
        raise SubmissionError("schema_error", "extra_templates", "Ожидается список строк")
    return SimilarityConfig(
        require_string(raw["version"], "version"),
        tuple(require_string(value, "extra_templates[]") for value in templates),
    )
