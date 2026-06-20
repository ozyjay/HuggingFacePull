from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from huggingface_pull.config import default_log_file


def write_log(message: str, /, **fields: Any) -> None:
    log_file = default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "fields": {key: str(value) for key, value in fields.items()},
    }
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
