from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


RepoType = Literal["model", "dataset", "space"]
RepoId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueueRequest(BaseModel):
    repo_id: RepoId
    revision: str = "main"
    repo_type: RepoType = "model"
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    local_dir: str | None = None


class InstalledRemoveRequest(BaseModel):
    repo_id: RepoId
    revision: str = "main"
    repo_type: RepoType = "model"


class CleanupRequest(BaseModel):
    include_partials: bool = False
    older_than_days: int = Field(default=7, ge=0)
