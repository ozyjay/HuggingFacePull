"""Reviewed, acquisition-only download definitions."""

from __future__ import annotations

import dataclasses

from .hub import HubRef


@dataclasses.dataclass(frozen=True)
class AcquisitionPreset:
    name: str
    description: str
    ref: HubRef


def _qwen35_q8_preset(size: str, commit: str) -> AcquisitionPreset:
    repo_id = f"bartowski/Qwen_Qwen3.5-{size}-GGUF"
    filename = f"Qwen_Qwen3.5-{size}-Q8_0.gguf"
    return AcquisitionPreset(
        name=f"qwen3.5-{size.lower()}-q8_0",
        description=f"Qwen3.5 {size} Q8_0 GGUF",
        ref=HubRef(
            repo_id=repo_id,
            revision=commit,
            expected_commit=commit,
            allow_patterns=(filename,),
            # Keep the application's default reliable HTTP transfer path explicit.
            xet_enabled=False,
        ),
    )


PRESETS: dict[str, AcquisitionPreset] = {
    preset.name: preset
    for preset in (
        _qwen35_q8_preset("0.8B", "f36b1ea49a332ede8fe5f389bbf5b3575ef71f48"),
        _qwen35_q8_preset("2B", "7d26695454df6de5fbcce2e58681e62dae06ce43"),
        _qwen35_q8_preset("4B", "4168f45a16a1290d65a4ec0fa312ae917a4c15d6"),
        _qwen35_q8_preset("9B", "182be2fd6c7bc44887d88a91cb03ff009cc9f549"),
    )
}


def get_preset(name: str) -> AcquisitionPreset:
    try:
        return PRESETS[name]
    except KeyError as error:
        raise KeyError(f"Unknown preset: {name}") from error
