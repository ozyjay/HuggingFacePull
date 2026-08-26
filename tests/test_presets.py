import huggingface_pull.presets as presets


def test_qwen35_q8_presets_map_to_reviewed_repositories_and_files():
    expected = {
        "qwen3.5-0.8b-q8_0": (
            "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
            "Qwen_Qwen3.5-0.8B-Q8_0.gguf",
            "f36b1ea49a332ede8fe5f389bbf5b3575ef71f48",
        ),
        "qwen3.5-2b-q8_0": (
            "bartowski/Qwen_Qwen3.5-2B-GGUF",
            "Qwen_Qwen3.5-2B-Q8_0.gguf",
            "7d26695454df6de5fbcce2e58681e62dae06ce43",
        ),
        "qwen3.5-4b-q8_0": (
            "bartowski/Qwen_Qwen3.5-4B-GGUF",
            "Qwen_Qwen3.5-4B-Q8_0.gguf",
            "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
        ),
        "qwen3.5-9b-q8_0": (
            "bartowski/Qwen_Qwen3.5-9B-GGUF",
            "Qwen_Qwen3.5-9B-Q8_0.gguf",
            "182be2fd6c7bc44887d88a91cb03ff009cc9f549",
        ),
    }

    assert set(presets.PRESETS) == set(expected)
    for name, (repo_id, filename, commit) in expected.items():
        ref = presets.get_preset(name).ref
        assert ref.repo_id == repo_id
        assert ref.revision == commit
        assert ref.expected_commit == commit
        assert ref.allow_patterns == (filename,)
        assert ref.xet_enabled is False
