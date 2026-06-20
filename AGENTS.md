# Agent Notes

## Python and Test Dependencies

- This machine uses `pyenv`; prefer `python3` from the active `pyenv` version for Python tooling.
- The project `.venv` installs development dependencies with `.[dev]`, including `httpx2`.
- If running tests directly with `python3 -m pytest` instead of `./scripts/test.ps1`, ensure the active `pyenv` interpreter also has `httpx2` installed:

```bash
python3 -m pip install 'httpx2>=2.4,<3'
```

Without `httpx2` in the interpreter that imports `starlette.testclient`, tests may still pass but emit:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```
