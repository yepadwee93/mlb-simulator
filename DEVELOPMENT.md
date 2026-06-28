# Development Guide

Plain-language guide to the automated checks on this repo.

---

## What runs automatically and when

| What | When | Where you see it |
|------|------|-----------------|
| Code formatting (ruff) | Every commit (locally) | Terminal — blocks the commit if broken |
| Lint / code quality (ruff) | Every commit (locally) | Terminal — blocks the commit if broken |
| Secret scanner | Every commit (locally) | Terminal — blocks if API keys detected |
| File size check | Every commit (locally) | Terminal — blocks files over 500 KB |
| Full test suite (pytest) | Every push to GitHub | GitHub — green ✅ or red ❌ next to commit |
| Dependency vulnerability scan | Every push to GitHub | GitHub — green ✅ or red ❌ next to commit |

---

## One-time setup on a fresh machine

Run these once after cloning the repo:

```bash
pip install pre-commit ruff pytest
pre-commit install
```

That's it. After `pre-commit install`, the checks run automatically on every commit.

---

## What to do when a check fails

### "Ruff found errors" (lint)
The error message will say something like `web.py:42:5: E711 Comparison to None`.
- The line number tells you exactly where the problem is.
- Most of the time, running `ruff check --fix .` will auto-fix it for you.
- Then `git add` the fixed files and commit again.

### "Ruff format check failed"
Your code isn't formatted consistently.
- Run `ruff format .` to auto-fix all formatting.
- Then `git add` and commit again.

### "Potential secret detected" (detect-secrets)
A string in your code looks like an API key or password.
- If it's a false positive (not actually a secret), run:
  `detect-secrets scan > .secrets.baseline`
  then commit the updated `.secrets.baseline`.
- If it IS a real secret: remove it from the code, use an environment variable instead, and rotate the key immediately.

### "Test failed" (pytest on GitHub)
- Click the red ❌ on GitHub next to the commit.
- Click the failing job to see which test failed and what the error was.
- Fix the code, push again.

### "pip-audit found vulnerabilities"
A package you depend on has a known security issue.
- Run `pip-audit -r requirements.txt` locally to see details.
- Update the affected package in `requirements.txt` and test.

---

## How to add a new test

When you (or Claude) add a new feature, add a test like this:

1. Create or open a file in `tests/` — name it `test_<feature>.py`.
2. Write a function starting with `test_`:

```python
def test_my_new_thing(client):
    response = client.get("/my-new-route")
    assert response.status_code == 200
```

3. Run `pytest` locally to make sure it passes.
4. Commit — it'll run automatically on GitHub from then on.

---

## Seeing results on GitHub

After every push, go to your repo on GitHub:
- Look for a small circle next to the commit message — ✅ green = all good, ❌ red = something broke, 🟡 yellow = still running.
- Click the circle → "Details" to see exactly what failed.
