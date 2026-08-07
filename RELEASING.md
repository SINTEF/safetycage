# Releasing safetycage to PyPI

The checklist for cutting a release. Follow it in order.

**The governing rule: PyPI is irreversible, git is not.** Once a version is
uploaded that number can never be reused — not even after deleting the
release. Everything cheap and reversible happens first; the upload happens
last, after the metadata has already been checked and rehearsed.

---

## One-time setup

You need upload rights on the project *and* a token. A token only grants what
your account already has.

1. **Account.** Register at [pypi.org](https://pypi.org). 2FA is mandatory —
   authenticator app or security key.
2. **Maintainer rights.** Open
   <https://pypi.org/manage/project/safetycage/collaboration/>. If the page
   won't load, you don't have rights yet — ask an existing Owner to invite you.
   A valid token will still be rejected without this.
3. **Token.** Account settings → API tokens → Add API token. Set the scope to
   **Project: safetycage**, not "Entire account", so a leaked token can only
   affect this package.
4. **Copy it immediately.** It starts with `pypi-` and is shown exactly once.
   There is no way to retrieve it later, only to revoke and regenerate.

```bash
export UV_PUBLISH_TOKEN=pypi-AgEIcHlwaS5vcmc...
```

Prefer the environment variable over `uv publish --token ...`, which leaves
the credential in your shell history.

**TestPyPI is a separate service** with its own account, registration and
token. Get one from [test.pypi.org](https://test.pypi.org) for step 4.

---

## Releasing

### 1. Decide the version and write the changelog

Check what is actually on PyPI before choosing a number:

```bash
curl -s https://pypi.org/pypi/safetycage/json | python -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

Bump `version` in `pyproject.toml` to something strictly greater. Add a
`CHANGELOG.md` entry at the top in the existing format
(`## vX.Y.Z (DD/MM/YYYY)`, then `### Breaking` / `### Feature` / `### Fix`).

Call out breaking changes explicitly — a removed public function or a changed
save format belongs under `### Breaking`, not `### Fix`.

### 2. Confirm a clean tree

The build reads the working directory, not git, so uncommitted edits ship
silently. Commit everything first.

```bash
git status --short   # must be empty
```

Don't forget `uv.lock` if dependencies moved.

### 3. Build

```bash
rm -rf dist/         # see "Stale dist/" below — this bites people
uv build
ls dist/             # expect exactly one .whl and one .tar.gz, at the new version
```

### 4. Check the metadata

```bash
uvx twine check dist/*
```

Then rehearse the whole upload against TestPyPI, using your TestPyPI token:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-...
uv run --with safetycage --index-url https://test.pypi.org/simple/ \
       --no-project -- python -c "import safetycage; print(safetycage.__file__)"
```

This is the last point at which a mistake is free.

### 5. Push

```bash
git push
```

Before the upload, so that the moment the package is public the source is too.

### 6. Publish

```bash
uv publish
```

**This is the irreversible step.**

### 7. Tag what shipped

```bash
git tag -a v0.0.56 -m "Release 0.0.56"
git push origin v0.0.56
```

Tag *after* publishing, not before. If step 6 fails you will fix something and
rebuild; a tag created beforehand would point at a commit that was never
released and would have to be deleted and re-pushed. Tagging last means the
tag always describes reality.

Optionally attach release notes:

```bash
gh release create v0.0.56 --notes-file <(sed -n '/## v0.0.56/,/## v0.0.5 /p' CHANGELOG.md)
```

---

## Why tagging matters

Every release should be tagged, and historically this repo has not been. The
cost shows up later: without tags there is no way to answer

```bash
git diff v0.0.55..v0.0.56 --stat   # what changed between releases
git log v0.0.55..HEAD              # what is unreleased right now
git checkout v0.0.55               # the exact source a user installed
```

That last one is what you need when someone reports a bug against a version
whose code has since been deleted. Versions 0.0.6–0.0.55 were published
untagged, and their contents can no longer be reconstructed from the
repository — which is why the `v0.0.56` changelog entry carries a note
covering everything back to v0.0.5.

---

## Gotchas

**Stale `dist/`.** `uv publish` uploads everything matching `dist/*`, not just
what you last built. A leftover wheel from an old version will be uploaded
alongside the new one. Always `rm -rf dist/` before `uv build`.

**Versions are immutable.** Uploading `0.0.56` means `0.0.56` is used forever.
Deleting the release on PyPI does not free the number. There is no way to
replace a bad upload — only to publish `0.0.57`. This is why steps 3–4 exist.

**`requires-python = "==3.11.7"` is an exact pin.** Anyone on 3.11.6 or 3.11.9
gets `ERROR: Package requires a different Python` and cannot install at all.
This is the single largest barrier to adoption. Relaxing it to `>=3.11,<3.12`
is a real behaviour change for installers, so decide deliberately — but decide.

**Example tooling must stay out of the wheel.** torch, torchvision and the
Jupyter stack live in the PEP 735 `[dependency-groups] examples` table, *not*
in `[project.optional-dependencies]`. Extras are advertised in published wheel
metadata; dependency groups are not. If you find yourself adding a notebook or
example dependency, it goes in the group.

**Check the wheel contents if anything looks off:**

```bash
unzip -l dist/*.whl                       # what files shipped
unzip -p dist/*.whl '*/METADATA' | head -40   # declared deps and requires-python
```

---

## Better: publish from CI

The above requires a long-lived token on someone's laptop. PyPI trusted
publishing removes it entirely: a GitHub Actions workflow with
`permissions: id-token: write` authenticates over OIDC, and `uv publish` picks
it up with no token configured anywhere.

Triggering that workflow on tag push also inverts the ordering problem — the
tag becomes the trigger rather than an afterthought, so it can never disagree
with what was released. Worth setting up; the repo has no CI at present.
