.PHONY: help check-clean bump build tag publish release clean

VERSION ?=

help:
	@echo "make bump    VERSION=x.y.z   # set pyproject.toml version and commit it"
	@echo "make build                   # rm -rf dist, then uv build"
	@echo "make tag     VERSION=x.y.z   # push main, then tag and push vx.y.z"
	@echo "make publish                 # build, then uv publish dist/*"
	@echo "make release VERSION=x.y.z   # bump, build, tag, publish, in that order"
	@echo "make clean                   # rm -rf dist"

# Refuses to touch a dirty tree: bump/tag/release all commit or push, and
# doing that on top of unrelated uncommitted changes is easy to regret.
check-clean:
	@git diff --quiet && git diff --cached --quiet || \
		(echo "Working tree not clean. Commit or stash changes first." && exit 1)

bump: check-clean
ifndef VERSION
	$(error VERSION is required, e.g. make bump VERSION=0.0.59)
endif
	python3 -c "import pathlib, re; p = pathlib.Path('pyproject.toml'); p.write_text(re.sub(r'^version = \".*\"', 'version = \"$(VERSION)\"', p.read_text(), count=1, flags=re.M))"
	git add pyproject.toml
	git commit -m "Bump version to $(VERSION)"

clean:
	rm -rf dist

build: clean
	uv build

# Tags only after main is pushed, so the tag never points at a commit
# the remote doesn't have yet.
tag:
ifndef VERSION
	$(error VERSION is required, e.g. make tag VERSION=0.0.59)
endif
	git push origin main
	git tag v$(VERSION)
	git push origin v$(VERSION)

publish: build
	uv publish

release: bump build tag publish
	@echo "Released v$(VERSION)"
