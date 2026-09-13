# Release identity and procedure

A released version must name one reviewed object. This document separates the four identities a
release can have and gates each step that creates one. Building and verifying artifacts is
automated. Tagging and publishing always need explicit release authorization.

## Identities

| Identity | What it is | Who creates it |
| --- | --- | --- |
| Version | `slimx_agent.__version__` is the only maintained source. `pyproject.toml` derives the distribution version from it, and `scripts/check_version.py` keeps the changelog, installed metadata, and `/health` in agreement | Any reviewed change |
| Source commit | The exact reviewed Git commit and its tree | A reviewed PR head, or its normal merge commit |
| Tag | `v<version>` on the released commit, never moved or reused | Release authority only |
| Artifact | An sdist, wheel, or container image, identified by its digest | Release authority only |

- **A version is a candidate until tagged.** Until a tag exists, a version string names source,
  not a release.
- **Never reuse a released identity.** Never publish different bytes under a version that
  anyone has already consumed.
- **Friendly names are not provenance.** An image tag or branch name proves nothing about
  source identity; only a commit SHA or artifact digest does.

## Version identity history (recorded 2026-09-13)

- **No tags or artifacts.** No tag, GitHub release, or published package or image exists for any
  version.
- **0.18.0 and 0.19.0 are source commits only.** 0.18.0 is `6791590`; 0.19.0 is `fc6f4c5`.
  ControlRoom consumed 0.19.0 by exact source archive of `fc6f4c5`, and built its optional
  service from the same commit.
- **Earlier drift.** Before 0.20.0, `main` (`bf227a6`) declared 0.17.0 in its manifest while its
  runtime `__version__` read 0.16.0. From 0.20.0 the manifest version is derived from
  `__version__`.
- **0.20.0 is an unreleased source version.** It contains the 0.18.0/0.19.0 commits unchanged.
  Consumers identify it by exact commit until a tag exists.

## Procedure

1. **Prepare.** On a branch, set `__version__` and add the matching `CHANGELOG.md` heading.
   Describe every behavior change and its migration note.
2. **Verify at the exact head.** CI must pass on the supported Python matrix: version agreement,
   lint, format, strict typing, tests with coverage floors, and
   `scripts/verify_distribution.py`. That last script builds the sdist and wheel and proves that
   core-only and service installs work outside the source tree.
3. **Review.** Get an independent review of the full delta against both `main` and the
   version currently consumed downstream. Resolve security and contract findings before asking
   for merge.
4. **Merge normally** once merge is authorized. Record these alongside the release notes:
   - the PR head SHA;
   - the merge commit SHA;
   - the merge tree.

   A normal merge has a different SHA from the PR head. If `main` had not moved, the merge tree
   equals the reviewed tree.
5. **Pin downstream.** Consumers pin an exact commit whose tree was reviewed. Use the merge
   commit when its tree matches the reviewed tree, otherwise the reviewed head. Update the
   manifest, lock, service build context, compatibility record, and boundary tests together.
6. **Tag.** This needs separate release authorization. Create an annotated tag `v<version>` on
   the merged commit. Never move or delete a pushed tag; a mistake gets a new patch version.
7. **Publish artifacts (optional).** This needs separate release authorization. Build from the
   tag, run `scripts/verify_distribution.py`, and record each artifact's SHA-256. Container
   images are identified by digest, not by name.

Exact-commit source archives are an acceptable, reproducible way to consume this package, and
publishing to a package index is not a prerequisite for correctness fixes. Consumers should rely
on their own lockfile hashes for archive bytes.

## Rollback

Consumers roll back by re-pinning the previous exact commit (0.19.0 is `fc6f4c5`) in the same
coordinated set of files. No tag or published artifact is moved, deleted, or overwritten.
