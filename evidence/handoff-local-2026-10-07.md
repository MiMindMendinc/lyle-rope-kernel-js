# Local package handoff evidence — 2026-10-07

**Label:** TESTED LOCALLY (Node v22.23.3, Linux x86_64).  
**Not:** CI VERIFIED (await PR checks) · not verified `main` · not npm published.

Source tip at branch start: `f9b70514446f1821562ad5e552e79df3d394a617`

Raw command transcripts and the packed tarball for this handoff live on the agent box under `/workspace/showcase-handoff/lyle/` (not committed; large/local). In-repo references:

- [docs/PRERELEASE_NOTES.md](../docs/PRERELEASE_NOTES.md)
- Prior hardening snapshot: [hardening-2026-10-05.json](hardening-2026-10-05.json)

Commands run with exit 0 locally: `npm test`, `npm run example`, `npm run test:package`, `npm run evidence`, `npm run verify`, `npm pack`, plus a fresh offline consumer install exercising `applyRoPE` / `applyRoPEQK`.
