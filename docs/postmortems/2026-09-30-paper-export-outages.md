# Post-mortem: Paper's export service went down six times while we built the demo

**Dates:** 2026-09-29 evening and 2026-09-30 morning (Pacific).
**Org:** zro54. **Systems:** Paper's export cassette (`/v1/cassettes/export/sessions/<id>`, behind `paperctl sessions export`), the autolabel cassette run locally at revision `1a43a65`, and this repo's `export` command.
**Impact:** the export cassette stopped answering for 5 to 15 minutes each time, six times. Anyone in the org exporting sessions during those windows got "could not reach the tapes API". A teammate reported the container going down and a Grafana alert. No data was lost; some exports arrived truncated and were discarded.

## Timeline (approximate, Pacific)

| When | What we ran | What happened |
|---|---|---|
| 09-29 20:07 | `label apology` dry run over the 25 newest sessions, through the cassette | cassette read 4 sessions, then export failed; service down |
| 09-29 20:10 | the same run again | same failure at the same point |
| 09-29 20:13 | 25 sequential `paperctl sessions export` calls by hand, to diagnose | first 14 failed (service still down), last 11 succeeded |
| 09-30 ~05:30 | `label apology --sessions 1000` dry run | 181 exports in, service down; run failed |
| 09-30 ~05:45 | the same run plus a per-model breakdown script at the same time (two cassette runs, eight parallel exports) | service down again; the `--apply` run failed before writing labels |
| 09-30 06:24 | this repo's `export`, one session at a time, 1 s pause, after the service recovered | 65 fetches, then a 4.5 MB response cut mid-string; the client crashed on the parse |
| 09-30 06:35 | `export` again with truncation handled and one session skipped | 83 fetches, then "could not reach"; the loop fell back to the cache and finished with 400 sessions |
| 09-30 07:0x | `export` through core's traces endpoint via paperd's proxy, while the export cassette was still timing out | 490 sessions, 0 failures; the export cassette was answering again afterwards |

## What we found

1. **The export cassette dies under a sustained stream of ordinary exports.** It fell over after 65, 78, and 83 consecutive one-per-second fetches of sessions of every size, including empty ones. It also died under the autolabel cassette's four-parallel fetches. The sessions in flight at each crash were not the same and were mostly small. Two sessions came back truncated on every attempt (`01a0b601`, 9 turns, ~200k output tokens; `019ef13f`, 11 turns), which looked like poison sessions at the time but is consistent with the service cutting whatever was in flight as it went down.

2. **Core kept answering while the wrapper was down.** `GET /v1/sessions/<id>/traces` on core, reached through the proxy paperd runs on localhost, returned the same record (same keys, schema, traces, spans, and parse) in about a second during an outage, and served 490 sessions in a row without incident. The export cassette is a wrapper over this endpoint.

3. **The pinned autolabel cassette multiplied the load.** It treats any session it cannot parse as missing and re-exports it on every run. Empty zero-turn sessions never parse, and about 60% of the org's 1000 newest sessions are empty, so each run re-exported hundreds of them. Its cache held 722 files for 404 distinct sessions; one session had been exported 18 times. Over the two days the cassette issued 914 export requests from this machine.

4. **Truncated responses were kept.** When the service died mid-response, `paperctl` wrote the partial file and the cassette kept it, then failed on the next run with a utf-8 decode error at the truncation point. Our own client crashed on the JSON parse of a truncated record instead of treating it as a failed fetch.

5. **We kept retrying into an outage.** Retrying a run right after a failure, and running two runs at once, prolonged the outages. On the first evening the diagnostic loop of 25 exports started while the service was already down.

## How this is prevented from now on

Each outage had a trigger; each trigger now has a guard. The first three are code and cannot be skipped by accident. The last two are rules for whoever runs the pipeline.

| Trigger | Guard | Where | Owner |
|---|---|---|---|
| Streaming session exports through the export cassette | Records come from core's traces endpoint via paperd's proxy; the wrapper is only a fallback when the daemon has no proxy | `Paper.export_session` | this repo |
| Hundreds of re-exports of empty sessions per run | Zero-turn sessions are never sent to the cassette | `label_candidates` | this repo |
| Continuing to request into a down service | After one outage-shaped failure the export loop makes no more requests and reads the rest from `data/cache/`; a truncated record is a failure, never cached | `run_export`, `Paper.export_session` | this repo |
| Bursts | One request per second between real fetches (`TAPES_EJECT_EXPORT_PAUSE`); one run at a time, never a label run and an export in parallel | config; RUNBOOK | operator |
| Retrying a failed run immediately | Probe the service with one tiny session first; if it does not answer, wait, do not retry the run | RUNBOOK | operator |

What is not covered here, and would prevent it for every client rather than this one: the export cassette releasing whatever it holds per request (Paper), the autolabel cassette treating unparseable sessions as known-empty and validating downloads (cassette repo, or moot once autolabel ships as a feature, #5), and `paperctl sessions export` failing on a partial write instead of exiting 0.

## What we changed (this repo)

- Session records are fetched from core's traces endpoint through paperd's proxy, found once from `paperctl status`; `paperctl sessions export` is the fallback when the daemon reports no proxy. (#6)
- Empty zero-turn sessions are never sent to the cassette. (#4)
- Sessions over `TAPES_EJECT_MAX_TURNS` or `TAPES_EJECT_MAX_OUTPUT_TOKENS` are skipped before any request; `TAPES_EJECT_SKIP_SESSIONS` lists individual sessions never to fetch. (#1, #6)
- `export` pauses `TAPES_EJECT_EXPORT_PAUSE` seconds (default 1) after each real fetch. On an outage-shaped failure (`could not reach`, `timed out`, `truncated`) it stops requesting and reads the rest from `data/cache/`, so an outage costs one run's fresh fetches, not the whole export. (#4, #6)
- A truncated record raises a `PaperError` and is never cached. (#6)
- Runs are one at a time, and a tiny-session probe is used to check the service before a run rather than retrying a failed run. (RUNBOOK, this document)

## What belongs elsewhere

- **Paper's export cassette:** it should survive a stream of small exports. Whatever it holds per request is not released between them. The traces endpoint on core does not have this problem, which points at the wrapper rather than the projection.
- **The autolabel cassette:** treat "cannot parse" as "known empty", not "missing", so it stops re-exporting the same sessions; validate a downloaded record before keeping it; lower the default export parallelism, or read from core's traces endpoint the way its HEAD revision already does when `CASSETTE_CORE_URL` is set. Once autolabel ships as a Paper feature, none of this runs on a laptop (#5).
- **paperctl:** `sessions export` exited 0 after writing a truncated response at least once. A partial write should be an error.

## What we would do differently

- Probe a new data path with a handful of requests and watch the service before running it over hundreds of sessions.
- Read the newest revision of a dependency before pinning an old one. The fix was already in the cassette's HEAD, in a comment.
- Stop at the first outage. Every retry into a down service extended the outage and produced corrupt files.
