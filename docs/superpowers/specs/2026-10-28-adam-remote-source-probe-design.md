# Adam 50h remote ZIP source verification — design (2026-10-28)

## Why
The 17.5-GB Adam ZIP has a broken Hugging Face viewer and no independently decoded sample in VietDub. Inspect bounded genuine source material without redistributing it.

## Approaches
- Full download: reliable, but expensive network/storage.
- Dataset viewer: unusable because audio post-processing errors.
- **Selected: ZIP central directory + HTTP Range**: seekable read-only adapter yields a few WAVs. Fail closed if CDN does not support range.

## Interface and safety
- `HTTPRangeReader(url, block_size=65536, max_transfer_bytes=16MiB)`: exact 206/Content-Range/size and bounded cache, only HF/loopback origin.
- `probe_remote_zip(url=REMOTE_ARCHIVE, limit=3)`: validates metadata and up to three WAVs; returns counts/durations only.
- Workflow artifact contains only `adam-source-qa.json`, never raw audio/text.
- No claims about training rights, speech quality, upstream speaker identity, or pronunciation.

## Acceptance
1. Local HTTP Range tests for seek/cache, server ignoring Range, ZIP member decoding, and transfer caps.
2. GitHub Actions real remote QA if CDN permits exact ranges; otherwise named failure.
3. No audio uploads, no model training/approval.
