"""Bounded, read-only HTTP Range probe of the *actual* Adam ZIP.

No archive, speech recordings, transcripts, or private model artifacts are saved.
Does not grant or imply permission to train on upstream synthetic voice assets.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import io
import json
import statistics
from urllib import request
import zipfile

import numpy as np
import soundfile as sf

from .adam_tts_vocoder import (
    SOURCE_URL, _validate_zip_archive, _read_metadata, MAX_COMPRESSED_WAV_BYTES,
)

REMOTE_ARCHIVE = SOURCE_URL + '/resolve/main/finnal_tts.zip?download=true'
MAX_ARCHIVE_BYTES = 24 * 1024**3


class RemoteProbeError(ValueError):
    """Remote server or ZIP metadata failed a strict bounded inspection."""


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reader; rejects servers that cannot serve exact byte ranges.

    Memory: <= cache_blocks * block_size. Total GET payload is capped explicitly.
    A server returning full-body 200 instead of 206 is rejected *before* reading it.
    """

    def __init__(self, url: str, *, block_size: int = 65536,
                 max_transfer_bytes: int = 16 * 1024 * 1024,
                 cache_blocks: int = 12, timeout: int = 30):
        super().__init__()
        if not url.startswith(('http://127.0.0.1:', 'https://huggingface.co/')):
            raise RemoteProbeError('Only canonical Hugging Face or local test server allowed')
        if not 512 <= block_size <= 1024 * 1024 or cache_blocks < 1:
            raise RemoteProbeError('Invalid HTTP Range cache settings')
        if max_transfer_bytes < block_size + 1:
            raise RemoteProbeError('Insufficient transfer limit for Range probe')
        self.url, self.timeout, self.block_size = url, timeout, block_size
        self.max_transfer_bytes, self.cache_blocks = max_transfer_bytes, cache_blocks
        self.position = 0
        self.downloaded_bytes = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        try:
            with request.urlopen(request.Request(url, method='HEAD',
                           headers={'User-Agent': 'VietDub-Adam-Metadata-Probe/1.0'}), timeout=timeout) as response:
                self.size = int(response.headers.get('Content-Length', '0'))
        except (OSError, ValueError) as exc:
            raise RemoteProbeError('HEAD failed or returned invalid Content-Length') from exc
        if not 1 <= self.size <= MAX_ARCHIVE_BYTES:
            raise RemoteProbeError('Remote archive size missing or outside permitted bounds')
        self._range(0, 0)  # Ensure reliable 206 before attempting ZIP parsing.

    def _range(self, start: int, end: int) -> bytes:
        if not (0 <= start <= end < self.size):
            raise RemoteProbeError('Range outside archive')
        length = end - start + 1
        if self.downloaded_bytes + length > self.max_transfer_bytes:
            raise RemoteProbeError('Remote transfer limit exceeded; refusing full archive download')
        req = request.Request(self.url, headers={
            'User-Agent': 'VietDub-Adam-Metadata-Probe/1.0',
            'Range': f'bytes={start}-{end}',
            'Accept-Encoding': 'identity',
        })
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 206:
                    raise RemoteProbeError('Server does not support exact HTTP Range / partial content (206)')
                content_range = response.headers.get('Content-Range', '')
                expected = f'bytes {start}-{end}/{self.size}'
                if content_range != expected:
                    raise RemoteProbeError(f'Mismatched Content-Range; expected {expected}')
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) != length:
                    raise RemoteProbeError('Unexpected partial response length')
                payload = response.read(length + 1)
        except RemoteProbeError:
            raise
        except (OSError, ValueError) as exc:
            raise RemoteProbeError('Range request failed; the remote object may have changed') from exc
        if len(payload) != length:
            raise RemoteProbeError('Truncated or oversized ranged response')
        self.downloaded_bytes += len(payload)
        return payload

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        start = index * self.block_size
        data = self._range(start, min(start + self.block_size, self.size) - 1)
        self._cache[index] = data
        if len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError('Unknown seek mode')
        if not 0 <= target <= self.size:
            raise RemoteProbeError('Seek outside archive')
        self.position = target
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError('I/O operation on closed remote reader')
        if size is None or size < 0:
            size = self.size - self.position
        remaining = min(size, self.size - self.position)
        chunks: list[bytes] = []
        while remaining:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            chunk = block[offset:offset + remaining]
            if not chunk:
                raise RemoteProbeError('Remote block read made no progress')
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)


def probe_remote_zip(url: str = REMOTE_ARCHIVE, *, limit: int = 3,
                     max_transfer_bytes: int = 16 * 1024 * 1024,
                     block_size: int = 65536) -> dict:
    """Inspect up to 3 actual WAV members; expose no audio/text or IDs in report."""
    if not 1 <= limit <= 3:
        raise RemoteProbeError('limit must be between 1 and 3')
    if max_transfer_bytes < block_size + 1:
        raise RemoteProbeError('Insufficient transfer limit')
    with HTTPRangeReader(url, max_transfer_bytes=max_transfer_bytes,
                         block_size=block_size) as remote:
        try:
            with zipfile.ZipFile(remote) as zf:
                prefix, entries = _validate_zip_archive(zf)
                manifest_info = entries[prefix + 'data.jsonl']
                rows = _read_metadata(zf, manifest_info)
                # Representative, reproducible sparse selection: beginning/middle/end
                # of metadata, rather than only the easiest first few files.
                indexes = sorted({round(i * (len(rows)-1)/max(limit-1,1)) for i in range(limit)})
                if limit == 1:
                    indexes = [len(rows)//2]
                durations, errors = [], []
                for index in indexes:
                    row = rows[index]
                    member = entries.get(f"{prefix}wav/{row['id']}.wav")
                    if not member or not 1000 <= member.file_size <= MAX_COMPRESSED_WAV_BYTES:
                        errors.append('missing_or_oversized_source_audio')
                        continue
                    try:
                        raw = zf.read(member)
                        info = sf.info(io.BytesIO(raw))
                        if (info.samplerate != 48000 or info.channels != 1
                                or info.subtype != 'PCM_16' or not 0.4 <= info.duration <= 20):
                            errors.append('source_format_mismatch')
                            continue
                        wave, rate = sf.read(io.BytesIO(raw), dtype='float32')
                        rms = float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
                        clipped = float(np.mean(np.abs(wave) >= 0.999))
                        if not np.isfinite(wave).all() or rms < .002 or clipped > .005:
                            errors.append('invalid_amplitude')
                            continue
                        durations.append(float(info.duration))
                    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
                        errors.append('audio_decode_error')
                return {
                    'source_dataset': SOURCE_URL,
                    'source_format': '48000Hz PCM16 mono WAV',
                    'archive_bytes': remote.size,
                    'zip_members': len(entries),
                    'metadata_rows': len(rows),
                    'source_audio_verified': len(durations),
                    'sampled_seconds': round(sum(durations), 3),
                    'sample_median_seconds': round(statistics.median(durations), 3) if durations else None,
                    'rejection_reasons': errors,
                    'downloaded_bytes': remote.downloaded_bytes,
                    'transfer_fraction': round(remote.downloaded_bytes / remote.size, 7),
                    'training_approved': False,
                    'intelligibility_verified': False,
                    'note': 'Remote source-only technical QA; no raw audio/transcripts or training rights verified.',
                }
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise RemoteProbeError(f'ZIP metadata/audio probe failed: {exc}') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded real Adam ZIP inspection without 17.5GB download')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args(argv)
    result = probe_remote_zip(limit=args.limit)
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
