"""HTTP Range ZIP probe tests use an in-process local server, no real voice/audio."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
from threading import Thread
import zipfile

import numpy as np
import pytest
import soundfile as sf

from vietdub_native.adam_remote_probe import HTTPRangeReader, RemoteProbeError, probe_remote_zip


def make_zip() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_STORED) as z:
        rows=[]
        for i in range(1, 7):
            rows.append({'id':i, 'text':f'Đây là câu tiếng Việt thứ {i}'})
            t = np.arange(int(0.6 * 48000))/48000
            signal=(0.12 * np.sin(2*np.pi*(145+i*23)*t)).astype('float32')
            data=io.BytesIO()
            sf.write(data,signal,48000,format='WAV',subtype='PCM_16')
            z.writestr(f'example/wav/{i}.wav',data.getvalue())
        z.writestr('example/data.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    return bio.getvalue()


@contextmanager
def zip_server(blob: bytes, *, supports_range: bool=True):
    class Handler(BaseHTTPRequestHandler):
        downloaded = 0
        def log_message(self, *args): pass
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length',str(len(blob)))
            self.end_headers()
        def do_GET(self):
            header=self.headers.get('Range','')
            if supports_range and header.startswith('bytes='):
                start,end=map(int,header[6:].split('-'))
                if start >= len(blob):
                    self.send_response(416); self.end_headers();return
                end=min(end,len(blob)-1)
                chunk=blob[start:end+1]
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(blob)}')
            else:
                chunk=blob
                self.send_response(200)
            self.send_header('Content-Length',str(len(chunk)))
            self.end_headers()
            Handler.downloaded += len(chunk)
            try: self.wfile.write(chunk)
            except BrokenPipeError: pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=Thread(target=server.serve_forever,daemon=True);worker.start()
    try:yield f'http://127.0.0.1:{server.server_port}/finnal_tts.zip',Handler
    finally:server.shutdown();server.server_close();worker.join(timeout=2)


def test_remote_reader_range_and_seek_with_caching():
    payload=b'abcdef'*20000
    with zip_server(payload) as (url, handler):
        with HTTPRangeReader(url, block_size=1024, max_transfer_bytes=16384) as remote:
            assert remote.read(4)==b'abcd'
            assert remote.seek(-3,2)==len(payload)-3
            assert remote.read(3)==payload[-3:]
            remote.seek(1)
            assert remote.read(3)==b'bcd'
            assert remote.seekable() and remote.readable()
        assert handler.downloaded <= 3072


def test_rejects_server_ignoring_range_before_downloading_archive():
    with zip_server(make_zip(),supports_range=False) as (url,handler):
        with pytest.raises(RemoteProbeError,match='Range|partial'):
            HTTPRangeReader(url)
        assert handler.downloaded <= len(make_zip())  # fail closed; no fake partial result


def test_remote_probe_real_wav_schema_without_full_zip_download():
    blob=make_zip()
    with zip_server(blob) as (url, handler):
        report=probe_remote_zip(url,limit=2, max_transfer_bytes=3*1024*1024)
        assert report['source_audio_verified'] == 2
        assert report['metadata_rows'] == 6
        assert report['training_approved'] is False
        assert report['source_format'] == '48000Hz PCM16 mono WAV'
        assert report['sampled_seconds'] > 1.1
        assert report['downloaded_bytes'] < len(blob)
        assert 'text' not in json.dumps(report).lower()
        assert handler.downloaded == report['downloaded_bytes']


def test_remote_probe_limits_and_mismatched_header():
    with zip_server(make_zip()) as (url,_):
        with pytest.raises(RemoteProbeError,match='limit'):
            probe_remote_zip(url,limit=0)
        with pytest.raises(RemoteProbeError,match='transfer'):
            probe_remote_zip(url,limit=2,max_transfer_bytes=10)


def test_real_probe_counts_invalid_text_without_exposing_it():
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, 'w', compression=zipfile.ZIP_STORED) as z:
        rows=[]
        for i in range(1, 7):
            rows.append({'id': i, 'text': ' ' if i == 2 else f'Xin chào {i}'})
            t=np.arange(int(.6*48000))/48000
            signal=(.12*np.sin(2*np.pi*(180+i*11)*t)).astype('float32')
            b=io.BytesIO();sf.write(b,signal,48000,format='WAV',subtype='PCM_16')
            z.writestr(f'dataset/wav/{i}.wav',b.getvalue())
        z.writestr('dataset/data.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with zip_server(blob.getvalue()) as (url,_):
        report=probe_remote_zip(url,limit=2,max_transfer_bytes=4*1024*1024)
    assert report['metadata_rows'] == 5
    assert report['metadata_invalid_transcripts'] == 1
    assert report['source_audio_verified'] == 2
    assert 'Xin chào' not in str(report)
