"""Benchmark controller: framed worker IPC, privacy-aware previews, batches and gates."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import select
import struct
import subprocess
import threading
import time
import zipfile

from core import wipe

PRIVATE_IMAGES = True
ROOT = Path(os.environ.get("BEAUTY_ROOT", "/content/beauty_phase0")).resolve()
CODE_ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / "venv/bin/python")
ENV = dict(os.environ)
CLIENT = None
MAX_IPC_BYTES = 80 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def configure(root, code_root, python, env):
    global ROOT, CODE_ROOT, PYTHON, ENV
    ROOT = Path(root).resolve()
    CODE_ROOT = Path(code_root).resolve()
    PYTHON = str(python)
    ENV = dict(env)


def clear_buffer(value):
    if value is not None:
        wipe(value)


def recv_exact(stream, length, deadline):
    out = bytearray()
    while len(out) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker_ipc_timeout")
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            raise TimeoutError("worker_ipc_timeout")
        chunk = os.read(stream.fileno(), length - len(out))
        if not chunk:
            raise EOFError("worker_ipc_closed")
        out.extend(chunk)
    return bytes(out)


def _send(stream, value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_IPC_BYTES:
        raise ValueError("ipc_payload_too_large")
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _recv(stream, timeout=1800):
    deadline = time.monotonic() + timeout
    header = recv_exact(stream, 8, deadline)
    length = struct.unpack("!Q", header)[0]
    if length > MAX_IPC_BYTES:
        raise ValueError("ipc_payload_too_large")
    return pickle.loads(recv_exact(stream, length, deadline))


class ModelClient:
    def __init__(self, service, timeout=1800):
        self.service = service
        self.timeout = timeout
        self.lock = threading.Lock()
        worker = CODE_ROOT / "worker.py"
        if not worker.is_file():
            raise RuntimeError("worker.py is missing; regenerate derived sources first")
        child_env = dict(ENV)
        child_env.setdefault("BEAUTY_ROOT", str(ROOT))
        child_env.setdefault("BEAUTY_WORKER_NO_IMAGE_WRITE", "1" if PRIVATE_IMAGES else "0")
        self.process = subprocess.Popen(
            [PYTHON, str(worker), service],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=child_env,
        )
        try:
            ready = _recv(self.process.stdout, timeout)
        except Exception as exc:
            diagnostic = self._stderr_text()
            self.close(kill=True)
            raise RuntimeError(diagnostic or str(exc)) from exc
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            diagnostic = self._stderr_text()
            self.close(kill=True)
            raise RuntimeError(diagnostic or str(ready))

    def _stderr_text(self):
        if self.process.stderr is None:
            return ""
        try:
            if self.process.poll() is None:
                return ""
            data = self.process.stderr.read() or b""
            return data.decode("utf-8", "replace")[-8000:]
        except Exception:
            return ""

    def call(self, request):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(self._stderr_text() or "worker_not_running")
            try:
                _send(self.process.stdin, request)
                return _recv(self.process.stdout, self.timeout)
            except Exception as exc:
                diagnostic = self._stderr_text()
                if diagnostic:
                    raise RuntimeError(diagnostic) from exc
                raise

    def close(self, kill=False):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        if not kill:
            try:
                _send(process.stdin, {"op": "stop"})
                _recv(process.stdout, 5)
            except Exception:
                kill = True
        if kill and process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()


def start_engine(service):
    global CLIENT
    if service not in {"hairstyle", "color"}:
        raise ValueError("unknown_service")
    if CLIENT is not None and CLIENT.service == service and CLIENT.process.poll() is None:
        return CLIENT
    stop_engine()
    CLIENT = ModelClient(service)
    return CLIENT


def stop_engine():
    global CLIENT
    client, CLIENT = CLIENT, None
    if client is not None:
        client.close()


def show_comparison(png):
    if PRIVATE_IMAGES:
        raise PermissionError("private_image_display_blocked")
    from IPython.display import display, Image as DisplayImage
    display(DisplayImage(data=bytes(png)))


def evaluate(face, reference=None, *, show=False, repeats=1, preview_fn=None,
             retain_output=False, target_rgb=(116, 52, 40), strength=.85,
             reference_color=False):
    if CLIENT is None:
        raise RuntimeError("engine_not_started")
    response = None
    try:
        request = {
            "op": "evaluate",
            "face": face,
            "reference": reference,
            "repeats": int(repeats),
            "return_images": bool(show or preview_fn or retain_output),
            "debug_public": bool(show and not PRIVATE_IMAGES),
            "target_rgb": list(target_rgb),
            "strength": float(strength),
            "reference_color": bool(reference_color),
        }
        response = CLIENT.call(request)
        metrics = dict(response.get("metrics", {}))
        if metrics.get("status") != "ok":
            return metrics
        comparison = response.get("comparison_png")
        if preview_fn is not None and comparison is not None:
            preview_fn(comparison)
        elif show and comparison is not None:
            show_comparison(comparison)
        if retain_output:
            return response
        return metrics
    finally:
        clear_buffer(face)
        clear_buffer(reference)
        if response is not None and not retain_output:
            clear_buffer(response.get("output_png"))
            clear_buffer(response.get("comparison_png"))
            response.clear()


def single_demo(*, show=True, repeats=3):
    if CLIENT is None:
        raise RuntimeError("engine_not_started")
    face = bytearray((ROOT / "public_demo/6.png").read_bytes())
    reference = bytearray((ROOT / "public_demo/7.png").read_bytes())
    previous = PRIVATE_IMAGES
    globals()["PRIVATE_IMAGES"] = False
    try:
        return evaluate(face, reference, show=show, repeats=repeats)
    finally:
        globals()["PRIVATE_IMAGES"] = previous


def _safe_relpath(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid_manifest_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("invalid_manifest_path")
    return value


def validate_manifest(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("invalid_manifest")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid_manifest")
        ident = case.get("id")
        group = case.get("group")
        if not isinstance(ident, str) or ident in seen or group not in {"A", "B", "C"}:
            raise ValueError("invalid_manifest")
        seen.add(ident)
        _safe_relpath(case.get("target"))
        if case.get("reference") is not None:
            _safe_relpath(case.get("reference"))
    return cases


@contextlib.contextmanager
def zip_cases(blob):
    if not isinstance(blob, (bytes, bytearray)) or not blob:
        raise ValueError("invalid_zip")
    raw = io.BytesIO(bytes(blob))
    clear_buffer(blob)
    archive = zipfile.ZipFile(raw, "r")
    released = set()
    loaded = {}
    try:
        infos = archive.infolist()
        if len(infos) > 256:
            raise ValueError("zip_too_many_entries")
        total = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise ValueError("encrypted_zip_not_supported")
            if info.file_size > 25 * 1024 * 1024:
                raise ValueError("zip_entry_too_large")
            total += info.file_size
            if total > 512 * 1024 * 1024:
                raise ValueError("zip_uncompressed_too_large")
            if info.compress_size and info.file_size / max(info.compress_size, 1) > 200:
                raise ValueError("zip_compression_ratio_too_high")
        if "manifest.json" not in archive.namelist():
            raise ValueError("manifest_missing")
        cases = json.loads(archive.read("manifest.json").decode("utf-8"))
        validate_manifest(cases)

        def loader(name):
            _safe_relpath(name)
            if name in released:
                raise RuntimeError("zip_member_already_released")
            if name not in archive.namelist():
                raise FileNotFoundError(name)
            value = bytearray(archive.read(name))
            loaded[name] = value
            return value

        def release(name):
            value = loaded.pop(name, None)
            clear_buffer(value)
            released.add(name)

        yield cases, loader, release
    finally:
        for value in loaded.values():
            clear_buffer(value)
        archive.close()
        raw.close()


def record_scores(case, metrics):
    labels = (("identity", "Identity"), ("hairline", "Hairline"), ("reference", "Reference"))
    result = {}
    for key, label in labels:
        while True:
            value = input(f"{case.get('id')} — {label} (1-5): ").strip()
            if value in {"1", "2", "3", "4", "5"}:
                result[key] = int(value)
                break
    return result


def run_batch(cases, loader, *, preview_fn=None, score_fn=None, show=False,
              save_dir=None, allow_public_export=False, release_fn=None):
    validate_manifest(cases)
    if PRIVATE_IMAGES and save_dir:
        raise PermissionError("private_image_export_blocked")
    if save_dir and not allow_public_export:
        raise PermissionError("public_export_confirmation_required")
    destination = Path(save_dir) if save_dir else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        face = reference = None
        response = None
        try:
            face = loader(case["target"])
            if case.get("reference"):
                reference = loader(case["reference"])
            request = {
                "op": "evaluate", "face": face, "reference": reference,
                "repeats": 3, "return_images": bool(preview_fn or show or destination),
                "debug_public": bool(show and not PRIVATE_IMAGES),
            }
            response = CLIENT.call(request)
            metrics = dict(response.get("metrics", {}))
            row = {"id": case["id"], "group": case["group"],
                   "engine": getattr(CLIENT, "service", None), **metrics}
            comparison = response.get("comparison_png")
            output = response.get("output_png")
            if destination and comparison is not None:
                (destination / f"{case['id']}_comparison.png").write_bytes(bytes(comparison))
                if output is not None:
                    (destination / f"{case['id']}_result.png").write_bytes(bytes(output))
            if preview_fn is not None and comparison is not None:
                preview_fn(comparison)
            elif show and comparison is not None:
                show_comparison(comparison)
            if score_fn is not None and row.get("status") == "ok":
                scores = score_fn(case, row)
                if scores:
                    for key in ("identity", "hairline", "reference"):
                        value = scores.get(key)
                        if isinstance(value, bool) or value not in (1, 2, 3, 4, 5):
                            raise ValueError("invalid_human_score")
                        row[key] = int(value)
            rows.append(row)
        except Exception as exc:
            rows.append({"id": case.get("id"), "group": case.get("group"),
                         "engine": getattr(CLIENT, "service", None),
                         "status": "error", "error_code": type(exc).__name__})
        finally:
            clear_buffer(face)
            clear_buffer(reference)
            if response:
                clear_buffer(response.get("output_png"))
                clear_buffer(response.get("comparison_png"))
                response.clear()
            if release_fn:
                release_fn(case.get("target"))
                if case.get("reference"):
                    release_fn(case.get("reference"))
    return rows


def _expected_ids():
    return {f"{group}{i:02d}" for group in "ABC" for i in range(1, 11)}


def quality_gate(records, seconds=8.0):
    rows = list(records)
    ids = [row.get("id") for row in rows]
    if len(rows) != 30 or set(ids) != _expected_ids() or len(set(ids)) != 30:
        return {"gate": "INCOMPLETE", "denominator": 30, "quality_passes": 0, "joint_passes": 0}
    engines = {row.get("engine") for row in rows}
    if len(engines) != 1:
        return {"gate": "INCOMPLETE", "denominator": 30, "quality_passes": 0, "joint_passes": 0}
    assessed = True
    quality = 0
    joint = 0
    latencies = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        ratings = [row.get(k) for k in ("identity", "hairline", "reference")]
        latency = row.get("inference_s")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in ratings):
            assessed = False
            continue
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(float(latency)):
            assessed = False
            continue
        latencies.append(float(latency))
        q = all(float(v) >= 4 for v in ratings)
        quality += int(q)
        joint += int(q and float(latency) < float(seconds))
    gate = "UNASSESSED" if not assessed else ("PASS" if quality >= 21 and joint >= 21 else "FAIL")
    result = {"gate": gate, "denominator": 30, "quality_passes": quality, "joint_passes": joint}
    if latencies:
        ordered = sorted(latencies)
        result["p95_inference_s"] = ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)]
    return result


def export_metrics(records, path, seconds=8.0):
    path = Path(path)
    rows = list(records)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(
        json.dumps({"gate": quality_gate(rows, seconds=seconds), "records": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def batch_folder(folder, **kwargs):
    if PRIVATE_IMAGES:
        raise PermissionError("private_folder_mode_blocked")
    folder = Path(folder)
    cases = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(cases)
    def loader(name):
        return bytearray((folder / _safe_relpath(name)).read_bytes())
    return run_batch(cases, loader, **kwargs)


def batch_zip_demo(blob=None, *, preview_fn=None, score_fn=None, show=False,
                   save_dir=None, allow_public_export=False):
    if blob is None:
        raise ValueError("zip_bytes_required")
    with zip_cases(blob) as (cases, loader, release):
        return run_batch(cases, loader, preview_fn=preview_fn, score_fn=score_fn,
                         show=show, save_dir=save_dir,
                         allow_public_export=allow_public_export, release_fn=release)
