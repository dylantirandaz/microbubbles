from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import modal


app = modal.App("microbubbles-mach-reference")
volume = modal.Volume.from_name("microbubbles-mach-reference", create_if_missing=True)
SAMPLE_URL = (
    "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/"
    "sanitized_neutral_ultratrace.h5"
)
USER_AGENT = "ultratrace-ulm/0.1"
DOWNLOAD_CHUNK = 8 * 1024 * 1024
COMMIT_EVERY_BYTES = 1024 * 1024 * 1024

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "cupy-cuda12x",
        "git+https://github.com/alephneuro/microbubbles.git",
        "h5py",
        "mach-beamform",
        "numpy",
        "scipy",
        "tqdm",
    )
)

download_image = modal.Image.debian_slim(python_version="3.12")


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


@app.function(
    image=download_image,
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def ensure_sample() -> str:
    output = Path("/data/sample_ultratrace.h5")
    output.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(SAMPLE_URL)
    existing = output.stat().st_size if output.exists() else 0
    if total is not None and existing == total:
        print(f"Already complete: {output} ({_fmt_bytes(total)})")
        return str(output)
    if total is not None and existing > total:
        output.unlink()
        existing = 0

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {_fmt_bytes(existing)} ...")

    req = urllib.request.Request(SAMPLE_URL, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            print(f"Already complete: {output} ({_fmt_bytes(existing)})")
            return str(output)
        raise

    with resp:
        mode = "ab" if existing and resp.status == 206 else "wb"
        done = existing if mode == "ab" else 0
        next_commit = done + COMMIT_EVERY_BYTES
        grand_total = total
        content_length = resp.headers.get("Content-Length")
        if grand_total is None and content_length is not None:
            grand_total = done + int(content_length)

        with output.open(mode) as fh:
            while True:
                block = resp.read(DOWNLOAD_CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if grand_total:
                    pct = 100.0 * done / grand_total
                    bar = f"{_fmt_bytes(done)} / {_fmt_bytes(grand_total)} ({pct:.1f}%)"
                else:
                    bar = _fmt_bytes(done)
                print(f"\r  {bar}        ", end="", file=sys.stderr, flush=True)

                if done >= next_commit:
                    fh.flush()
                    os.fsync(fh.fileno())
                    volume.commit()
                    print(f"\nCommitted partial download at {_fmt_bytes(done)}")
                    next_commit = done + COMMIT_EVERY_BYTES

            fh.flush()
            os.fsync(fh.fileno())

    volume.commit()
    print("", file=sys.stderr)
    print(f"Saved {output} ({_fmt_bytes(output.stat().st_size)})")
    return str(output)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def make_reference(
    acq_start: int = 0,
    num_acqs: int = 1,
    elev_planes: int = 1,
    z_coarseness: float = 2.0,
    x_coarseness: float = 2.0,
    large_fov: bool = False,
) -> str:
    input_path = Path("/data/sample_ultratrace.h5")
    output_path = Path(f"/data/mach_acq{acq_start}_n{num_acqs}.h5")
    if not input_path.exists():
        raise FileNotFoundError("sample_ultratrace.h5 is missing; run ensure_sample first")

    cmd = [
        "python",
        "-m",
        "ultratrace_ulm.cli",
        "beamform",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--acq-start",
        str(acq_start),
        "--num-acqs",
        str(num_acqs),
        "--elev-planes",
        str(elev_planes),
        "--z-coarseness",
        str(z_coarseness),
        "--x-coarseness",
        str(x_coarseness),
    ]
    if not large_fov:
        cmd.append("--no-large-fov")

    subprocess.run(cmd, check=True)
    volume.commit()
    return str(output_path)


@app.local_entrypoint()
def main(
    acq_start: int = 0,
    num_acqs: int = 1,
    elev_planes: int = 1,
    z_coarseness: float = 2.0,
    x_coarseness: float = 2.0,
    large_fov: bool = False,
) -> None:
    sample_path = ensure_sample.remote()
    print(f"Sample ready at {sample_path}")
    output_path = make_reference.remote(
        acq_start=acq_start,
        num_acqs=num_acqs,
        elev_planes=elev_planes,
        z_coarseness=z_coarseness,
        x_coarseness=x_coarseness,
        large_fov=large_fov,
    )
    print(f"Modal reference written to {output_path}")
    print("Download with:")
    print(
        "  modal volume get microbubbles-mach-reference "
        f"/{Path(output_path).name} {Path(output_path).name}"
    )
