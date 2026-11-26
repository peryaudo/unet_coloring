#!/usr/bin/env python3
"""R2 helper for the ml-storage-layout convention.

One file, usable two ways:
  * CLI:     python r2.py <command> ...          (or python -m tools.r2 ... in a repo)
  * Module:  from tools.r2 import put_ckpt, write_status, pull_dataset

Layout (all keys live under s3://$R2_BUCKET/$PROJECT/):
  raw/<source>/<YYYY-MM-DD>/part-00001.jsonl.zst
  datasets/<name>/v001/ + manifest.json, datasets/<name>/LATEST
  runs/<run_id>/{config.yaml,status.json,metrics.jsonl,ckpt/,samples/,logs/}

Environment:
  R2_ENDPOINT_URL        https://<account_id>.r2.cloudflarestorage.com   (required)
  R2_ACCESS_KEY_ID       R2 API token access key                          (required)
  R2_SECRET_ACCESS_KEY   R2 API token secret                              (required)
  R2_BUCKET              bucket name (required; R2 names are 3-63 chars, e.g. "ml-data")
  PROJECT                project prefix (repo name); falls back to the git repo name
  RUN_ID                 current run, needed by run-scoped commands

Requires boto3 (pip install boto3). Credentials are never printed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

VERSION = "2026.11.26-1"

KEEP_CKPTS_DEFAULT = 3
STEP_RE = re.compile(r"step_(\d{8})\.pt$")
VERSION_RE = re.compile(r"^v(\d{3,})$")
RUN_STATES = ("starting", "running", "done", "failed", "timeout")


class R2Error(RuntimeError):
    pass


# --------------------------------------------------------------------------- env


def _env(name: str, default: Optional[str] = None, required: bool = True) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise R2Error(
            f"environment variable {name} is not set. Add it to the session's cloud "
            "environment, `fly secrets` (crawler) or RunPod Secrets (Pod); never paste "
            "keys into chat."
        )
    return val or ""


def bucket() -> str:
    return _env("R2_BUCKET")


def project() -> str:
    p = os.environ.get("PROJECT")
    if p:
        return p
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return Path(top).name.lower()
    except Exception:
        raise R2Error("PROJECT is not set and the current directory is not a git repo")


def run_id() -> str:
    return _env("RUN_ID")


_CLIENT = None


def client():
    """boto3 S3 client pointed at R2. Cached per process."""
    global _CLIENT
    if _CLIENT is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise R2Error("boto3 is not installed: pip install boto3")
        _CLIENT = boto3.client(
            "s3",
            endpoint_url=_env("R2_ENDPOINT_URL"),
            aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
        )
    return _CLIENT


def _public_client(region: str):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", region_name=region, config=Config(signature_version=UNSIGNED))


# ------------------------------------------------------------------------- keys


def key(*parts: str) -> str:
    """Key under the project prefix: key('datasets', 'x', 'v001') -> '<project>/datasets/x/v001'."""
    clean = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join([project(), *clean])


def run_key(*parts: str, rid: Optional[str] = None) -> str:
    return key("runs", rid or run_id(), *parts)


def make_run_id(slug: str, sha: Optional[str] = None, now: Optional[_dt.datetime] = None) -> str:
    """<YYYYMMDD>-<HHMM>-<sha7>-<slug> in UTC."""
    if sha is None:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise R2Error("not inside a git repository; pass --sha")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    if not slug:
        raise R2Error("run slug is empty")
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y%m%d-%H%M}-{sha[:7]}-{slug}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------- primitives


def put_file(local: str | os.PathLike, k: str) -> str:
    client().upload_file(str(local), bucket(), k)
    return k


def get_file(k: str, local: str | os.PathLike) -> Path:
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    client().download_file(bucket(), k, str(local))
    return local


def put_bytes(data: bytes, k: str, content_type: str = "application/octet-stream") -> str:
    client().put_object(Bucket=bucket(), Key=k, Body=data, ContentType=content_type)
    return k


def get_bytes(k: str) -> Optional[bytes]:
    try:
        return client().get_object(Bucket=bucket(), Key=k)["Body"].read()
    except client().exceptions.NoSuchKey:
        return None
    except Exception as e:  # some S3 implementations raise a generic 404 ClientError
        if getattr(e, "response", {}).get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise


def list_objects(prefix: str, s3=None, bkt: Optional[str] = None) -> list[dict]:
    s3 = s3 or client()
    out: list[dict] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bkt or bucket(), Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def delete_keys(keys: Iterable[str]) -> int:
    keys = list(keys)
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        client().delete_objects(Bucket=bucket(), Delete={"Objects": [{"Key": k} for k in chunk]})
    return len(keys)


def sync_down(prefix: str, dest: str | os.PathLike, s3=None, bkt: Optional[str] = None, workers: int = 16) -> int:
    """Download every object under prefix into dest, skipping files already present with the same size."""
    s3 = s3 or client()
    bkt = bkt or bucket()
    dest = Path(dest)
    prefix = prefix.rstrip("/") + "/"
    objs = [o for o in list_objects(prefix, s3=s3, bkt=bkt) if not o["Key"].endswith("/")]
    if not objs:
        raise R2Error(f"nothing found under s3://{bkt}/{prefix}")

    def one(o):
        target = dest / o["Key"][len(prefix) :]
        if target.exists() and target.stat().st_size == o["Size"]:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bkt, o["Key"], str(target))
        return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, objs))


def sync_up(src: str | os.PathLike, prefix: str, workers: int = 16) -> int:
    src = Path(src)
    files = [p for p in src.rglob("*") if p.is_file()]
    prefix = prefix.rstrip("/")

    def one(p: Path):
        put_file(p, f"{prefix}/{p.relative_to(src).as_posix()}")
        return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, files))


# ------------------------------------------------------------------------- runs


def read_status(rid: Optional[str] = None) -> Optional[dict]:
    raw = get_bytes(run_key("status.json", rid=rid))
    return json.loads(raw) if raw else None


def write_status(state: str, step: Optional[int] = None, message: Optional[str] = None, **extra) -> dict:
    """Merge into runs/<run_id>/status.json. A final state (done/failed/timeout) is not overwritten
    by a later non-final one, so a watchdog's 'timeout' survives the exit trap."""
    if state not in RUN_STATES:
        raise R2Error(f"state must be one of {RUN_STATES}")
    cur = read_status() or {}
    final = ("done", "failed", "timeout")
    if cur.get("state") in final and state not in final:
        return cur
    if cur.get("state") in ("timeout", "done") and state == "failed":
        return cur  # exit trap after a timeout kill or a clean finish must not relabel the run
    cur.update({k: v for k, v in extra.items() if v is not None})
    cur["state"] = state
    if step is not None:
        cur["step"] = int(step)
    cur.setdefault("step", 0)
    if message is not None:
        cur["message"] = message
    cur["updated_at"] = _now()
    cur.setdefault("started_at", cur["updated_at"])
    for env_name, field in (("RUNPOD_POD_ID", "pod_id"), ("RUN_GPU", "gpu"), ("GIT_SHA", "git_sha")):
        if os.environ.get(env_name):
            cur.setdefault(field, os.environ[env_name])
    put_bytes(json.dumps(cur, indent=2).encode(), run_key("status.json"), "application/json")
    return cur


def put_config(path: str | os.PathLike) -> str:
    return put_file(path, run_key("config.yaml"))


def step_ckpts(rid: Optional[str] = None) -> list[tuple[int, str]]:
    """[(step, key)] sorted by step ascending."""
    out = []
    for o in list_objects(run_key("ckpt/", rid=rid)):
        m = STEP_RE.search(o["Key"])
        if m:
            out.append((int(m.group(1)), o["Key"]))
    return sorted(out)


def prune_ckpts(keep: int = KEEP_CKPTS_DEFAULT, rid: Optional[str] = None) -> list[str]:
    """Keep the newest `keep` step_*.pt; best.pt is never touched."""
    if keep < 1:
        raise R2Error("keep must be >= 1")
    ckpts = step_ckpts(rid=rid)
    doomed = [k for _, k in ckpts[:-keep]]
    if doomed:
        delete_keys(doomed)
    return doomed


def put_ckpt(local: str | os.PathLike, step: Optional[int] = None, best: bool = False, keep: int = KEEP_CKPTS_DEFAULT) -> str:
    """Upload a checkpoint, then prune old ones. Pass step for step_XXXXXXXX.pt, best=True for best.pt."""
    if best:
        k = run_key("ckpt", "best.pt")
        put_file(local, k)
        return k
    if step is None:
        raise R2Error("put_ckpt needs step=<int> or best=True")
    k = run_key("ckpt", f"step_{int(step):08d}.pt")
    put_file(local, k)  # upload first; prune only after it succeeded
    prune_ckpts(keep=keep)
    write_status("running", step=step)  # no-op once the run is final
    return k


def latest_ckpt(rid: Optional[str] = None) -> Optional[str]:
    ckpts = step_ckpts(rid=rid)
    return ckpts[-1][1] if ckpts else None


def put_sample(local: str | os.PathLike, step: int) -> str:
    ext = Path(local).suffix or ".png"
    return put_file(local, run_key("samples", f"step_{int(step):08d}{ext}"))


def put_log(local: str | os.PathLike) -> str:
    return put_file(local, run_key("logs", Path(local).name))


def put_metrics(local: str | os.PathLike) -> str:
    """Upload the whole local metrics.jsonl (R2 has no append)."""
    return put_file(local, run_key("metrics.jsonl"))


def list_runs() -> list[str]:
    s3 = client()
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=key("runs") + "/", Delimiter="/"):
        out += [p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in page.get("CommonPrefixes", [])]
    return sorted(out)


# --------------------------------------------------------------------- datasets


def dataset_versions(name: str) -> list[str]:
    s3 = client()
    vs = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket(), Prefix=key("datasets", name) + "/", Delimiter="/"
    ):
        for p in page.get("CommonPrefixes", []):
            v = p["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if VERSION_RE.match(v):
                vs.append(v)
    return sorted(vs, key=lambda v: int(v[1:]))


def resolve_dataset(spec: str) -> str:
    """'datasets/<name>/LATEST' or 'datasets/<name>/vNNN' -> 'datasets/<name>/vNNN' (project-relative)."""
    parts = spec.strip("/").split("/")
    if len(parts) != 3 or parts[0] != "datasets":
        raise R2Error(f"dataset spec must be datasets/<name>/LATEST or datasets/<name>/vNNN, got {spec!r}")
    _, name, ver = parts
    if ver == "LATEST":
        raw = get_bytes(key("datasets", name, "LATEST"))
        if not raw:
            raise R2Error(f"{key('datasets', name, 'LATEST')} does not exist")
        ver = raw.decode().strip()
    if not VERSION_RE.match(ver):
        raise R2Error(f"bad dataset version {ver!r}")
    return f"datasets/{name}/{ver}"


def pull_dataset(spec: str, dest: str | os.PathLike = "data") -> Path:
    """Fetch a dataset to local disk and return its directory.

    spec forms:
      datasets/<name>/LATEST | datasets/<name>/vNNN   project data in R2 -> dest/<name>/
      hf:<repo_id>[@<revision>]                       Hugging Face dataset -> dest/<repo name>/
      s3-public://<bucket>/<prefix>[?region=<r>]      public S3 (e.g. nuScenes) -> dest/<last prefix part>/
      none                                            nothing to fetch
    """
    dest = Path(dest)
    if spec in ("", "none", "None"):
        return dest
    if spec.startswith("hf:"):
        repo, _, rev = spec[3:].partition("@")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise R2Error("huggingface_hub is not installed: pip install huggingface_hub")
        target = dest / repo.split("/")[-1]
        snapshot_download(repo_id=repo, repo_type="dataset", revision=rev or None, local_dir=str(target))
        return target
    if spec.startswith("s3-public://"):
        rest, _, query = spec[len("s3-public://") :].partition("?")
        bkt, _, prefix = rest.partition("/")
        region = dict(q.split("=", 1) for q in query.split("&") if "=" in q).get("region", "us-east-1")
        target = dest / (prefix.rstrip("/").rsplit("/", 1)[-1] or bkt)
        n = sync_down(prefix, target, s3=_public_client(region), bkt=bkt)
        print(f"downloaded {n} new files to {target}", file=sys.stderr)
        return target
    rel = resolve_dataset(spec)
    name = rel.split("/")[1]
    target = dest / name
    marker = target / ".r2-source"
    src = f"s3://{bucket()}/{key(rel)}"
    if target.exists() and (not marker.exists() or marker.read_text().strip() != src):
        import shutil  # local cache of another version (or unknown origin): start clean

        print(f"replacing {target} (was {marker.read_text().strip() if marker.exists() else 'unknown'})", file=sys.stderr)
        shutil.rmtree(target)
    n = sync_down(key(rel), target)
    marker.write_text(src + "\n")
    if not (target / "manifest.json").exists():
        print(f"warning: {key(rel)} has no manifest.json", file=sys.stderr)
    print(f"downloaded {n} new files from {key(rel)} to {target}", file=sys.stderr)
    return target


def publish_dataset(local_dir: str | os.PathLike, name: str, manifest: dict) -> str:
    """Upload local_dir as the next immutable version, write manifest.json, then move LATEST."""
    local_dir = Path(local_dir)
    vs = dataset_versions(name)
    ver = f"v{(int(vs[-1][1:]) + 1 if vs else 1):03d}"
    prefix = key("datasets", name, ver)
    files = [p for p in local_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    if not files:
        raise R2Error(f"{local_dir} has no files")
    sync_up(local_dir, prefix)
    m = dict(manifest)
    m.setdefault("created_at", _now())
    m.setdefault("num_files", len(files))
    m.setdefault("num_bytes", sum(p.stat().st_size for p in files))
    put_bytes(json.dumps(m, indent=2).encode(), f"{prefix}/manifest.json", "application/json")
    put_bytes(f"{ver}\n".encode(), key("datasets", name, "LATEST"), "text/plain")  # last
    return f"datasets/{name}/{ver}"


# -------------------------------------------------------------------------- CLI


def _cmd_check(a):
    s3 = client()
    s3.list_objects_v2(Bucket=bucket(), Prefix=key(), MaxKeys=1)
    print(f"ok: can list s3://{bucket()}/{key()}/ (r2.py {VERSION})")
    if a.write:
        k = key("_write_test")
        put_bytes(b"ok", k)
        delete_keys([k])
        print("ok: can write")


def _cmd_ls(a):
    prefix = key(a.prefix) if a.prefix else key() + "/"
    for o in list_objects(prefix):
        print(f"{o['Size']:>14,}  {o['LastModified']:%Y-%m-%d %H:%M}  {o['Key']}")


def _cmd_status(a):
    st = read_status(rid=a.run_id or run_id())
    print(json.dumps(st, indent=2) if st else "no status.json")


def _cmd_write_status(a):
    st = write_status(a.state, step=a.step, message=a.message)
    print(json.dumps(st))


def delete_run(rid: str, force: bool = False) -> int:
    """Delete everything under runs/<rid>/ (smoke tests, aborted runs). Refuses a live run unless force."""
    if not rid or "/" in rid:
        raise R2Error(f"bad run id {rid!r}")
    st = read_status(rid) or {}
    if st.get("state") in ("starting", "running") and not force:
        raise R2Error(f"run {rid} is {st['state']}; stop it first or pass --force")
    keys = [o["Key"] for o in list_objects(run_key("", rid=rid) + "/")]
    return delete_keys(keys)


def _cmd_rm_run(a):
    if not a.yes:
        n = len(list_objects(run_key("", rid=a.run_id) + "/"))
        print(f"would delete {n} objects under {key('runs', a.run_id)}/; rerun with --yes")
        return
    print(f"deleted {delete_run(a.run_id, force=a.force)} objects under {key('runs', a.run_id)}/")


def _cmd_put_ckpt(a):
    print(put_ckpt(a.file, step=a.step, best=a.best, keep=a.keep))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="verify credentials and bucket access")
    s.add_argument("--write", action="store_true", help="also test a write (needs a write token)")
    s.set_defaults(fn=_cmd_check)

    s = sub.add_parser("version")
    s.set_defaults(fn=lambda a: print(VERSION))

    s = sub.add_parser("ls", help="list objects under <project>/<prefix>")
    s.add_argument("prefix", nargs="?", default="")
    s.set_defaults(fn=_cmd_ls)

    s = sub.add_parser("get", help="download <project>/<key> to a local path ('-' = stdout)")
    s.add_argument("key"), s.add_argument("local")
    s.set_defaults(
        fn=lambda a: sys.stdout.buffer.write(get_bytes(key(a.key)) or b"")
        if a.local == "-"
        else print(get_file(key(a.key), a.local))
    )

    s = sub.add_parser("put", help="upload a local file to <project>/<key>")
    s.add_argument("local"), s.add_argument("key")
    s.set_defaults(fn=lambda a: print(put_file(a.local, key(a.key))))

    s = sub.add_parser("run-id", help="print a new run id for the current commit")
    s.add_argument("slug"), s.add_argument("--sha")
    s.set_defaults(fn=lambda a: print(make_run_id(a.slug, a.sha)))

    s = sub.add_parser("runs", help="list run ids")
    s.set_defaults(fn=lambda a: print("\n".join(list_runs())))

    s = sub.add_parser("status", help="print runs/<run_id>/status.json")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(fn=_cmd_status)

    s = sub.add_parser("rm-run", help="delete runs/<run_id>/ (dry run without --yes)")
    s.add_argument("run_id"), s.add_argument("--yes", action="store_true")
    s.add_argument("--force", action="store_true", help="also delete a run whose status is starting/running")
    s.set_defaults(fn=_cmd_rm_run)

    s = sub.add_parser("write-status", help="update status.json of $RUN_ID")
    s.add_argument("--state", required=True, choices=RUN_STATES)
    s.add_argument("--step", type=int), s.add_argument("--message")
    s.set_defaults(fn=_cmd_write_status)

    s = sub.add_parser("put-config", help="upload the run's config.yaml")
    s.add_argument("file")
    s.set_defaults(fn=lambda a: print(put_config(a.file)))

    s = sub.add_parser("put-ckpt", help="upload a checkpoint and prune old ones")
    s.add_argument("file")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--step", type=int), g.add_argument("--best", action="store_true")
    s.add_argument("--keep", type=int, default=KEEP_CKPTS_DEFAULT)
    s.set_defaults(fn=_cmd_put_ckpt)

    s = sub.add_parser("prune-ckpts")
    s.add_argument("--keep", type=int, default=KEEP_CKPTS_DEFAULT)
    s.set_defaults(fn=lambda a: print("\n".join(prune_ckpts(a.keep)) or "nothing to prune"))

    s = sub.add_parser("latest-ckpt", help="print the key of the newest step checkpoint")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(fn=lambda a: print(latest_ckpt(rid=a.run_id) or ""))

    s = sub.add_parser("put-sample")
    s.add_argument("file"), s.add_argument("--step", type=int, required=True)
    s.set_defaults(fn=lambda a: print(put_sample(a.file, a.step)))

    s = sub.add_parser("put-log")
    s.add_argument("file")
    s.set_defaults(fn=lambda a: print(put_log(a.file)))

    s = sub.add_parser("put-metrics")
    s.add_argument("file")
    s.set_defaults(fn=lambda a: print(put_metrics(a.file)))

    s = sub.add_parser("resolve-dataset")
    s.add_argument("spec")
    s.set_defaults(fn=lambda a: print(resolve_dataset(a.spec)))

    s = sub.add_parser("pull-dataset", help="fetch a dataset to local disk")
    s.add_argument("spec"), s.add_argument("dest", nargs="?", default="data")
    s.set_defaults(fn=lambda a: print(pull_dataset(a.spec, a.dest)))

    s = sub.add_parser("publish-dataset", help="upload a local dir as the next dataset version")
    s.add_argument("local_dir"), s.add_argument("name")
    s.add_argument("--manifest", required=True, help="JSON file with source_prefixes, converter_commit, schema, notes")
    s.set_defaults(fn=lambda a: print(publish_dataset(a.local_dir, a.name, json.loads(Path(a.manifest).read_text()))))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except R2Error as e:
        print(f"r2: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # botocore errors: one readable line, never credentials
        err = getattr(e, "response", {}).get("Error", {})
        if err:
            print(f"r2: {err.get('Code')}: {err.get('Message')} (bucket {os.environ.get('R2_BUCKET')})", file=sys.stderr)
            return 3
        if type(e).__module__.startswith(("botocore", "boto3", "s3transfer")):
            print(f"r2: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
