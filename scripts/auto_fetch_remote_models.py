#!/usr/bin/env python3
"""Poll a remote training server and download finished model artifacts.

支持多套训练产物预设，便于：
- 兼容旧版训练目录命名
- 下载当前新版 `goal-conditioned / sparse / joint_position_delta` 训练轮次
- 按“全部产物就绪后统一回传”或“谁先完成先回传”两种模式工作
"""

from __future__ import annotations

import argparse
import posixpath
import stat
import sys
import time
from pathlib import Path

import paramiko


ARTIFACT_PRESETS = {
    "legacy_total_queue": {
        "classic_ppo_final": (
            "models/classic/ppo/ur5_cxy/server_classic_ppo/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_ppo_best": (
            "logs/classic/ppo/ur5_cxy/server_classic_ppo/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
        "classic_sac_final": (
            "models/classic/sac/ur5_cxy/server_classic_sac/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_sac_best": (
            "logs/classic/sac/ur5_cxy/server_classic_sac/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
        "classic_td3_final": (
            "models/classic/td3/ur5_cxy/server_classic_td3/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_td3_best": (
            "logs/classic/td3/ur5_cxy/server_classic_td3/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
        "warp_ppo_final": (
            "models/warp_gpu/ppo/ur5_cxy/server_warp_ppo",
            ("final_policy.msgpack", "config.json"),
        ),
        "warp_sac_final": (
            "models/warp_gpu/sac/ur5_cxy/server_warp_sac",
            ("final_policy.msgpack", "config.json"),
        ),
    },
    "gc_total_queue": {
        "warp_ppo_gc_final": (
            "models/warp_gpu/ppo/ur5_cxy/server_warp_ppo_gc",
            ("final_policy.msgpack", "config.json"),
        ),
        "warp_sac_gc_final": (
            "models/warp_gpu/sac/ur5_cxy/server_warp_sac_gc",
            ("final_policy.msgpack", "config.json"),
        ),
        "classic_sac_her_final": (
            "models/classic/sac/ur5_cxy/server_classic_sac_her/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_sac_her_best": (
            "logs/classic/sac/ur5_cxy/server_classic_sac_her/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
        "classic_td3_her_final": (
            "models/classic/td3/ur5_cxy/server_classic_td3_her/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_td3_her_best": (
            "logs/classic/td3/ur5_cxy/server_classic_td3_her/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
        "classic_ppo_gc_final": (
            "models/classic/ppo/ur5_cxy/server_classic_ppo_gc/final",
            ("model.zip", "vec_normalize.pkl"),
        ),
        "classic_ppo_gc_best": (
            "logs/classic/ppo/ur5_cxy/server_classic_ppo_gc/best_model",
            ("best_model.zip", "vec_normalize.pkl"),
        ),
    },
}


def _remote_is_dir(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(remote_path).st_mode)
    except OSError:
        return False


def _remote_exists(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        sftp.stat(remote_path)
        return True
    except OSError:
        return False


def _download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_child = posixpath.join(remote_dir, entry.filename)
        local_child = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            _download_dir(sftp, remote_child, local_child)
        else:
            sftp.get(remote_child, str(local_child))


def _artifact_ready(sftp: paramiko.SFTPClient, remote_root: str, rel_dir: str, required_files: tuple[str, ...]) -> bool:
    remote_dir = posixpath.join(remote_root, rel_dir)
    if not _remote_is_dir(sftp, remote_dir):
        return False
    return all(_remote_exists(sftp, posixpath.join(remote_dir, name)) for name in required_files)


def _download_artifact(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    local_root: Path,
    artifact_name: str,
    rel_dir: str,
    layout: str,
) -> None:
    remote_dir = posixpath.join(remote_root, rel_dir)
    if layout == "artifact":
        local_dir = local_root / artifact_name
    else:
        local_dir = local_root / Path(rel_dir)
    _download_dir(sftp, remote_dir, local_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-download remote trained models when artifacts appear.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", default="downloads/remote_models")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--layout",
        choices=("mirror", "artifact"),
        default="mirror",
        help="本地落盘方式：mirror 按远端目录镜像；artifact 按预设名平铺。",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(ARTIFACT_PRESETS.keys()),
        default="legacy_total_queue",
        help="选择本轮训练对应的产物目录预设。",
    )
    parser.add_argument(
        "--download-mode",
        choices=("all_at_end", "incremental"),
        default="all_at_end",
        help="all_at_end: 全部产物就绪后统一回传；incremental: 谁先就绪先下载谁。",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    local_root = Path(args.local_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    artifacts = ARTIFACT_PRESETS[args.preset]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    sftp = client.open_sftp()
    downloaded: set[str] = set()
    try:
        while len(downloaded) < len(artifacts):
            ready_now: list[tuple[str, str]] = []
            for artifact_name, (rel_dir, required_files) in artifacts.items():
                if artifact_name in downloaded:
                    continue
                if _artifact_ready(sftp, args.remote_root, rel_dir, required_files):
                    ready_now.append((artifact_name, rel_dir))

            if args.download_mode == "incremental":
                for artifact_name, rel_dir in ready_now:
                    if artifact_name in downloaded:
                        continue
                    print(f"[ready] {artifact_name}: {rel_dir}", flush=True)
                    _download_artifact(sftp, args.remote_root, local_root, artifact_name, rel_dir, args.layout)
                    downloaded.add(artifact_name)
                    target_dir = (local_root / artifact_name) if args.layout == "artifact" else (local_root / Path(rel_dir))
                    print(f"[downloaded] {artifact_name} -> {target_dir}", flush=True)
            else:
                if len(ready_now) == len(artifacts):
                    print(f"[ready] all configured artifacts are available ({len(artifacts)}/{len(artifacts)})", flush=True)
                    for artifact_name, rel_dir in ready_now:
                        _download_artifact(sftp, args.remote_root, local_root, artifact_name, rel_dir, args.layout)
                        downloaded.add(artifact_name)
                        target_dir = (local_root / artifact_name) if args.layout == "artifact" else (local_root / Path(rel_dir))
                        print(f"[downloaded] {artifact_name} -> {target_dir}", flush=True)
            if len(downloaded) < len(artifacts):
                print(
                    f"[poll] preset={args.preset} mode={args.download_mode} ready={len(ready_now)}/{len(artifacts)} downloaded={len(downloaded)}/{len(artifacts)} waiting={args.poll_seconds}s",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
    finally:
        sftp.close()
        client.close()
    print("[done] all configured artifacts downloaded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
