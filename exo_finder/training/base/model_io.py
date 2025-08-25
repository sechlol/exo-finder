from pathlib import Path


def get_checkpoint_path(base_path: Path, version: int) -> Path:
    log_folder = base_path / "lightning_logs" / f"version_{version}" / "checkpoints"
    if not log_folder.exists():
        raise FileNotFoundError(f"Checkpoint folder {log_folder} does not exist")
    return max(log_folder.glob("*.ckpt"), key=lambda p: p.stem.split("=")[-1])
