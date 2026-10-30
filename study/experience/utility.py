import hashlib
import json
from pathlib import Path

def read(path: Path):
    return json.loads(path.read_text())

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)

def correction_source(seed: int, role: str, success: bool) -> str:
    if role not in ('P', 'G') or type(success) is not bool:
        raise ValueError('source role/success')
    return 'generated' if role == 'G' and success else 'parent'
