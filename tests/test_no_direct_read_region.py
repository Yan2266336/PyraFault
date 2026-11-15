"""Only pyrafault/io/read.py and pyrafault/io/slide.py may touch raw OpenSlide reads.

Walks both pyrafault/ (the library) and scripts/ (the ten stage CLIs that do all the
actual pixel work) -- the original guard covered only the former, leaving the stage
scripts, which are the modules most likely to reach for a "quick" direct read, entirely
unguarded.
"""
import pathlib
import re

ALLOWED = {"pyrafault/io/read.py", "pyrafault/io/slide.py"}
WALK_ROOTS = ("pyrafault", "scripts")

# Matches the bare `read_region` identifier (the raw OpenSlide call), but not as a
# prefix of a longer identifier such as `read_region_at_mpp` — the latter is a
# provenance-preserving wrapper defined in pyrafault/io/read.py that other modules are
# expected to import and call by name.
RAW_READ_REGION = re.compile(r"\bread_region\b")


def test_read_region_is_not_called_outside_the_io_layer():
    offenders = []
    for root_name in WALK_ROOTS:
        root = pathlib.Path(root_name)
        for path in root.rglob("*.py"):
            rel = path.as_posix()
            if rel in ALLOWED:
                continue
            if RAW_READ_REGION.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)
    assert offenders == [], (
        f"{offenders} call read_region directly; all pixel access must go through "
        "pyrafault.io.read.read_at_mpp/read_region_at_mpp so extraction stays "
        "deterministic and versioned"
    )
