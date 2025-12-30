from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


ALLOWED_LABELS = {"O", "B-METAPHOR"}
UNCERTAIN_LABEL = "-1"


@dataclass(frozen=True)
class TokenLabel:
    token: str
    label: str
    raw: str


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for line in file:
            yield line


def _parse_token_label(line: str, path: Path, line_no: int) -> TokenLabel:
    raw = line.rstrip("\n")
    if "\t" in raw:
        token, label = raw.split("\t", 1)
    else:
        parts = raw.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_no}: expected 'TOKEN<TAB>LABEL', got: {raw!r}")
        token, label = " ".join(parts[:-1]), parts[-1]
    return TokenLabel(token=token, label=label, raw=raw)


def _pairwise_files(es_dir: Path, en_dir: Path) -> Iterable[tuple[Path, Path, Path]]:
    for es_path in sorted(es_dir.rglob("*.tsv")):
        rel = es_path.relative_to(es_dir)
        en_path = en_dir / rel
        yield es_path, en_path, rel


def _ensure_allowed(label: str, path: Path, line_no: int) -> None:
    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"{path}:{line_no}: unexpected label {label!r} (expected one of {sorted(ALLOWED_LABELS)})"
        )


def triangulate_file(es_path: Path, en_path: Path, out_path: Path) -> dict[str, int]:
    if not en_path.exists():
        raise FileNotFoundError(f"Missing aligned file: {en_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"tokens": 0, "mismatches": 0, "sentences": 0}
    in_sentence = False

    es_iter = _iter_lines(es_path)
    en_iter = _iter_lines(en_path)

    with out_path.open("w", encoding="utf-8", newline="\n") as out_file:
        for line_no, (es_line, en_line) in enumerate(zip(es_iter, en_iter, strict=True), start=1):
            if es_line.strip() == "" and en_line.strip() == "":
                if in_sentence:
                    stats["sentences"] += 1
                    in_sentence = False
                out_file.write("\n")
                continue

            if (es_line.strip() == "") != (en_line.strip() == ""):
                raise ValueError(
                    f"Line misalignment at {es_path}:{line_no} / {en_path}:{line_no}: "
                    f"blank vs non-blank"
                )

            in_sentence = True
            es_item = _parse_token_label(es_line, es_path, line_no)
            en_item = _parse_token_label(en_line, en_path, line_no)

            if es_item.token != en_item.token:
                raise ValueError(
                    f"Token mismatch at line {line_no}:\n"
                    f"  {es_path}: {es_item.raw!r}\n"
                    f"  {en_path}: {en_item.raw!r}"
                )

            _ensure_allowed(es_item.label, es_path, line_no)
            _ensure_allowed(en_item.label, en_path, line_no)

            if es_item.label == en_item.label:
                out_label = es_item.label
            else:
                out_label = UNCERTAIN_LABEL
                stats["mismatches"] += 1

            stats["tokens"] += 1
            out_file.write(f"{es_item.token}\t{out_label}\n")

        if in_sentence:
            stats["sentences"] += 1

    return stats


def _default_paths(cwd: Path) -> tuple[Path, Path, Path]:
    # Prefer user-stated names if they exist; otherwise fall back to this repo's layout.
    es_cat = cwd / "es-cat"
    en_cat = cwd / "en-cat"
    if es_cat.is_dir() and en_cat.is_dir():
        return es_cat, en_cat, cwd / "es-en-cat"

    ca_es = cwd / "data" / "meta4xnli" / "detection" / "projected_labels" / "ca-es"
    ca_en = cwd / "data" / "meta4xnli" / "detection" / "projected_labels" / "ca-en"
    if ca_es.is_dir() and ca_en.is_dir():
        out_dir = cwd / "data" / "meta4xnli" / "detection" / "projected_labels_triangulated" / "es-en-cat"
        return ca_es, ca_en, out_dir

    return es_cat, en_cat, cwd / "es-en-cat"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Triangulate ES->CA and EN->CA projected labels over aligned CA tokens: "
            "keep the label when both projections agree, otherwise set it to -1."
        )
    )
    default_es_dir, default_en_dir, default_out_dir = _default_paths(Path.cwd())
    parser.add_argument("--es-dir", type=Path, default=default_es_dir, help="Directory for ES->CA projection TSVs")
    parser.add_argument("--en-dir", type=Path, default=default_en_dir, help="Directory for EN->CA projection TSVs")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir, help="Output directory for triangulated TSVs")
    args = parser.parse_args()

    es_dir: Path = args.es_dir
    en_dir: Path = args.en_dir
    out_dir: Path = args.out_dir

    if not es_dir.is_dir():
        raise SystemExit(f"--es-dir not found or not a directory: {es_dir}")
    if not en_dir.is_dir():
        raise SystemExit(f"--en-dir not found or not a directory: {en_dir}")

    total = {"tokens": 0, "mismatches": 0, "sentences": 0, "files": 0}

    for es_path, en_path, rel in _pairwise_files(es_dir, en_dir):
        out_path = out_dir / rel
        stats = triangulate_file(es_path, en_path, out_path)
        total["files"] += 1
        for key in ("tokens", "mismatches", "sentences"):
            total[key] += stats[key]

    print(
        "Done.\n"
        f"  es_dir: {es_dir}\n"
        f"  en_dir: {en_dir}\n"
        f"  out_dir: {out_dir}\n"
        f"  files: {total['files']}\n"
        f"  sentences: {total['sentences']}\n"
        f"  tokens: {total['tokens']}\n"
        f"  mismatches: {total['mismatches']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

