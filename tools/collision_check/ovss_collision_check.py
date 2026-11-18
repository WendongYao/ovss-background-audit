#!/usr/bin/env python3
"""Collision check for background vocabularies in open-vocabulary semantic segmentation (OVSS).

A "collision" is a background word that also names (part of) a target query class, e.g. the background
word ``bed`` when ``bed`` is one of the classes to segment, or ``wall`` when ``wall-concrete`` is a query.
Such words make the model assign pixels of a target class to background.

The rule is purely lexical (class-name strings only, no images, no predictions). It is the rule used in
the paper (Section 3.3) and in its pre-registered contracts (S2, S3):

  1. Tokenise: split the background line and every query line by the separator (``;`` for ProxyCLIP- and
     Trident-style name files, ``,`` otherwise) and strip whitespace.
  2. Normalise: lowercase; replace ``-`` and ``_`` by spaces; drop a trailing word ``other`` or ``merged``;
     collapse whitespace.
  3. Match: background word w collides with query word q iff, after normalisation, w == q, or w is one of
     the space-separated words of q, or q is one of the space-separated words of w.
  4. Filter: remove colliding words from the background line only; query lines are never changed and the
     word ``background`` is never removed. If nothing is left, the background line becomes ``background``.

Name-file format (as used by SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP, ProxyCLIP, Trident, ...): one class
per line; the first non-empty line is the background class; a line may list several names (synonyms)
separated by the separator.

Usage:
  python ovss_collision_check.py NAME_FILE                      # background line vs. the other lines
  python ovss_collision_check.py NAME_FILE --targets CLASSES    # background line vs. a new class list
  python ovss_collision_check.py NAME_FILE --out CLEAN_FILE --json REPORT.json --sep ';' --strict

Exit status: 0 = done (also with collisions, unless --strict), 1 = collisions found with --strict, 2 = error.
Python >= 3.8, standard library only.
"""
import argparse
import json
import sys

__version__ = "1.0.0"
NEVER_REMOVE = "background"


def normalise(term):
    """Step 2 of the rule."""
    words = term.lower().replace("-", " ").replace("_", " ").split()
    if words and words[-1] in ("other", "merged"):
        words = words[:-1]
    return " ".join(words)


def collide(w, q):
    """Step 3 of the rule: True if background word w collides with query word q."""
    w, q = normalise(w), normalise(q)
    if not w or not q:
        return False
    return w == q or w in q.split() or q in w.split()


def split_line(line, sep):
    """Step 1 of the rule."""
    return [t.strip() for t in line.split(sep) if t.strip()]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\r") for l in f.read().split("\n") if l.strip()]


def detect_sep(lines):
    return ";" if any(";" in l for l in lines) else ","


def check(background_line, query_lines, sep=","):
    """Apply steps 1-4. Returns a dict with the collisions and the cleaned background line.

    background_line: str, the background class line (may hold several words).
    query_lines: list of str, one line per target class (each may hold several names).
    """
    bg = split_line(background_line, sep)
    queries = [t for l in query_lines for t in split_line(l, sep)]
    removed = [w for w in bg if w.lower() != NEVER_REMOVE and any(collide(w, q) for q in queries)]
    hits = {w: sorted({q for q in queries if collide(w, q)}) for w in removed}
    kept = [w for w in bg if w not in removed] or [NEVER_REMOVE]
    join = "; " if sep == ";" else ", "
    return {"n_background_words": len(bg), "n_removed": len(removed), "removed": hits,
            "kept": kept, "clean_background_line": join.join(kept), "separator": sep}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lexical collision check between a background vocabulary and target class names.")
    ap.add_argument("name_file", help="class-name file; first non-empty line = background line")
    ap.add_argument("--targets", help="optional class list to check against instead of the other lines of NAME_FILE")
    ap.add_argument("--sep", default="auto", help="separator inside a line: ',' or ';' (default: auto-detect)")
    ap.add_argument("--out", help="write NAME_FILE with the cleaned background line here")
    ap.add_argument("--json", help="write a JSON report here")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if any collision is found")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    try:
        lines = read_lines(a.name_file)
        if not lines:
            raise ValueError(f"{a.name_file} is empty")
        targets = read_lines(a.targets) if a.targets else lines[1:]
        sep = detect_sep(lines + targets) if a.sep == "auto" else a.sep
        if sep not in (",", ";"):
            raise ValueError("--sep must be ',' or ';'")
        rep = check(lines[0], targets, sep)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rep.update({"name_file": a.name_file, "targets": a.targets or "(other lines of name_file)", "tool_version": __version__})
    print(f"background words: {rep['n_background_words']}   colliding: {rep['n_removed']}   separator: '{sep}'")
    for w, qs in rep["removed"].items():
        print(f"  {w!r:>20}  collides with  {', '.join(repr(q) for q in qs)}")
    print(f"clean background line: {rep['clean_background_line']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join([rep["clean_background_line"]] + lines[1:]) + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    return 1 if (a.strict and rep["n_removed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
