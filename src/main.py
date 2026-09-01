"""CLI for the face-recognition attendance system.

Phase 1 provides the enrollment pipeline:

    python -m src.main enroll --person-id alice --name "Alice Doe" \\
        --images samples/alice_1.npy samples/alice_2.npy
    python -m src.main list
    python -m src.main remove --person-id alice

Images may be ``.npy`` arrays (offline / tests) or ordinary image files when
Pillow is installed. The real dlib backend is selected with ``--backend dlib``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.embeddings import load_embedder
from src.enrollment import EnrollmentError, EnrollmentStore, enroll_person
from src.images import ImageLoadError, load_image

DEFAULT_STORE = Path("data/enrollments.json")


def _cmd_enroll(args: argparse.Namespace) -> int:
    store = EnrollmentStore(args.store)
    try:
        embedder = load_embedder(args.backend)
        images = [load_image(p) for p in args.images]
        record = enroll_person(
            store,
            args.person_id,
            args.name or args.person_id,
            images,
            embedder,
            on_duplicate=args.on_duplicate,
        )
    except (EnrollmentError, ImageLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"enrolled {record.person_id} ({record.name}) from {record.num_samples} "
        f"image(s) using backend '{record.backend}'"
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    store = EnrollmentStore(args.store)
    if not len(store):
        print("no enrolled people")
        return 0
    for rec in store.all():
        print(f"{rec.person_id}\t{rec.name}\tsamples={rec.num_samples}\t{rec.updated_at}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    store = EnrollmentStore(args.store)
    if store.remove(args.person_id):
        print(f"removed {args.person_id}")
        return 0
    print(f"error: {args.person_id} is not enrolled", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attendance", description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="Enrollment registry path.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Enroll a person from one or more images.")
    p_enroll.add_argument("--person-id", required=True)
    p_enroll.add_argument("--name", default="")
    p_enroll.add_argument("--images", nargs="+", required=True, type=Path)
    p_enroll.add_argument("--backend", default="hash", choices=["hash", "dlib"])
    p_enroll.add_argument("--on-duplicate", default="update", choices=["update", "reject"])
    p_enroll.set_defaults(func=_cmd_enroll)

    p_list = sub.add_parser("list", help="List enrolled people.")
    p_list.set_defaults(func=_cmd_list)

    p_remove = sub.add_parser("remove", help="Remove an enrolled person.")
    p_remove.add_argument("--person-id", required=True)
    p_remove.set_defaults(func=_cmd_remove)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
