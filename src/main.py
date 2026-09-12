"""CLI for the face-recognition attendance system.

Phase 1 provides the enrollment pipeline:

    python -m src.main enroll --person-id alice --name "Alice Doe" \\
        --images samples/alice_1.npy samples/alice_2.npy
    python -m src.main list
    python -m src.main remove --person-id alice
    python -m src.main recognize --image samples/frame.npy --threshold 0.6

Images may be ``.npy`` arrays (offline / tests) or ordinary image files when
Pillow is installed. The real dlib backend is selected with ``--backend dlib``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.attendance import AttendanceStore, DEFAULT_DEDUP_WINDOW_SECONDS
from src.embeddings import load_embedder
from src.enrollment import EnrollmentError, EnrollmentStore, enroll_person
from src.images import ImageLoadError, load_image
from src.matching import DEFAULT_THRESHOLD, NoEnrollmentsError, match_embedding

DEFAULT_STORE = Path("data/enrollments.json")
DEFAULT_ATTENDANCE_DB = Path("data/attendance.db")


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


def _cmd_recognize(args: argparse.Namespace) -> int:
    store = EnrollmentStore(args.store)
    try:
        embedder = load_embedder(args.backend)
        image = load_image(args.image)
        embedding = embedder.embed(image)
        result = match_embedding(embedding, store, threshold=args.threshold)
    except (ImageLoadError, ValueError, NoEnrollmentsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.person_id == "unknown":
        print(f"unknown (best similarity {result.similarity:.3f} < threshold {args.threshold:.3f})")
        return 0

    print(f"{result.person_id}\t{result.name}\tsimilarity={result.similarity:.3f}")
    if not args.no_checkin:
        with AttendanceStore(args.attendance_db, dedup_window_seconds=args.dedup_window_seconds) as attendance:
            checkin = attendance.check_in(result.person_id)
        if checkin.logged:
            print(f"checked in {result.person_id} at {checkin.record.timestamp}")
        else:
            print(
                f"duplicate check-in for {result.person_id} within "
                f"{args.dedup_window_seconds:.0f}s window (last: {checkin.record.timestamp}) -- not logged again"
            )
    return 0


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

    p_recognize = sub.add_parser("recognize", help="Match one image against enrolled people.")
    p_recognize.add_argument("--image", required=True, type=Path)
    p_recognize.add_argument("--backend", default="hash", choices=["hash", "dlib"])
    p_recognize.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_recognize.add_argument("--attendance-db", type=Path, default=DEFAULT_ATTENDANCE_DB)
    p_recognize.add_argument("--dedup-window-seconds", type=float, default=DEFAULT_DEDUP_WINDOW_SECONDS)
    p_recognize.add_argument(
        "--no-checkin", action="store_true", help="Recognize without logging a check-in."
    )
    p_recognize.set_defaults(func=_cmd_recognize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
