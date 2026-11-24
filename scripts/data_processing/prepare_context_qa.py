#!/usr/bin/env python3
"""Split context-only records and paired no-context QA records from evaluation JSONL."""

import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


TASKS = tuple(
    task + suffix
    for task in ("google_boolq", "squadv2", "google_re", "nq_swap", "clasheval")
    for suffix in ("", "_rnd_id")
)
QUESTION_HEADER = re.compile(r"(?:Question|<\d+>):")
ANSWER_CUE = re.compile(r"(?:Answer|<\d+>):")


def find_split(task_dir):
    for split in ("val", "test", "validation"):
        source = task_dir / f"{split}.jsonl"
        if source.is_file():
            return source
    raise FileNotFoundError(f"No val.jsonl, test.jsonl or validation.jsonl in {task_dir}")


def split_task(source):
    root = source.parent.parent
    task = source.parent.name
    context_file = root / f"{task}_context" / source.name
    qa_file = root / f"{task}_nocontext" / source.name

    with TemporaryDirectory(dir=root) as staging_dir:
        staged_context = Path(staging_dir) / "context.jsonl"
        staged_qa = Path(staging_dir) / "nocontext.jsonl"
        rows = 0
        with (
            source.open(encoding="utf-8") as original,
            staged_context.open("w", encoding="utf-8") as context_out,
            staged_qa.open("w", encoding="utf-8") as qa_out,
        ):
            for line_number, line in enumerate(original, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    context = row["context"]
                    if not isinstance(context, str) or not context.strip():
                        raise ValueError("Expected a nonempty context")
                    # Preserve the original suffix because rnd_id also transforms
                    # the section labels and can transform question text in place.
                    _, question, answer = row["prompt"].rsplit("\n\n", 2)
                    if not QUESTION_HEADER.match(question) or not ANSWER_CUE.fullmatch(answer.strip()):
                        raise ValueError("Expected a question section and an empty answer cue")
                    context_only = {"id": row["id"], "context": context}
                    no_context = dict(row, context="", prompt=f"{question}\n\n{answer}")
                    no_context.pop("title", None)
                    no_context.pop("prompt_ori", None)
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{source}:{line_number}: {error}") from error
                context_out.write(json.dumps(context_only, ensure_ascii=False) + "\n")
                qa_out.write(json.dumps(no_context, ensure_ascii=False) + "\n")
                rows += 1

        if not rows:
            raise ValueError(f"No records in {source}")
        context_file.parent.mkdir(exist_ok=True)
        qa_file.parent.mkdir(exist_ok=True)
        staged_context.replace(context_file)
        staged_qa.replace(qa_file)
    print(f"{task}/{source.name}: {rows:,} records in each output", file=sys.stderr)
    return context_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-root", type=Path, default=Path("input/evaluate_data/jsonl"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    sources = [find_split(args.jsonl_root / task) for task in dict.fromkeys(args.tasks)]
    for source in sources:
        context_file = split_task(source)
        print(f"{source.parent.name}\t{context_file}")


if __name__ == "__main__":
    main()
