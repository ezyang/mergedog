"""``mergedog`` command-line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mergedog import notify, shepherd, stack_shepherd
from mergedog.config import LLM_PROVIDERS, get_llm_config, set_llm_config
from mergedog.paths import REPO_SLUG


def _parse_pr(value: str) -> int:
    """Accept either a bare PR number or a full PR URL."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    if "/pull/" in value:
        tail = value.rsplit("/pull/", 1)[-1]
        num = tail.split("/", 1)[0].split("#", 1)[0]
        if num.isdigit():
            return int(num)
    raise argparse.ArgumentTypeError(
        f"expected a PR number or {REPO_SLUG} PR URL, got {value!r}"
    )


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the single-PR and stack entry points."""
    parser.add_argument(
        "pr",
        type=_parse_pr,
        help=f"PR number (or full PR URL) on {REPO_SLUG}",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help=(
            "Before polling CI, refresh the PR's view of origin/main: for "
            "regular fork PRs this is a merge of origin/main into the PR "
            "branch (and a push); for ghstack PRs it's a ``git rebase`` of "
            "/orig followed by ``ghstack submit``. Default behavior is to "
            "never auto-refresh based on merge-base age -- mergedog only "
            "refreshes against main as a piggyback on a fix commit it was "
            "going to push anyway. Use --rebase when you want a one-shot "
            "upfront refresh on this run."
        ),
    )
    parser.add_argument(
        "--accept-divergence",
        action="store_true",
        help=(
            "Proceed even if the PR head differs from the latest maintainer "
            "approval's commit. Use this only after re-reviewing the "
            "additional commits yourself."
        ),
    )
    parser.add_argument(
        "--ignore-sev",
        action="store_true",
        help=(
            "Don't park on open ``ci: sev`` issues. By default mergedog "
            "waits before any action that would trigger fresh CI (claude "
            "fix invocations, pushes, ciflow/trunk label) so we don't "
            "stampede already-broken trunk."
        ),
    )
    parser.add_argument(
        "--reassess",
        action="store_true",
        help=(
            "Ignore previously-persisted spurious-check judgments and "
            "re-invoke claude for any current CI failures. Useful for "
            "testing or when you suspect a prior spurious verdict was wrong."
        ),
    )
    parser.add_argument(
        "--manage-mergedog-label",
        action="store_true",
        help=(
            "Add the ``mergedog`` coordination label at startup and remove "
            "it on exit. Disabled by default because label churn mutates the "
            "PR even when no CI/fix action is otherwise needed."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help=(
            "Override the on-disk root (default: ``~/.mergedog`` or the "
            "``MERGEDOG_ROOT`` env var). Useful to run a second mergedog "
            "session against a disjoint set of PRs without sharing the "
            "clone, worktrees, state, or logs of the default install. "
            "When invoked from ``mergedog.mux``, the env var is "
            "inherited by every spawned shepherd automatically."
        ),
    )
    parser.add_argument(
        "--gchat-to",
        metavar="USER",
        help=(
            "Send a Google Chat DM to USER (unixname) whenever the "
            "shepherd HALTs and needs human intervention. Requires the "
            "``meta`` CLI; silently skipped if ``meta`` is not installed."
        ),
    )
    extra = parser.add_mutually_exclusive_group()
    extra.add_argument(
        "--extra-context",
        metavar="TEXT",
        help=(
            "Operator-supplied hint string injected into claude's fix-CI "
            "prompt as a trusted section. Use to steer claude on this run "
            "(e.g. \"the lint failure on file X is pre-existing; ignore\"). "
            "Mutually exclusive with --extra-context-file."
        ),
    )
    extra.add_argument(
        "--extra-context-file",
        metavar="PATH",
        type=Path,
        help=(
            "Like --extra-context, but reads the hint text from a file. "
            "Useful for longer playbooks. Mutually exclusive with "
            "--extra-context."
        ),
    )


def _resolve_extra_context(args: argparse.Namespace) -> str | None:
    if args.extra_context_file is not None:
        try:
            return args.extra_context_file.read_text()
        except OSError as e:
            raise SystemExit(f"failed to read --extra-context-file: {e}")
    return args.extra_context


def _single_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog",
        description=(
            f"Autonomously shepherd an approved {REPO_SLUG} PR through "
            "CI to the point it is ready for human merge."
        ),
    )
    _add_common_flags(parser)
    args = parser.parse_args(argv)

    notify.configure(pr=args.pr, gchat_to=args.gchat_to)

    try:
        shepherd.shepherd(
            args.pr,
            rebase=args.rebase,
            accept_divergence=args.accept_divergence,
            ignore_sev=args.ignore_sev,
            reassess=args.reassess,
            manage_mergedog_label=args.manage_mergedog_label,
            extra_context=_resolve_extra_context(args),
        )
    except KeyboardInterrupt:
        print("\ninterrupted; partial state left in ~/.mergedog/", file=sys.stderr)
        return 130
    return 0


def _stack_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog stack",
        description=(
            "Shepherd a whole ghstack stack from a single foreground "
            "process. Pass any PR in the stack; mergedog reads the "
            "stack listing from the PR body and drives every PR in "
            "bottom-up order."
        ),
    )
    _add_common_flags(parser)
    parser.add_argument(
        "--force-ghstack",
        action="store_true",
        help=(
            "Pass ``--force`` to every ``ghstack submit`` invocation, "
            "bypassing ghstack's anti-clobber 'Cowardly refusing to "
            "push' check. Use when local bookkeeping disagrees with "
            "origin and you've decided it's safe to force the push -- "
            "an operator-controlled escape hatch for testing."
        ),
    )
    args = parser.parse_args(argv)

    notify.configure(pr=args.pr, gchat_to=args.gchat_to)

    try:
        stack_shepherd.run_stack(
            args.pr,
            rebase=args.rebase,
            accept_divergence=args.accept_divergence,
            ignore_sev=args.ignore_sev,
            reassess=args.reassess,
            force_ghstack=args.force_ghstack,
            manage_mergedog_label=args.manage_mergedog_label,
            extra_context=_resolve_extra_context(args),
        )
    except KeyboardInterrupt:
        print("\ninterrupted; partial state left in ~/.mergedog/", file=sys.stderr)
        return 130
    return 0


def _rage_main(argv: list[str]) -> int:
    stack = bool(argv and argv[0] == "stack")
    if stack:
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        prog="mergedog rage stack" if stack else "mergedog rage",
        description=(
            "Create a private paste with redacted diagnostics for "
            + ("a whole ghstack stack." if stack else "a PR.")
        ),
    )
    parser.add_argument(
        "pr",
        type=_parse_pr,
        help=f"PR number (or full PR URL) on {REPO_SLUG}",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        type=Path,
        help=(
            "Override the on-disk root to inspect (default: ``~/.mergedog`` "
            "or the ``MERGEDOG_ROOT`` env var)."
        ),
    )
    args = parser.parse_args(argv)

    from mergedog import rage as rage_mod
    from mergedog.paths import ROOT

    root = args.root if args.root is not None else ROOT
    try:
        paste = rage_mod.rage(args.pr, root=root, stack=stack)
    except Exception as e:
        print(f"rage failed: {e}", file=sys.stderr)
        return 1
    print(paste)
    return 0


def _config_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog config",
        description="Inspect or update persistent mergedog settings.",
    )
    subparsers = parser.add_subparsers(dest="setting", required=True)

    llm = subparsers.add_parser(
        "llm",
        help="Show or set the LLM provider used for fix/resolution agents.",
    )
    llm.add_argument(
        "provider",
        nargs="?",
        choices=LLM_PROVIDERS,
        help="Provider to use for future agent invocations.",
    )
    llm.add_argument(
        "--model",
        help=(
            "Provider-specific model name. Defaults to opus for claude and "
            "the provider CLI's default for codex/metacode."
        ),
    )
    llm.add_argument(
        "--clear-model",
        action="store_true",
        help="Remove any configured model override for this provider.",
    )
    args = parser.parse_args(argv)

    if args.setting == "llm":
        if args.provider is None:
            if args.model is not None or args.clear_model:
                parser.error(
                    "mergedog config llm needs a provider when changing model"
                )
            cfg = get_llm_config()
        else:
            cfg = set_llm_config(
                args.provider, model=args.model, clear_model=args.clear_model
            )
        model = cfg.effective_model
        if cfg.model:
            print(f"llm.provider={cfg.provider}")
            print(f"llm.model={cfg.model}")
        elif model:
            print(f"llm.provider={cfg.provider}")
            print(f"llm.model={model} (default)")
        else:
            print(f"llm.provider={cfg.provider}")
            print("llm.model=<provider default>")
        return 0

    raise AssertionError(f"unknown config setting: {args.setting}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Subcommand dispatch: ``mergedog stack <pr> [...]`` runs the stack
    # shepherd; everything else stays on the single-PR path so existing
    # ``mergedog <pr>`` invocations (and ``mux.py`` spawns) keep working.
    if argv and argv[0] == "stack":
        return _stack_main(argv[1:])
    if argv and argv[0] == "rage":
        return _rage_main(argv[1:])
    if argv and argv[0] == "config":
        return _config_main(argv[1:])
    return _single_main(argv)
