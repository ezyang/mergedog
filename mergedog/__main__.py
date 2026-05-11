import sys


from mergedog.bootstrap import promote_early_env


promote_early_env(sys.argv[1:])

from mergedog.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
