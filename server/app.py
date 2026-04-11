"""Compatibility wrapper around the packaged FastAPI app."""

from medtriage_env.server.app import app, main as packaged_main

__all__ = ["app", "main"]


def main() -> None:
    packaged_main()


if __name__ == "__main__":
    main()
