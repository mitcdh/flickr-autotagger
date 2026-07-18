#!/usr/bin/env python3
"""Backward-compatible entry point for the Flickr autotagger CLI."""

from autotagger.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
