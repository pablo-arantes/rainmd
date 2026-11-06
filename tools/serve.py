#!/usr/bin/env python3
"""Serve the repository for development.

🔴 USE THIS RATHER THAN `python3 -m http.server`. A plain http.server sends no
cache headers, so Chrome caches the ES modules heuristically and a reload serves
the old ones - an edit lands, nothing changes, and the code looks wrong. This
sends `no-store` on everything, which is right for every file here because
RainMD has no large cacheable asset: a force-field topology arrives from the
user's disk, not from the server.
"""
import argparse
import functools
import http.server
import os
import pathlib


class NoStoreHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        # WebGPU needs no special headers, but a SharedArrayBuffer-based worker
        # would; left here as the one place to add them if that ever changes.
        super().end_headers()

    def log_message(self, format, *args):
        if "GET" in format % args and " 200 " not in format % args:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    os.chdir(root)
    handler = functools.partial(NoStoreHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {root} at http://{args.host}:{args.port}/  (no-store)")
    print("open that URL in Chrome/Edge 113+ or Safari 18+")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
