"""Train the Titanic model and launch the browser interface."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import webbrowser


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
HOST = "127.0.0.1"
PORT = 8000
MAX_PORT = 8010


def main() -> None:
    subprocess.run([sys.executable, str(PROJECT_DIR / "train.py")], cwd=PROJECT_DIR, check=True)

    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    server = None
    for port in range(PORT, MAX_PORT + 1):
        try:
            server = ThreadingHTTPServer((HOST, port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Could not start the website on ports {PORT}-{MAX_PORT}.")

    url = f"http://{HOST}:{server.server_port}/"
    print(f"\nTitanic Survival Oracle is running at {url}")
    print("Keep this terminal open. Press Ctrl+C to stop the server.")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the website.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
