import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_tests(suite):
    args = [sys.executable, "-m", "pytest", f"tests/{suite}"]
    if suite == "core":
        args.extend(
            [
                "--timeout=180",
                "-k",
                "not test_load_objects and not test_load_single_object",
            ]
        )
    return subprocess.call(args)


def run_with_mysql():
    import MySQLdb

    os.environ.update(
        CP_MYSQL_TEST_HOST="127.0.0.1",
        CP_MYSQL_TEST_DB="cellprofiler_test",
        CP_MYSQL_TEST_USER="root",
        CP_MYSQL_TEST_PASSWORD="None",
    )
    mysqld = shutil.which("mysqld")
    if mysqld is None:
        raise RuntimeError("The Pixi environment does not contain mysqld")
    base_dir = Path(sys.prefix) / "Library" if os.name == "nt" else Path(sys.prefix)
    with tempfile.TemporaryDirectory(prefix="cp-mysql-") as directory:
        root = Path(directory)
        log_path = root / "mysql.log"
        command = [
            mysqld,
            "--no-defaults",
            f"--basedir={base_dir}",
            f"--plugin-dir={base_dir / 'lib' / 'plugin'}",
            f"--lc-messages-dir={base_dir / 'share' / 'mysql'}",
            f"--datadir={root / 'data'}",
            f"--log-error={log_path}",
        ]
        server = None
        try:
            subprocess.run(command + ["--initialize-insecure"], check=True)
            server_args = [
                "--bind-address=127.0.0.1",
                "--port=3306",
                "--mysqlx=0",
                "--local-infile=1",
            ]
            if os.name != "nt":
                server_args.append(f"--socket={root / 'mysql.sock'}")
            server = subprocess.Popen(command + server_args)
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError("MySQL exited before accepting connections")
                try:
                    connection = MySQLdb.connect(
                        host="127.0.0.1", user="root", connect_timeout=2
                    )
                    break
                except MySQLdb.OperationalError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
            try:
                with connection.cursor() as cursor:
                    cursor.execute("CREATE DATABASE cellprofiler_test")
            finally:
                connection.close()
            return run_tests("frontend")
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if log_path.exists():
                print(log_path.read_text(errors="replace"), flush=True)


if __name__ == "__main__":
    suite = sys.argv[1]
    if suite not in {"frontend", "core", "library", "centriole"}:
        raise ValueError(f"Unknown test suite: {suite}")
    sys.exit(run_with_mysql() if suite == "frontend" else run_tests(suite))
