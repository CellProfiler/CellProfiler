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


TEST_HOST = "127.0.0.1"
TEST_DB = "cellprofiler_test"
TEST_USER = "root"
TEST_PASSWORD = "None"

def run_with_mysql():
    import MySQLdb

    os.environ.update(
        CP_MYSQL_TEST_HOST=TEST_HOST,
        CP_MYSQL_TEST_DB=TEST_DB,
        CP_MYSQL_TEST_USER=TEST_USER,
        CP_MYSQL_TEST_PASSWORD=TEST_PASSWORD,
    )
    mysqld = shutil.which("mysqld")
    if mysqld is None:
        raise RuntimeError("The Pixi environment does not contain mysqld")
    base_dir = Path(sys.prefix) / "Library" if os.name == "nt" else Path(sys.prefix)
    with tempfile.TemporaryDirectory(prefix="cp-mysql-") as directory:
        rootdir = Path(directory)
        log_path = rootdir / "mysql.log"
        command = [
            mysqld,
            # ignore default config, e.g. /etc/my.cnf, ~/.my.cnf
            "--no-defaults",
            # root path to MySQL installation dir
            f"--basedir={base_dir}",
            # engine and authentication plugins
            f"--plugin-dir={base_dir / 'lib' / 'plugin'}",
            # localized server error messages
            f"--lc-messages-dir={base_dir / 'share' / 'mysql'}",
            # table files, metadata, system tables
            f"--datadir={rootdir / 'data'}",
            # startup and runtime errors
            f"--log-error={log_path}",
        ]
        server = None
        try:
            # server init: create system database tables, generate root@localhost
            # .run: synchronous, blocking, one-time task; exit on complete
            subprocess.run(command + ["--initialize-insecure"], check=True)
            server_args = [
                # restrict incoming connections to local machine (localhost)
                f"--bind-address={TEST_HOST}",
                # listen on default MySQL TCP port
                "--port=3306",
                # disable MySQL X plugin (document store/ X DevAPI)
                # reduces overhead and frees port 33060
                "--mysqlx=0",
                # enable LOAD DATA LOCAL INFILE, needed by data pipelines like
                # CellProfiler to bulk-load CSV/tab-delimited measurement data
                # directly into database tables
                "--local-infile=1",
            ]
            if os.name == "nt":
                server_args.append("--no-monitor")
            else:
                # Unix domain socket file for local client connection via socket
                # rather than TCP, where preferred by the client
                server_args.append(f"--socket={rootdir / 'mysql.sock'}")
            # Popen: async, non-blocking, long-running daemon; keep in background
            server = subprocess.Popen(command + server_args)
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError("MySQL exited before accepting connections")
                try:
                    connection = MySQLdb.connect(
                        host=TEST_HOST, user=TEST_USER, connect_timeout=2
                    )
                    break
                except MySQLdb.OperationalError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(f"CREATE DATABASE {TEST_DB}")
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
