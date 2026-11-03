import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_frozen(executable):
    executable = Path(executable).resolve(strict=True)
    environment = os.environ.copy()
    for key in list(environment):
        if key.upper().startswith(("PIXI_", "CONDA_")) or key.upper() in {
            "PYTHONPATH", "PYTHONHOME", "JAVA_HOME", "JDK_HOME",
            "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__",
            "CP_JAVA_HOME", "CP_JAVA_JARS", "CLASSPATH",
        }:
            environment.pop(key)
    system_root = os.environ["SystemRoot"]
    environment["PATH"] = os.pathsep.join(
        [str(Path(system_root) / "System32"), system_root]
    )
    with tempfile.TemporaryDirectory(prefix="cp-frozen-smoke-") as directory:
        environment["APPDATA"] = str(Path(directory) / "appdata")
        process = subprocess.Popen(
            [str(executable), "--help"],
            cwd=directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        try:
            stdout, _ = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False
            )
            stdout, _ = process.communicate(timeout=15)
            print(stdout)
            return 124
        print(stdout)
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Check frozen executable startup")
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Run the executable check on Windows")
    return check_frozen(args.executable)


if __name__ == "__main__":
    sys.exit(main())
