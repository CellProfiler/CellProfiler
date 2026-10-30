import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import jpype
from cellprofiler_core.preferences import set_awt_headless
from cellprofiler_core.utilities.java import start_java, stop_java


def main():
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    set_awt_headless(True)
    start_java()
    try:
        jpype.JClass("loci.formats.ImageReader")
        classpath = jpype.getClassPath().split(os.pathsep)
        classpath += str(
            jpype.JClass("java.lang.System").getProperty("java.class.path")
        ).split(os.pathsep)
        jars = {}
        for entry in classpath:
            for path in sorted(glob.glob(entry)):
                source = Path(path).resolve()
                if source.suffix != ".jar":
                    continue
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if source.name in jars and jars[source.name]["sha256"] != digest:
                    raise RuntimeError(f"Conflicting Java jars: {source.name}")
                jars[source.name] = {"source": str(source), "sha256": digest}
                shutil.copy2(source, destination / source.name)
        if not jars:
            raise RuntimeError("No Java jars were resolved")
        for path in destination.glob("*.jar"):
            if path.name not in jars:
                path.unlink()
        manifest = {"java_home": os.environ.get("JAVA_HOME"), "jars": jars}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        stop_java()


if __name__ == "__main__":
    main()
