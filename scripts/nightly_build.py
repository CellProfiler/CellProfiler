import os
import re
# TODO: eventually switch to tomllib, introduced in python 3.11
import toml

def split_package(package):
    """
    given a string containing a package and its version specifier:
    "package-a>=3,<5"
    splits the string into 2-tuple of (package, version_specifier)
    ("package-a", ">=3,<5")
    """
    match = re.match(r'([a-zA-Z0-9-]+)(.*)', package)
    if match:
        return match.group(1), match.group(2).strip()
    else:
        raise Exception("Could not split package name")
    
def conv_deps_nightly(toml_data: dict) -> dict:
    split_deps = list(map(split_package, toml_data["project"]["dependencies"]))
    # converted deps, with -nightly attached to relevant ones
    conv_deps = list(map(lambda sd: sd[0] + "-nightly" + sd[1] if sd[0].lower().startswith('cellprofiler') else sd[0] + sd[1], split_deps))
    toml_data["project"]["dependencies"] = conv_deps
    return toml_data

def conv_name_nightly(toml_data: dict) -> dict:
    toml_data["project"]["name"] = toml_data["project"]["name"] + "-nightly"
    return toml_data

def toml_load(path: str) -> dict:
    with open(path, "r") as f:
        return toml.load(f)
    
def toml_save(path: str, data: dict) -> None:
    with open(path, "w") as f:
        toml.dump(data, f)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    filenames = ["../src/subpackages/library/pyproject.toml", "../src/subpackages/core/pyproject.toml", "../src/frontend/pyproject.toml"]
    for filename in filenames:
        toml_data = toml_load(filename)
        toml_data = conv_name_nightly(toml_data)
        toml_data = conv_deps_nightly(toml_data)
        toml_save(filename, toml_data)
