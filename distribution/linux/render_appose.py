import argparse
import json
from pathlib import Path
import re
import sys
import tomllib


def render_appose(template: str, manifest_path: Path) -> str:
    with manifest_path.open('rb') as stream:
        manifest = tomllib.load(stream)
    dependency = manifest.get('pypi-dependencies', {}).get('appose')
    if not isinstance(dependency, dict):
        raise ValueError('Appose must be declared as a pinned Git dependency in pixi.toml')
    url = dependency.get('git')
    commit = dependency.get('rev')
    if not isinstance(url, str) or not url:
        raise ValueError('Appose must have a Git URL in pixi.toml')
    if not isinstance(commit, str) or re.fullmatch(r'[0-9a-fA-F]{40}', commit) is None:
        raise ValueError('Appose must be pinned to a full Git commit in pixi.toml')
    for token, value in [('{{APPOSE_GIT_URL}}', url), ('{{APPOSE_COMMIT}}', commit)]:
        if template.count(token) != 1:
            raise ValueError(f'The Flatpak template must contain exactly one {token}')
        template = template.replace(token, json.dumps(value))
    return template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        rendered = render_appose(sys.stdin.read(), args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(rendered)


if __name__ == '__main__':
    main()
