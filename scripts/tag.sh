#!/usr/bin/env -S bash

ggrep -Po '(?<=version=\")\d+\.\d+\.\d+(?=\",)' setup.py | xargs -I{} git tag -a v{} -m 'version {}'
echo "Reminder: run 'git push --tags'"
