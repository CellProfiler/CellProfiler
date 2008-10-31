#!/bin/bash
## Finds all *.c, *.cpp, *.m, *.h, *.txt, *.command, *.py files and sets svn properties.
## Should be run from the DeveloperScripts directory (or at least from one directory below the trunk) to find all files properly.

## Notes:
## The user will still need to add the $Revision: keyword for the Revision # to be updated.
## Setting you auto-propset config values (in ~/.subversion/config) is preferable to running this script, but sometimes it may be necessary

find .. \( -name \*.c -or -name \*.cpp -or -name \*.m -or -name \*.h -or -name \*.txt -or -name \*.command -or -name \*.py \) -exec svn propset svn:keywords "Author Date Id Revision" {} \;
find .. \( -name \*.c -or -name \*.cpp -or -name \*.m -or -name \*.h -or -name \*.txt -or -name \*.command -or -name \*.py \) -exec svn propset svn:eol-style "native" {} \;

