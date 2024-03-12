#!/bin/bash
#
# deps:
#   pip install py-spy
#   https://github.com/benfred/py-spy
#   
#   pip install memray
#   https://github.com/bloomberg/memray
# 
# example usage:
#   remove all files in default output dir (use zsh N glob to avoid error) if dir empty
#   then run doprofile script
#   generate sppedscope json for speedscope.app
#   prefix naem with "profile"
#   launch CP automatically
#   in headless mode
#   pass -p and -o args to cellprofiler
#
#   rm -f -- ~/Documents/cellprofiler/*(N) && doprofile.sh --speedscope --outname headless_profile --launch --headless -- --pipeline="profiles/beginner_seg/src/beginner_seg.cppipe" --image-directory="profiles/beginner_seg/src/20585_images" -o ~/Documents/cellprofiler |& tee guid_log.txt

# exit on error
set -e

PROCNAME="CellProfiler"
LAUNCHER="${CONDA_PREFIX}/python.app/Contents/MacOS/python"
OUTNAME="profile"
THEPID=""
B_DOLAUNCH=false
CPARGS=""
DEFAULT_PROFILE_TYPE="svg"
PROFILE_TYPE="${DEFAULT_PROFILE_TYPE}"

usage () {
  echo "usage:"
  echo "  doprofiler.sh [--speedscope] [--outname <name>] [-- <cp_args..>]"
  echo "    try to find existing process and attach to pid"
  echo "  doprofiler.sh --pid <pid> [--speedscope] [--memray] [--outname <name>] [-- <cp_args..>]"
  echo "    attach to existing passed in pid"
  echo "  doprofiler.sh --launch [--speedscope] [--memray] [--outname <name>] [-- <cp_args..>] [-- <cp_args..>] [--gui | --headless]"
  echo "    launch process and attach to its pid"
  echo ""
  echo "  --outname is the name of the output profile with .svg, .json, or .bin automatically appended"
  echo "  --speedscope will output a speedscope .json file rather than a plain .svg"
  echo "      open in www.speedscope.app"
  echo "  --memray will output a memray .bin which can be converted to html with the memray's report generators"
  echo "  --speedscope and --memray are mutually exclusive"
  echo "  [--gui | --headless] if --launch, choose whether to run the GUI (default) or headless mode"
  echo "  -- <cp_args..> are cellprofiler arguments e.g. -h"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outname)
            OUTNAME="${2}"
            shift 2
            ;;
        --pid)
            THEPID="$2"
            shift 2
            ;;
        --launch)
            B_DOLAUNCH=true
            shift 1
            ;;
        --help)
            usage
            exit 0
            ;;
        -h)
            usage
            exit 0
            ;;
        --gui)
            GUI=true
            shift 1
            ;;
        --headless)
            GUI=false
            shift 1
            ;;
        --speedscope)
            if [[ "${PROFILE_TYPE}" != "${DEFAULT_PROFILE_TYPE}" ]]; then
                echo "Error: can't give both --speedscope and --memray"
                usage
                exit 1
            fi
            PROFILE_TYPE="speedscope"
            shift 1
            ;;
        --memray)
            if [[ "${PROFILE_TYPE}" != "${DEFAULT_PROFILE_TYPE}" ]]; then
                echo "Error: can't give both --speedscope and --memray"
                usage
                exit 1
            fi
            PROFILE_TYPE="memray"
            shift 1
            ;;
        --)
            shift 1
            CPARGS+=${*}
            shift $#
            ;;
        *)
            echo "Invalid argument: $1"
            usage
            exit 1
            ;;
    esac
done

# uncomment to enable debugging
#trap '(echo -e -n "\033[0;38;2;3;252;53m[$BASH_SOURCE:$LINENO]\033[0m $BASH_COMMAND" && read)' DEBUG

# if --gui or --headless weren't passed, then GUI wasn't set, do so
if [ -z ${GUI+x} ]; then
    GUI=true
else
    # if we passed --gui or --headless, but not --launch, err
    if ! ${B_DOLAUNCH}; then
        echo "Error: passed --gui or --headless, without specifying --launch"
        usage
        exit 1
    fi
fi

# if we passed both --launch and --pid, err
if ${B_DOLAUNCH} && [[ -n "${THEPID}" ]]; then
    echo "Error: can't give both --pid and --launch"
    usage
    exit 1
fi

# prints nothing but does so with sudo forcing credentials
# for later sudo call of py-spy
# important because program is potentially run as background job first
# so we don't have time to wait for user to input credentials between
# program start and py-spy invocation
# assumes sudo timeout is not 0
# https://apple.stackexchange.com/questions/10139/how-do-i-increase-sudo-password-remember-timeout
sudo printf "\033[0K"

if ${B_DOLAUNCH}; then
    # debug flag args, always
    CPARGS+=" -L 10"
    if ! ${GUI}; then
        # headless flags
        CPARGS+=" -c -r"
    fi
fi

# if we passed --launch, launch the program
# but not with memray, because it can launch without needing to be attached
# to a preexisting pid
if ${B_DOLAUNCH} && [[ ${PROFILE_TYPE} != "memray" ]]; then
    # launch the program with the args as bg job
    ${LAUNCHER} -m cellprofiler ${CPARGS} &
    THEPID=$!
fi

# if we didn't pass --pid or --launch, then find it
if ! ${B_DOLAUNCH} && [[ -z "${THEPID}" ]]; then
    # pythonw itself launches python.app's python, which is the real process we want to attach to so filter it out
    #THEPID=$(ps -A | grep -v grep | grep -iE "${PROCNAME}" | grep -vi "vscode" | grep -vi "pythonw" | awk '{print $1}' | head -n 1)
    THEPID=$(pgrep -nif cellprofiler)
    if [[ -z "${THEPID}" ]]; then
        echo "Error: --pid argument missing, and could not find ${PROCNAME} process id automatically."
        usage
        exit 1
    fi
fi

if [[ "${PROFILE_TYPE}" == "svg" ]]; then
    sudo py-spy record -o "${OUTNAME}.svg" --pid "${THEPID}" --subprocesses
elif [[ "${PROFILE_TYPE}" == "speedscope" ]]; then
    sudo py-spy record -o "${OUTNAME}_speedscope.json" --pid "${THEPID}" --subprocesses --format speedscope
elif [[ "${PROFILE_TYPE}" == "memray" ]] && ${B_DOLAUNCH}; then
    # "memray run ... -m <script>" must have -m as last option, as it termiantes the option list
    ${LAUNCHER} -m memray run -o "${OUTNAME}_memray.bin" -m cellprofiler ${CPARGS}
elif [[ "${PROFILE_TYPE}" == "memray" ]] && [[ -n "${THEPID}" ]]; then
    ${LAUNCHER} -m memray attach "${THEPID}" -o "${OUTNAME}_memray.bin"
else
    echo "Error: unknown profile type: ${PROFILE_TYPE}"
    usage
    exit 1
fi
