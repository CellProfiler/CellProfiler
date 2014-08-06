:: jenkins_build.bat
::
:: Batch command for building CellProfiler on a Jenkins Windows node
::
:: CellProfiler is distributed under the GNU General Public License.
:: See the accompanying file LICENSE for details.
:: 
:: Copyright (c) 2003-2009 Massachusetts Institute of Technology
:: Copyright (c) 2009-2014 Broad Institute
:: 
:: Please see the AUTHORS file for credits.
:: 
:: Website: http://www.cellprofiler.org
::
::--------
::
:: The setup assumes that Apache Ant is installed and that the
:: Microsoft Windows SDK is installed
::
::--------
::
:: The root of the network share that contains the projects to build
::
:: PROJECTS_ROOT=x:\projects
::
:: The branch to pull from
::
:: GIT_BRANCH=javabridge-2
::
:: The project to build
::
:: PROJECT_NAME=CellProfiler-javabridge-2
::
:: The path to the ANT executable
::
:: ANT_PATH=C:\Users\Developer\ant\apache-ant-1.9.4\bin\ant
::
:: The path to the vcvars.bat script to run (or other compiler) to
:: set up the compile environment
::
:: VCVARS_BAT=c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars64.bat
::
SETLOCAL
echo off
if not defined PROJECTS_ROOT (
echo "PROJECTS_ROOT is not defined"
end /b -1
) else (
echo "PROJECTS_ROOT=%PROJECTS_ROOT%"
)
if not defined GIT_BRANCH (
echo "GIT_BRANCH is not defined"
end /b -1
) else (
echo "Building GIT branch %GIT_BRANCH%"
)
if not defined PROJECT_NAME (
echo "PROJECT_NAME is not defined"
end /b -1
) else (
echo "Project folder = %PROJECT_NAME%"
)
if not defined VCVARS_BAT (
echo "Please define VCVARS_BAT to point to the vcvars.bat file that sets up the compiler environment"
end /b -1
)
if not defined MVN_PATH (
echo "Please install Apache Maven and define MVN_PATH to point to the executable"
end /b -1
)
if not defined ANT_PATH (
set ANT_PATH=ant
)
%VCVARS_BAT%
pushd %PROJECTS_ROOT%\%PROJECT_NAME%
cd CellProfiler
git pull origin %GIT_BRANCH%
if not errorlevel 0 (
end /b
)
del ..\build.xml
if not errorlevel 0 (
end /b
)
copy jenkins\windows\scripts\build.xml ..
if not errorlevel 0 (
end /b
)
::
:: Find JAVA_HOME using javabridge.locate
::
python -c "from javabridge.locate import find_javahome;print find_javahome()" > my_javahome.txt
set /P JAVA_HOME=<my_javahome.txt
del my_javahome.txt
cd ..
%ANT_PATH% clean
if not errorlevel 0 (
echo "Failed to clean"
end /b
)
%ANT_PATH% compile
if not errorlevel 0 (
echo "Failed to compile"
end /b
)
%ANT_PATH% test
if not errorlevel 0 (
echo "Failed during tests"
end /b
)
%ANT_PATH% windows-build
popd

