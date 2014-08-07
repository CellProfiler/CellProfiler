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
:: The Jenkins workspace for the project. The GIT clone should be to
::     WORKSPACE\CellProfiler.
::
:: WORKSPACE=<jenkins-assigned workspace>
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
if not defined WORKSPACE (
echo WORKSPACE is not defined
exit /b -1
) else (
echo "PROJECT_ROOT=%WORKSPACE%"
)
if not defined GIT_BRANCH (
echo "GIT_BRANCH is not defined"
exit /b -1
) else (
echo "Building GIT branch %GIT_BRANCH%"
)
if not defined VCVARS_BAT (
echo "Please define VCVARS_BAT to point to the vcvars.bat file that sets up the compiler environment"
end /b -1
)
if not defined MVN_PATH (
echo "Please install Apache Maven and define MVN_PATH to point to the executable"
exit /b -1
)
if not defined ANT_PATH (
set ANT_PATH=ant
)
call "%VCVARS_BAT%"
if defined VCINSTALLDIR (
echo Using compiler at "%VCINSTALLDIR%"
)
if exist %WORKSPACE%\build.xml (
echo Deleting old build.xml
del %WORKSPACE%\build.xml
if not errorlevel 0 (
exit /b
))
set BUILD_XML=%WORKSPACE%\CellProfiler\jenkins\windows\scripts\build.xml
if not exist "%BUILD_XML%" (
echo "Missing build.xml from CellProfiler git clone at %BUILD_XML%"
exit /b
) else (
echo Copying build.xml from %BUILD_XML%
)
copy "%WORKSPACE%\CellProfiler\jenkins\windows\scripts\build.xml" "%WORKSPACE%\build.xml"
if not errorlevel 0 (
exit /b
)
::
:: Find JAVA_HOME using javabridge.locate
::
python -c "from javabridge.locate import find_javahome;open('%mjh%','w').write(find_javahome)"
set /P JAVA_HOME=<"%mjh%"
del %mjh%
echo JAVA_HOME=%JAVA_HOME%
echo Changing directory to %WORKSPACE%
pushd %WORKSPACE%
echo Cleaning derived files (.pyd, .jar etc)
call "%ANT_PATH%" clean
if not errorlevel 0 (
echo "Failed to clean"
exit /b
)
echo Compiling Cython and .c files
call "%ANT_PATH%" compile
if not errorlevel 0 (
echo "Failed to compile"
exit /b
)
echo Running tests
call "%ANT_PATH%" test
if not errorlevel 0 (
echo "Failed during tests"
exit /b
)
echo Building Windows .msi file
call "%ANT_PATH%" windows-build
popd
endlocal

