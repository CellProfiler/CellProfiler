cd /Users/mvokes/CellProfiler/trunk/CellProfiler
files = dir('.');

addpath('./CPsubfunctions')
addpath('./DataTools')
addpath('./Help')
addpath('./ImageTools')
addpath('./Modules')

CompileWizard
delete CellProfiler.m
copyfile( 'CompileWizard_CellProfiler.m', 'CellProfiler.m');
delete CompileWizard_CellProfiler.m
mcc -m CellProfiler -a './CPsubfunctions/CPsplash.jpg'

mkdir('.', 'CompiledCellProfiler')
mkdir('./CompiledCellProfiler', 'Modules')
copyfile('Modules/*.txt', './CompiledCellProfiler/Modules')
copyfile('CellProfiler.*','./CompiledCellProfiler/')
delete CellProfiler.m
unix('svn update')