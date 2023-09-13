:: run grid constructor

@echo off
 
set gridsizes=1020 60 240
set countries=bgd deu
set maxworkers=6
set outdir=l:\10_IO\2307_funcAgg\outs\expo_stats\pdist\plots\
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O main_01_pdist.py --maxworkers %maxworkers% --gridsizes %gridsizes% --countries %countries% --outdir %outdir%
 
 
::again
 

cmd.exe /k