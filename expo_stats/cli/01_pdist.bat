:: run grid constructor

@echo off
 
set gridsizes=60
set countries=bgd
set maxworkers=4
 
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O main_01_pdist.py --maxworkers %maxworkers% --gridsizes %gridsizes% --countries %countries% 
 
 
::again
 

cmd.exe /k