:: run grid constructor

@echo off
 
set gridsizes=1020 240 60
set countries=bgd
set maxworkers=6
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O main_01_pdist.py --maxworkers %maxworkers% --gridsizes %gridsizes% --countries %countries%
 
 
::again
set countries=deu
python -O main_01_pdist.py --maxworkers %maxworkers% --gridsizes %gridsizes% --countries %countries%

cmd.exe /k