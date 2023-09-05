:: run grid constructor

@echo off
 
set gridsizes=1020 240 60
set countries=zaf bra can deu
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O main_01_grid.py --gridsizes %gridsizes% --countries %countries%
 

cmd.exe /k