:: run grid constructor

@echo off
 
set gridsizes=1020 240 60
set countries=zaf bra can deu
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O %~dp0..\_01_pdist.py
 

cmd.exe /k