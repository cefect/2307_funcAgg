:: run grid constructor

@echo off
 
 
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O %~dp0..\tools.py
 

cmd.exe /k