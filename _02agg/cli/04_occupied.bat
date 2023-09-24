:: run grid constructor

@echo off
 
set country=deu
set geom_type=poly
 
 

:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute
python -O main_04_occupied.py --geom_type %geom_type%
 
ECHO finished
cmd.exe /k