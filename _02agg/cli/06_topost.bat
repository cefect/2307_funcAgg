:: run grid constructor

@echo off
 
set country=deu
 
 
set gridsize=60
 

:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute
python -O main_06_topost.py --grid_size %gridsize% --country_key %country% 
 
ECHO finished
cmd.exe /k