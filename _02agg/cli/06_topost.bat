:: run grid constructor

@echo off
 
set country=deu
 
 
 
 

:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute
for %%G in (1020, 240, 60) do (
    python -O main_06_topost.py --grid_size %%G --country_key %country%
)    
 
ECHO finished
cmd.exe /k