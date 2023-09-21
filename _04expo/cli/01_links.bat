:: run grid constructor

@echo off
 
set country=deu
 
 
:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute in parallel

for %%G in (1020, 240, 60) do (
    start cmd /k python -O main_01_links.py --grid_size %%G --country_key %country% 
)
 
ECHO finished
cmd.exe /k