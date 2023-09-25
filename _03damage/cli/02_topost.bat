:: run grid constructor

@echo off
 
set country=deu
 
 
:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute

python -O main_02_topost.py --country_key %country% 

ECHO finished
cmd.exe /k