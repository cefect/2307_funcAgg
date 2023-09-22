:: run grid constructor

@echo off
 
set country=deu
 
 
:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute

python -O %~dp0..\da_loss.py

ECHO finished
cmd.exe /k