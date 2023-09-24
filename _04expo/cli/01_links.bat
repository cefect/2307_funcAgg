:: run grid constructor

@echo off
 
set country=deu
set filter_cent_expo=False
 
 
:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute in parallel
python -O main_01_links.py  

 
ECHO finished
cmd.exe /k