:: run the hazard-building intersect script

 
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute
python -O %~dp0..\_01_grids.py
 

cmd.exe /k