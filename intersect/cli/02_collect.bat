:: collect samples

 
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute


python -O %~dp0\intersect_02collect.py 

cmd.exe /k