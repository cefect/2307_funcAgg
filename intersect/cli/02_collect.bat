:: collect samples

SET MAX_WORKERS=4
 

:: Activate environment
call %~dp0..\env\conda_activate

:: execute


python -O %~dp0\intersect_02collect.py --max_workers=%MAX_WORKERS%

cmd.exe /k