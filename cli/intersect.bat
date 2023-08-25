:: run the hazard-building intersect script

SET COUNTRY_KEY=BGD
SET HAZARD_KEY=500_fluvial

:: Activate environment
call %~dp0..\env\conda_activate

:: execute
python -O %~dp0\intersect_main.py %COUNTRY_KEY% %HAZARD_KEY%

cmd.exe /k