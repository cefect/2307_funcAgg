:: run the hazard-building intersect script

SET COUNTRY_KEY=BGD
SET HAZARD_KEYS=500_fluvial 100_fluvial 050_fluvial 010_fluvial
SET MAX_WORKERS=4
 

:: Activate environment
call %~dp0..\env\conda_activate

:: execute


for %%H in (%HAZARD_KEYS%) do (
    ECHO on %%H
    python -O %~dp0\intersect_main.py %COUNTRY_KEY% %%H --max_workers %HAZARD_KEYS%
    ECHO finished %%H
)

 

cmd.exe /k