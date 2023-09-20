:: run the hazard-building intersect script

SET COUNTRY_KEY=DEU
SET HAZARD_KEYS=100_fluvial 050_fluvial 010_fluvial
SET MAX_WORKERS=4
 

:: Activate environment
call %~dp0..\..\env\conda_activate

:: execute


for %%H in (%HAZARD_KEYS%) do (
    ECHO on %%H
    python -O %~dp0\intersect_01sample.py %COUNTRY_KEY% %%H --max_workers %MAX_WORKERS%
    ECHO finished %%H
)

 

cmd.exe /k