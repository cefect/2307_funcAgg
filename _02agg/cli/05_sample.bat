:: run grid constructor

@echo off
 
set country=deu
 
set workers=6
set gridsize=1020
 

:: Activate environment
call %~dp0..\..\env\conda_activate
echo on
:: execute
for %%G in (500_fluvial, 100_fluvial, 050_fluvial, 010_fluvial) do (
 
    python -O main_05_sample.py --grid_size %gridsize% --country_key %country% --hazard_key %%G --max_workers %workers%
    ECHO finished %%G
)
ECHO finished %gridsize%

:: next gridsize
set gridsize=240
for %%G in (500_fluvial, 100_fluvial, 050_fluvial, 010_fluvial) do (
 
    python -O main_05_sample.py --grid_size %gridsize% --country_key %country% --hazard_key %%G --max_workers %workers%
    ECHO finished %%G
)
ECHO finished %gridsize%

:: next gridsize
set workers=4
set gridsize=60
for %%G in (500_fluvial, 100_fluvial, 050_fluvial, 010_fluvial) do (
 
    python -O main_05_sample.py --grid_size %gridsize% --country_key %country% --hazard_key %%G --max_workers %workers%
    ECHO finished %%G
)
ECHO finished %gridsize%
 
ECHO finished
cmd.exe /k