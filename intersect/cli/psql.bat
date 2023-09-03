@ECHO off
SET PATH=%PATH%;C:\Program Files\PostgreSQL\14\bin\
ECHO %PATH%

SET HOST=localhost
SET PORT=5432

psql.exe --host=%HOST% --port=%PORT% --username=postgres 