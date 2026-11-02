@echo off
rem Runs board\tests with the signed interpreter + CoralCapstone site-packages (no pytest needed).
rem On the board itself:  cd board && python3 -m unittest discover -s tests -v
setlocal
if not defined REEFWATCH_PYTHON set "REEFWATCH_PYTHON=C:\Users\nicho\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
if not defined REEFWATCH_SITE_PACKAGES set "REEFWATCH_SITE_PACKAGES=C:\Users\nicho\Projects\CoralCapstone\.venv\Lib\site-packages"
set "PYTHONPATH=%REEFWATCH_SITE_PACKAGES%"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0.."
"%REEFWATCH_PYTHON%" -m unittest discover -s tests -v %*
endlocal & exit /b %ERRORLEVEL%
