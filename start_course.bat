@echo off
rem Double-click this file to start the course (Windows).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 start_course.py
) else (
    python start_course.py
)
if errorlevel 1 pause
