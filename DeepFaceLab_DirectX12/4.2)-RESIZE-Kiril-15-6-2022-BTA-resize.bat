@echo off
call _internal\setenv.bat
REM SET use_bw_input=1
REM SET resize_color_to_grayscale=1
REM SET only_convert_to_grayscale=1
SET resize_color_to_grayscale=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool resize ^
  --input-dir "T:\bta_aligned"

pause