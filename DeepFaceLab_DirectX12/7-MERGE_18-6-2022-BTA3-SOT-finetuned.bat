@echo off
call _internal\setenv.bat

REM Don't forget to put ^ at the end of the line if editing!
SET use_bw_input=1
SET merge_bw_special_imwrite=1
SET default_merge_cpu_count=1
REM SET use_color_input_and
SET extract_to_bw=1
SET apply_reshape_test=1
SET debug_merge_masked=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" merge ^
    --input-dir "Z:\bta3_select"  ^
    --output-dir "T:\bta3_select_MERGED_18-6-2022" ^
    --output-mask-dir "Z:\bta3_select_MERGED_18-6-2022_MASK" ^
    --aligned-dir "Z:\bta3_aligned"  ^
    --model-dir "C:\BACKUP\saehdbw\Kiril2Arnold" ^
    --model SAEHDBW
pause
pause
