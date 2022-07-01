@echo off
call _internal\setenv.bat

SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW
echo "CALL TRAIN?"
echo "%PYTHON_EXECUTABLE%"
echo "%DFL_ROOT%"
echo "%DFL_ROOT_SAEHDBW%"
REM Don't forget to put ^ at the end of the line if editing!
SET use_bw_input=1
SET merge_bw_special_imwrite=1
SET default_merge_cpu_count=1
REM SET use_color_input_and
SET extract_to_bw=1
SET apply_reshape_test=1
SET debug_merge_masked=1
REM OVERRIDE PATH
REM CUDA when the GPU device list hangs: Set your data, the max_gpu_memory could be more than what Win10 returns, but it seems it is just "info"
SET force_gpu_data=1
SET max_gpu_memory=1556925644
SET forced_gpu_id=750 Ti
REM SET debug_color_transfer=1

echo "%DFL_ROOT%=" %DFL_ROOT% 

"%PYTHON_EXECUTABLE%" "%DFL_ROOT_SAEHDBW%\main.py" merge ^
    --input-dir "T:\bta"  ^
    --output-dir "T:\bta-TEST-28-6" ^
    --output-mask-dir "T:\bta-TESt-28-6_MASK" ^
    --aligned-dir "T:\bta_aligned"  ^
    --model-dir "C:\Models\saehdbw\Ivan-to-Dragan-500K" ^
    --model SAEHDBW
    
pause