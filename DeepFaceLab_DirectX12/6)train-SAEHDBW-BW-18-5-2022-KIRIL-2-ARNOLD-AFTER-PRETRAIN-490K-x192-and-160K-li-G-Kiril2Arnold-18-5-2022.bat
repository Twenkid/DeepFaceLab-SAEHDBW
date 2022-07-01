@echo off
call _internal\setenv.bat
REM OVERRIdE PATH
REM FOR cudA etc. diff vers: SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW_22_5_2022
echo "CALL TRAIN?"
echo "%PYTHON_EXECUTABLE%"
echo "%DFL_ROOT%"
echo "%DFL_ROOT_SAEHDBW%"
pause
REM only if using color input! SET color_input_and_grayscale_model=1
REM SET user_bw_get_color_channel=2
SET use_bw_input=1
REM SET debug_on_train_one_iter=1
SET print_samples_info=1
SET preview_period=400
SET user_force_new_preview=1
REM SET print_debug_one_iter_filenames=1
REM SET print_debug_generate_next_samples=1
SET user_dfl_save_interval_min=10
SET font_size_minus_for_iter=2
REM SET default_silent_start=1
REM SET INTERNAL

REM CUDA when the GPU device list hangs: Set your data, the max_gpu_memory could be more than what Win10 returns, but it seems it is just "info"
REM SET force_gpu_data=1
REM SET max_gpu_memory=1556925644
REM SET forced_gpu_id=750 Ti

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "C:\DFL\Facesets\Arnold_bw_192_mf" ^
    --training-data-dst-dir "C:\DFL\Facesets\kiril_bw_192_mf_BIG_SET" ^
    --pretraining-data-dir "%INTERNAL%\pretrain_faces_x384_trash" ^
    --model-dir "G:\Biden2Arnold" ^
    --model SAEHDBW

@echo off

pause