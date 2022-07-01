@echo off
call _internal\setenv.bat
echo "CALL TRAIN?"
echo "%PYTHON_EXECUTABLE%"
echo "%DFL_ROOT%"
echo "%DFL_ROO_SAEHDBWT%"
REM only if using color input! SET color_input_and_grayscale_model=1
REM SET user_bw_get_color_channel=2
SET use_bw_input=1
REM SET debug_on_train_one_iter=1
REM SET print_samples_info=0
SET preview_period=1000
REM SET print_debug_one_iter_filenames=1
REM SET print_debug_generate_next_samples=1
SET user_dfl_save_interval_min=10
SET font_size_minus_for_iter=2
REM SET default_silent_start=1
REM 12.5.2022 --> CUDA fixed

REM CUDA when the GPU device list hangs: Set your data, the max_gpu_memory could be more than what Win10 returns, but it seems it is just "info"
SET force_gpu_data=1
SET max_gpu_memory=1556925644
SET forced_gpu_id=750 Ti

SET user_force_new_preview=1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT_SAEHDBW%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "T:\pretrain_faces_x384_resized" ^
    --model-dir "T:\DF-UD-192K-192-48-48-16-103K-it-29-5-2022-31-5-2022-8-6-2022-405K" ^
    --model SAEHDBW

pause