@echo off
call _internal\setenv.bat
REM OVERRIdE PATH
REM FOR cudA etc. diff vers: SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW_22_5_2022
SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW_22_5_2022
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
SET preview_period=1000
SET user_force_new_preview=1
REM SET print_debug_one_iter_filenames=1
REM SET print_debug_generate_next_samples=1
SET user_dfl_save_interval_min=10
SET font_size_minus_for_iter=2
REM SET default_silent_start=1
REM SET INTERNAL

REM CUDA when the GPU device list hangs: Set your data, the max_gpu_memory could be more than what Win10 returns, but it seems it is just "info"
REM TRY WITHOUT SETTING NOW - maybe fixed by some drivers, windows etc.? #21-2-2023 SET force_gpu_data=1
REM NO - it hangs earlier
SET force_gpu_data=1
SET max_gpu_memory=1556925644
SET forced_gpu_id=750 Ti

"%PYTHON_EXECUTABLE%" "%DFL_ROOT_SAEHDBW%\main.py" train ^
    --training-data-src-dir "C:\DFL\Facesets\Arnold_bw_192_mf" ^
    --training-data-dst-dir "Z:\tosh-192-color_resized"   ^
    --model-dir "Z:\LIAE-UD-192-96-32-32-TOSH-TO-ARNOLD-12-2-2023_mf" ^
    --model SAEHDBW

REM "%PYTHON_EXECUTABLE%" "%DFL_ROOT_SAEHDBW%\main.py" train ^
REM    --training-data-src-dir "C:\DFL\Facesets\Arnold_bw_192_mf" ^
REM    --training-data-dst-dir "C:\DFL\Facesets\kiril_bw_192_mf_BIG_SET"   ^
REM    --model-dir "G:\Biden2Arnold" ^
REM    --model SAEHDBW

@echo off

pause

REM G:\saehdbw - pretraining models
REM "%PYTHON_EXECUTABLE%" "%DFL_ROOT_SAEHDBW%\main.py" train ^
REM    --training-data-src-dir "C:\DFL\Facesets\Arnold_bw_192_mf" ^
REM    --training-data-dst-dir "C:\DFL\Facesets\kiril-aligned-bw-192-mf-18-5-2022"  ^
REM    --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM    --model-dir "G:\biden2arnold" ^
REM    --model SAEHDBW 

pause

REM ...
REM "C:\DFL\Facesets\biden_bw_192"
REM     --training-data-dst-dir "C:\DFL\Facesets\kiril_bw_192_mf" ^
REM data_src --> target face, Kiril Petkov/Arnold
REM dst --> faces of the original output video (Christian Bale, American Psycho)

REM    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
REM    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^


REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM    --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM      --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1"
REM Just fewer images
REM C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_trash

REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
REM     --pretraining-data-dir "%INTERNAL%\pretrain_faces" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM     --model-dir "G:\saehd" ^
REM     --model-dir "%WORKSPACE%\model" ^
    --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
    --model-dir "G:\biden2arnold" ^
    --model SAEHDBW 

pause

REM ...
REM "C:\DFL\Facesets\biden_bw_192"
REM     --training-data-dst-dir "C:\DFL\Facesets\kiril_bw_192_mf" ^
REM data_src --> target face, Kiril Petkov/Arnold
REM dst --> faces of the original output video (Christian Bale, American Psycho)

REM    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
REM    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^


REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM    --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM      --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1"
REM Just fewer images
REM C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_trash

REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384_ne-mik-i-dr\mik-1" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
REM     --pretraining-data-dir "%INTERNAL%\pretrain_faces" ^
REM     --pretraining-data-dir "C:\DFL\DeepFaceLab_DirectX12\_internal\pretrain_faces_x384" ^
REM     --model-dir "G:\saehd" ^
REM     --model-dir "%WORKSPACE%\model" ^