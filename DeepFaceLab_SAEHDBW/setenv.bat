rem ========== BASE ENV ==========
SET INTERNAL=%~dp0
SET INTERNAL=%INTERNAL:~0,-1%
rem overriding windows user/local environment
SET LOCALENV_DIR=%INTERNAL%\_e
SET TMP=%LOCALENV_DIR%\t
SET TEMP=%LOCALENV_DIR%\t
SET HOME=%LOCALENV_DIR%\u
SET HOMEPATH=%LOCALENV_DIR%\u
SET USERPROFILE=%LOCALENV_DIR%\u
SET LOCALAPPDATA=%USERPROFILE%\AppData\Local
SET APPDATA=%USERPROFILE%\AppData\Roaming

rem ========== PYTHON ENV ==========
SET PYTHON_PATH=%INTERNAL%\python-3.6.8
rem overriding default python env vars in order not to interfere with any system python installation
SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET QT_QPA_PLATFORM_PLUGIN_PATH=%PYTHON_LIB_PATH%\PyQt5\Qt\plugins
SET PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

rem ========== CUDA ENV ==========
SET PATH=%INTERNAL%\CUDA;%INTERNAL%\CUDNN;%PATH%
for /f "tokens=4-7 delims=[.] " %%i in ('ver') do (if %%i==Version (set v=%%j.%%k) else (set v=%%i.%%j))
if "%v%" == "10.0" (
    SET "PATH=%INTERNAL%\CUDNN\Win10.0;%PATH%"
    ECHO "%PATH%"
) else (
    SET "PATH=%INTERNAL%\CUDNN\Win6.x;%PATH%"
)

rem ========== ADDITIONAL ENV ==========
SET XNVIEWMP_PATH=%INTERNAL%\XnViewMP
SET FFMPEG_PATH=%INTERNAL%\ffmpeg
SET PATH=%XNVIEWMP_PATH%;%FFMPEG_PATH%;%PATH%
SET WORKSPACE=%INTERNAL%\..\workspace
SET DFL_ROOT=%INTERNAL%\DeepFaceLab
SET TF_MIN_REQ_CAP=30
SET DFL_ROOT_SAEHDBW=%INTERNAL%\DeepFaceLab_SAEHDBW
REM 10-5-2022
