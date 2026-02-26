@echo off
REM ===================================================================
REM   PlumeDEBuG Environment Setup Script (Windows)
REM   Filename: setup.bat
REM ===================================================================

REM 1. Figure out where this script lives (i.e. your project root)
SET SCRIPT_DIR=%~dp0
SET ENV_DIR=%SCRIPT_DIR%environment

REM 2. Check for conda in PATH
where conda >nul 2>nul
IF ERRORLEVEL 1 (
    echo.
    echo [Warning] 'conda' not found. Skipping Conda environment steps.
    goto pip_only
) ELSE (
    echo.
    echo -------- 1. Create or update Conda environment --------
    REM Try to create; if it fails (env already exists), update instead
    conda env create -f "%ENV_DIR%\environment.yml" --force 2>nul || (
        echo Environment already exists. Updating instead...
        conda env update -n PlumeDEBuG -f "%ENV_DIR%\environment.yml"
    )
    echo.

    echo -------- 2. Activate Conda environment --------
    call conda activate PlumeDEBuG
    echo.
)

: pip_only
echo -------- 3. Install pip requirements --------
pip install -r "%ENV_DIR%\requirements.txt"
echo.

echo ======== Environment setup complete ========
pause
