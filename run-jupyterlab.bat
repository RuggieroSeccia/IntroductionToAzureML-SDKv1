@echo off
set JUPYTER_PATH=.jupyter
CALL conda activate ./.venv
CALL jupyter-lab
pause
