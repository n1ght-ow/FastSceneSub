@echo off
chcp 65001 >nul
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo 未找到 uv，请先安装 https://docs.astral.sh/uv/
  pause
  exit /b 1
)
uv run python main.py
if errorlevel 1 pause
