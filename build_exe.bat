@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 缺少 .venv。请先双击 run.bat，或在此目录执行 uv sync。
  pause
  exit /b 1
)
if not exist "assets\silero_vad.onnx" (
  echo 缺少 assets\silero_vad.onnx。请先运行一次程序，让它下载 VAD 模型。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check pyinstaller
if errorlevel 1 (
  echo 安装 PyInstaller 失败
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --name FastSceneSub --onefile ^
  --add-data "assets\silero_vad.onnx;assets" ^
  --collect-all pyaudiowpatch ^
  --collect-all soundfile ^
  --collect-all onnxruntime ^
  --collect-all soxr ^
  --hidden-import pyaudiowpatch ^
  --hidden-import soxr ^
  --hidden-import h2 ^
  --hidden-import socksio ^
  main.py
if errorlevel 1 (
  echo 打包失败
  pause
  exit /b 1
)
echo 完成: dist\FastSceneSub.exe
