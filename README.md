# FastSceneSub

Windows 实时场景字幕。采集默认扬声器/耳机的 WASAPI 环回，本地用 Silero VAD（ONNX CPU）切句，再通过同一个 OpenRouter API Key 做语音识别和中文翻译。

不需要虚拟声卡，不需要 ffmpeg，也不安装 PyTorch。

## 启动

1. 安装 [uv](https://docs.astral.sh/uv/)。
2. 双击 `run.bat`（内部是 `uv run python main.py`）。
3. 点「设置」，填写 OpenRouter API Key。代理可留空；需要时填 `http://127.0.0.1:7890` 或 `socks5://127.0.0.1:7890`。
4. 看片前点选场景，再点「开始监听」。第一次监听会把约 2MB 的 `silero_vad.onnx` 下载到 `assets/`。

音频来自 Windows 当前默认输出设备，音箱和耳机都可以。

## 免安装

从 [Releases](https://github.com/n1ght-ow/FastSceneSub/releases) 下载 `FastSceneSub.exe`，放进单独的文件夹后双击。`config.json` 和 `assets` 写在 exe 旁边，API Key 不会打进程序。第一次监听会把随包的 `silero_vad.onnx` 复制到旁边的 `assets`；复制失败才联网下载。

## 自己打包

先用 `run.bat` 装好依赖，再双击 `build_exe.bat`。产物是 `dist\FastSceneSub.exe`。`build/`、`dist/` 和 spec 不进入 git，exe 只挂在 GitHub Release 上。

## 操作

- 四个场景会立刻切换 VAD、识别语言和翻译模型。切换场景或点「新片重置」会清空短期对白和后台剧情/赛况锚点。
- 「锁定字幕穿透」后，鼠标可以点穿底部字幕去操作播放器。顶栏不穿透。
- `Alt+H` 隐藏全部界面并暂停监听，再按一次恢复。
- 顶栏右侧的 × 退出。耗时灯格式是 `ASR: xxx ms | 翻译首字: xxx ms`。

`config.json` 里有 API Key，已在 `.gitignore` 中忽略。