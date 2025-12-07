Docker image: `linuxserver/ffmpeg:8.0.1` - **IMPORTANT**, `ffmpeg` - уже прописан в ENTRYPOINT


`RX 780M`, `Mesa 25.2.7`

`Kernel: 6.17.9`


Тестовый файл (H264):

`/home/stfu/Desktop/test-data.mp4`



# Пример команды с декодом на Vulkan + Энкод через VAAPI в av1

```sh
docker run --rm \
  --device=/dev/dri:/dev/dri \
  -e RADV_PERFTEST=video_decode \
  -v /home/stfu/Desktop:/data \
  linuxserver/ffmpeg:8.0.1 \
  -init_hw_device "vulkan=vk:0" \
  -filter_hw_device vk \
  -hwaccel vulkan \
  -hwaccel_output_format vulkan \
  -i /data/test-data.mp4 \
  -c:v av1_vulkan \
  -b:v 2000k \
  -c:a copy \
  -y /data/output-av1.mkv
```


# Vulkan support

Vulkan support has been added to x86_64 (tested with Intel and AMD iGPU)

```sh
docker run --rm -it \
  --device=/dev/dri:/dev/dri \
  -v $(pwd):/config \
  -e ANV_VIDEO_DECODE=1 \
  linuxserver/ffmpeg \
  -init_hw_device "vulkan=vk:0" \
  -hwaccel vulkan \
  -hwaccel_output_format vulkan \
  -i /config/input.mkv \
  -f null - -benchmark
```

Vulkan supports three drivers

    ANV: To enable for Intel, set the env var ANV_VIDEO_DECODE=1
    RADV: To enable on AMD, set the env var RADV_PERFTEST=video_decode
    NVIDIA: To enable on Nvidia, install Nvidia Vulkan Beta drivers on the host per this article⁠

## Ручной дебаг: оценка битрейта + ручной запуск в Docker

### 1) Задать вход

```sh
SRC="/home/stfu/Desktop/test-data.mp4"   # ваш файл
SRC_DIR=$(dirname "$SRC")
SRC_NAME=$(basename "$SRC")
```

### 2) ffprobe через Docker (сохраняем JSON)

```sh
docker run --rm \
  --device=/dev/dri:/dev/dri \
  -e RADV_PERFTEST=video_decode \
  -v "$SRC_DIR":/data \
  --entrypoint ffprobe \
  linuxserver/ffmpeg:8.0.1 \
  -v quiet -print_format json -show_format -show_streams \
  "/data/$SRC_NAME" \
  > /tmp/ffprobe-av1.json

jq '.format|{bit_rate,duration,size}' /tmp/ffprobe-av1.json
jq '.streams[]|select(.codec_type=="video")|{codec_name,width,height,r_frame_rate,bit_rate}' /tmp/ffprobe-av1.json
```

### 3) Расчёт таргетного битрейта по формуле скрипта

```sh
python3 - <<'PY'
import json, math, pathlib
info = json.loads(pathlib.Path("/tmp/ffprobe-av1.json").read_text())
fmt = info["format"]
v = next(s for s in info["streams"] if s["codec_type"] == "video")
source_br = int((v.get("bit_rate") or fmt.get("bit_rate") or 0) // 1000)
res = int(v["width"]) * int(v["height"])
alpha, scale = 0.72, 6.0
target = int(scale * (max(source_br, 1) ** alpha))
if res >= 3840*2160: min_br, max_r = 8000, 25000
elif res >= 2560*1440: min_br, max_r = 4000, 15000
elif res >= 1920*1080: min_br, max_r = 2000, 8000
elif res >= 1280*720: min_br, max_r = 1000, 5000
else: min_br, max_r = 500, 3000
target = max(min_br, min(target, max_r))
target = min(target, source_br)
maxrate = int(target * 1.6)
bufsize = target * 2
fps = v.get("r_frame_rate", "30/1")
num, den = (fps.split("/") + ["1"])[:2]
fps_val = float(num) / float(den)
gop = max(1, round(fps_val * 10))
keyint_min = max(1, round(fps_val))
print(f"source_bitrate={source_br} kbps")
print(f"target_bitrate={target} kbps")
print(f"maxrate={maxrate} kbps")
print(f"bufsize={bufsize} kbps")
print(f"gop={gop}, keyint_min={keyint_min}")
PY
```

Запомните цифры `target_bitrate`, `maxrate`, `bufsize`, `gop`, `keyint_min`.

### 4) Ручной запуск ffmpeg (Vulkan decode + Vulkan encode)

```sh
TARGET=2000    # подставьте из шага 3 (kbps)
MAXRATE=3200   # из шага 3 (kbps)
BUFSIZE=4000   # из шага 3 (kbps)
GOP=240        # из шага 3
KEYINT_MIN=24  # из шага 3
OUT_NAME="test-data-av1.mkv"

docker run --rm \
  --device=/dev/dri:/dev/dri \
  -e RADV_PERFTEST=video_decode \
  -v "$SRC_DIR":/data \
  linuxserver/ffmpeg:8.0.1 \
  -hide_banner \
  -init_hw_device vulkan=vk:0 \
  -filter_hw_device vk \
  -hwaccel vulkan \
  -hwaccel_output_format vulkan \
  -i "/data/$SRC_NAME" \
  -map 0 -map -0:d -map_metadata 0 -map_chapters 0 \
  -c:v av1_vulkan \
  -b:v "${TARGET}k" -maxrate "${MAXRATE}k" -bufsize "${BUFSIZE}k" \
  -g "$GOP" -keyint_min "$KEYINT_MIN" \
  -c:a copy -c:s copy -c:t copy \
  -progress pipe:1 \
  -y "/data/$OUT_NAME"
```

> Для Intel замените `RADV_PERFTEST=video_decode` на `ANV_VIDEO_DECODE=1`. Если нужен VAAPI-энкодер, поменяйте блок с `-c:v av1_vulkan` на VAAPI-конфиг из начала файла.