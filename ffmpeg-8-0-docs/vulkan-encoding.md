# Аппаратное ускорение FFmpeg на AMD под Linux

## Кратко
- AMD GPU (UVD/VCE/VCN) под Linux дают аппаратное декодирование и кодирование через открытые API: `VAAPI`, `VDPAU` и экспериментальный `Vulkan Video`; кодирование AMF доступно, но требует наличия AMF SDK/драйвера.
- Аппаратные кодеки обычно ограничены 8‑битным 4:2:0 и дают качество ниже хороших софт-энкодеров при том же битрейте, но экономят CPU/энергию.
- Для работы с аппаратными поверхностями (без копий в RAM) используйте совместимые фильтры (`*_vaapi`, `scale_cuda`/`scale_vaapi` и т.д.) или явные `hwupload` / `hwdownload`.

## Доступные API для AMD на Linux
- **VAAPI** – декодирование и кодирование (H.264, HEVC, AV1 при наличии поддержки драйвера). Рекомендуемый путь для транскодинга без копий.
- **VDPAU** – декодирование (H.264, MPEG-1/2/4, VC-1, AV1). Только вывод на GPU, возврат в RAM не поддержан без дополнительной логики.
- **Vulkan Video (экспериментально)** – декодирование H.264/HEVC/AV1 через Mesa; интерфейс может меняться.
- **AMF (Advanced Media Framework)** – кодирование H.264/HEVC/AV1. Нужны драйверы/SDK AMD; в FFmpeg энкодеры имеют постфикс `_amf`.

## Проверка доступности
- Список аппаратных ускорителей в сборке:  
  `ffmpeg -hwaccels`
- Проверить устройства VAAPI: наличие `/dev/dri/renderD*`; типичный путь `/dev/dri/renderD128`.

## Декодирование
- VAAPI (рекомендовано):
  - Бенчмарк декода в нулевой вывод:  
    `ffmpeg -init_hw_device vaapi=va:/dev/dri/renderD128 -hwaccel vaapi -hwaccel_output_format vaapi -i INPUT -f null - -benchmark`
- VDPAU:
  - `ffmpeg -hwaccel vdpau -i INPUT -f null - -benchmark`
  - Помните: VDPAU выводит кадры на GPU; возврат в RAM требует дополнительного кода.
- Vulkan (экспериментально):
  - `ffmpeg -init_hw_device "vulkan=vk:0" -hwaccel vulkan -hwaccel_output_format vulkan -i INPUT -f null - -benchmark`

## Кодирование (AMD)
- AMF энкодеры: `h264_amf`, `hevc_amf`, `av1_amf`  
  Пример: `ffmpeg -s 1920x1080 -pix_fmt yuv420p -i input.yuv -c:v h264_amf output.mp4`
- VAAPI энкодеры: `h264_vaapi`, `hevc_vaapi`, `av1_vaapi` (если драйвер поддерживает).  
  Минимальный пример без копий:  
  `ffmpeg -init_hw_device vaapi=va:/dev/dri/renderD128 -hwaccel vaapi -hwaccel_output_format vaapi -i INPUT -c:v h264_vaapi output.mp4`

## Транскодирование без копий (VAAPI)
- Декод → энкод:  
  `ffmpeg -init_hw_device vaapi=va:/dev/dri/renderD128 -hwaccel vaapi -hwaccel_output_format vaapi -i INPUT -c:v hevc_vaapi output.mp4`
- Со скейлом на GPU:  
  `ffmpeg -init_hw_device vaapi=va:/dev/dri/renderD128 -hwaccel vaapi -hwaccel_output_format vaapi -i INPUT -vf scale_vaapi=1280:720 -c:v h264_vaapi output.mp4`
- Если нужно фильтровать на CPU, кадры придется скачать: добавьте `hwdownload,format=yuv420p` перед CPU-фильтрами.

## Ограничения и советы
- Аппаратные декодеры/энкодеры часто не покрывают все профили/форматы; при несоответствии профиль будет отклонен (надо проверять ошибки запуска).
- Качество аппаратного кодирования обычно ниже софт-энкодеров (x264/x265) при том же битрейте. Для максимального качества оставьте софт-энкодер или повышайте битрейт.
- Для AV1 на AMD: проверяйте поддержку драйвером (Mesa/AMF). На старых GPU AV1 может отсутствовать.
- При наличии нескольких GPU выберите устройство: `-hwaccel_device N` (номер DRM или индекс Vulkan).
