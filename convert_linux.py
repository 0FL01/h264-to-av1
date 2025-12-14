#!/usr/bin/env python3
"""
H264 to AV1 Converter with Vulkan Hardware Decode + Encode (Docker Runtime)
Автоматическая конвертация видео с аппаратным ускорением Vulkan через Docker контейнер.
"""

import os
import sys
import signal
import subprocess
import shutil
import json
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from collections import deque

# Docker image для ffmpeg
DOCKER_IMAGE = "linuxserver/ffmpeg:8.0.1"
WORTHINESS_THRESHOLD = 0.95

# Кодеки, поддерживаемые Vulkan hwaccel для декодирования (AMD RADV)
VULKAN_DECODE_CODECS = {'h264', 'hevc', 'h265', 'av1', 'vp9'}

# Аудио кодеки, несовместимые с MKV контейнером в режиме copy
# (требуют транскодирования в AAC/Opus)
INCOMPATIBLE_AUDIO_CODECS = {'pcm_mulaw', 'pcm_alaw', 'pcm_s16le', 'pcm_s16be', 'pcm_u8'}

# ANSI цвета для красивого вывода
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


@dataclass
class VideoInfo:
    """Информация о видеофайле"""
    path: Path
    bitrate: int  # в kbps
    duration: float  # в секундах
    codec: str
    width: int
    height: int
    fps: float
    size_bytes: int
    audio_codec: Optional[str] = None  # кодек первого аудиопотока


@dataclass
class ConversionResult:
    """Результат конвертации"""
    success: bool
    source_size: int
    output_size: int
    message: str


class ConversionState(Enum):
    """Состояние конвертации"""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# Глобальные переменные для обработки прерываний
current_temp_file: Optional[Path] = None
current_process: Optional[subprocess.Popen] = None
conversion_state = ConversionState.IDLE


def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    global conversion_state, current_process, current_temp_file
    
    print(f"\n{Colors.YELLOW}⚠ Получен сигнал прерывания. Выполняется очистка...{Colors.RESET}")
    
    conversion_state = ConversionState.CANCELLED
    
    # Завершаем текущий процесс docker/ffmpeg
    if current_process and current_process.poll() is None:
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            current_process.kill()
    
    # Удаляем временный файл
    cleanup_temp_file()
    
    print(f"{Colors.GREEN}✓ Очистка завершена. Исходные файлы не затронуты.{Colors.RESET}")
    sys.exit(0)


def cleanup_temp_file():
    """Удаление временного файла"""
    global current_temp_file
    if current_temp_file and current_temp_file.exists():
        try:
            current_temp_file.unlink()
            print(f"{Colors.DIM}  Удалён временный файл: {current_temp_file.name}{Colors.RESET}")
        except OSError as e:
            print(f"{Colors.RED}  Ошибка удаления временного файла: {e}{Colors.RESET}")
        current_temp_file = None


def print_banner():
    """Вывод красивого баннера"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}H264 → AV1 Converter{Colors.RESET}{Colors.CYAN}                                        ║
║  {Colors.DIM}Vulkan Hardware Accelerated (Docker){Colors.RESET}{Colors.CYAN}                        ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


class DockerFFmpegRunner:
    """Обёртка для запуска ffmpeg/ffprobe через Docker с Vulkan ускорением"""
    
    def __init__(self, image: str = DOCKER_IMAGE):
        self.image = image
    
    def _get_docker_base_cmd(self) -> list[str]:
        """Базовая команда Docker с пробросом устройств для Vulkan"""
        return [
            'docker', 'run', '--rm',
            '--device=/dev/dri:/dev/dri',  # Vulkan требует полный /dev/dri
            '-e', 'RADV_PERFTEST=video_decode',  # AMD Vulkan decode
        ]
    
    def _map_path_to_container(self, host_path: Path) -> tuple[str, str]:
        """
        Маппинг пути хоста в контейнер.
        Возвращает: (volume_mount, container_path)
        """
        abs_path = host_path.resolve()
        parent_dir = abs_path.parent
        filename = abs_path.name
        container_dir = f"/data/{hash(str(parent_dir)) % 10000}"
        return f"{parent_dir}:{container_dir}", f"{container_dir}/{filename}"
    
    def run_ffprobe(self, file_path: Path) -> subprocess.CompletedProcess:
        """Запуск ffprobe через Docker"""
        volume_mount, container_path = self._map_path_to_container(file_path)
        
        cmd = self._get_docker_base_cmd() + [
            '-v', volume_mount,
            '--entrypoint', 'ffprobe',
            self.image,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            container_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # ffprobe может вернуть байты вне UTF-8
                check=False
            )
            return result
        except FileNotFoundError:
            print(f"{Colors.RED}Ошибка: Docker не найден. Установите Docker.{Colors.RESET}")
            sys.exit(1)
    
    def build_ffmpeg_cmd(
        self,
        source_path: Path,
        output_path: Path,
        ffmpeg_args: list[str]
    ) -> tuple[list[str], str, str]:
        """
        Формирует команду Docker run для ffmpeg.
        Возвращает: (полная команда, container_input_path, container_output_path)
        """
        source_mount, container_source = self._map_path_to_container(source_path)
        output_mount, container_output = self._map_path_to_container(output_path)
        
        # Собираем уникальные маунты
        mounts = [source_mount]
        if output_mount not in mounts:
            mounts.append(output_mount)
        
        cmd = self._get_docker_base_cmd()
        for mount in mounts:
            cmd.extend(['-v', mount])
        
        cmd.append(self.image)
        # ENTRYPOINT уже ffmpeg, добавляем аргументы
        cmd.extend(ffmpeg_args)
        
        return cmd, container_source, container_output


# Глобальный экземпляр Docker runner
docker_runner = DockerFFmpegRunner()


def get_video_info(file_path: Path) -> Optional[VideoInfo]:
    """Получение информации о видеофайле через ffprobe (Docker)"""
    result = docker_runner.run_ffprobe(file_path)
    
    if result.returncode != 0:
        return None
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    
    # Находим видеопоток и аудиопоток
    video_stream = None
    audio_stream = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video' and video_stream is None:
            video_stream = stream
        elif stream.get('codec_type') == 'audio' and audio_stream is None:
            audio_stream = stream
    
    if not video_stream:
        return None
    
    format_info = data.get('format', {})
    
    # Получаем битрейт (может быть в разных местах)
    bitrate = 0
    if 'bit_rate' in video_stream:
        bitrate = int(video_stream['bit_rate']) // 1000
    elif 'bit_rate' in format_info:
        bitrate = int(format_info['bit_rate']) // 1000
    
    # Если битрейт не найден, вычисляем из размера файла и длительности
    if bitrate == 0:
        duration = float(format_info.get('duration', 0))
        size = int(format_info.get('size', 0))
        if duration > 0 and size > 0:
            bitrate = int((size * 8) / duration / 1000)
    
    # Получаем FPS
    fps = 30.0
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        if int(den) > 0:
            fps = float(num) / float(den)
    
    # Получаем кодек аудио (если есть аудиопоток)
    audio_codec = audio_stream.get('codec_name') if audio_stream else None
    
    return VideoInfo(
        path=file_path,
        bitrate=bitrate,
        duration=float(format_info.get('duration', 0)),
        codec=video_stream.get('codec_name', 'unknown'),
        width=int(video_stream.get('width', 0)),
        height=int(video_stream.get('height', 0)),
        fps=fps,
        size_bytes=int(format_info.get('size', 0)),
        audio_codec=audio_codec
    )


def calculate_av1_bitrate(source_info: VideoInfo) -> tuple[int, int, int]:
    """
    Расчёт оптимального битрейта для AV1 с нелинейной кривой.
    Возвращает: (target_bitrate, max_rate, buf_size) в kbps.
    """
    # Нелинейная (логарифмическая) кривая сжатия:
    # при высоких битрейтах агрессивнее урезаем, при низких — бережнее.
    source_bitrate = max(source_info.bitrate, 1)
    alpha = 0.72    # крутизна кривой: <1 сжимает сильнее на верхах, мягче на низах
    scale = 6.0     # коэффициент масштаба (подгонка под целевые уровни качества)
    target_bitrate = int(scale * (math.pow(source_bitrate, alpha)))
    
    # Минимальные пороги качества в зависимости от разрешения
    resolution = source_info.width * source_info.height
    
    if resolution >= 3840 * 2160:  # 4K
        min_bitrate = 8000
        max_reasonable = 25000
    elif resolution >= 2560 * 1440:  # 1440p
        min_bitrate = 4000
        max_reasonable = 15000
    elif resolution >= 1920 * 1080:  # 1080p
        min_bitrate = 2000
        max_reasonable = 8000
    elif resolution >= 1280 * 720:  # 720p
        min_bitrate = 1000
        max_reasonable = 5000
    else:  # SD и ниже
        min_bitrate = 500
        max_reasonable = 3000
    
    # Применяем ограничения
    target_bitrate = max(min_bitrate, min(target_bitrate, max_reasonable))
    target_bitrate = min(target_bitrate, source_info.bitrate)
    
    # Максимальный битрейт для VBR (пиковые моменты)
    max_rate = int(target_bitrate * 1.6)
    
    # Размер буфера (обычно 2x от target)
    buf_size = target_bitrate * 2
    
    return target_bitrate, max_rate, buf_size


def calculate_gop_params(source_info: VideoInfo) -> tuple[int, int]:
    """
    Адаптивный GOP: ~10 секунд (FPS * 10) и минимальный интервал ключевых кадров ~1 секунда.
    Возвращает: (gop_size, keyint_min).
    """
    fallback_fps = 24.0
    fps = source_info.fps if source_info.fps > 0 else fallback_fps
    
    gop_size = max(1, int(round(fps * 10)))  # 10 секунд для удобной перемотки без потери эффективности
    keyint_min = max(1, int(round(fps)))     # минимум один ключевой кадр в секунду
    
    return gop_size, keyint_min


def format_size(size_bytes: int) -> str:
    """Форматирование размера файла"""
    for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} ТБ"


def format_duration(seconds: float) -> str:
    """Форматирование длительности"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    elif minutes > 0:
        return f"{minutes}м {secs}с"
    else:
        return f"{secs}с"


def format_eta(seconds: float) -> str:
    """Форматирование ETA в компактный вид HH:MM:SS или MM:SS"""
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def print_video_info(info: VideoInfo):
    """Вывод информации о видео"""
    print(f"\n{Colors.BLUE}📹 Информация о видео:{Colors.RESET}")
    print(f"   Файл: {Colors.BOLD}{info.path.name}{Colors.RESET}")
    print(f"   Кодек: {info.codec}")
    print(f"   Разрешение: {info.width}x{info.height}")
    print(f"   FPS: {info.fps:.2f}")
    print(f"   Битрейт: {info.bitrate} kbps")
    print(f"   Длительность: {format_duration(info.duration)}")
    print(f"   Размер: {format_size(info.size_bytes)}")


def print_conversion_params(target_br: int, max_br: int, buf_size: int, gop_size: int, keyint_min: int):
    """Вывод параметров конвертации"""
    print(f"\n{Colors.CYAN}⚙ Параметры AV1 кодирования:{Colors.RESET}")
    print(f"   Целевой битрейт: {target_br} kbps")
    print(f"   Максимальный: {max_br} kbps")
    print(f"   Буфер: {buf_size} kbps")
    print(f"   GOP: {gop_size} кадров (≈10 сек)")
    print(f"   keyint_min: {keyint_min} кадров (≈1 сек)")


def convert_video(source_path: Path, output_path: Path, video_info: VideoInfo) -> ConversionResult:
    """
    Конвертация видео H264 → AV1 с использованием VAAPI через Docker.
    """
    global current_temp_file, current_process, conversion_state
    # Храним хвост stdout/stderr, чтобы не блокировать ffmpeg и показать ошибку при сбое
    log_tail = deque(maxlen=200)
    
    # Вычисляем параметры кодирования
    target_br, max_br, buf_size = calculate_av1_bitrate(video_info)
    gop_size, keyint_min = calculate_gop_params(video_info)
    print_conversion_params(target_br, max_br, buf_size, gop_size, keyint_min)
    
    # Создаём временный файл (атомарность)
    temp_path = output_path.with_suffix('.tmp' + output_path.suffix)
    current_temp_file = temp_path
    conversion_state = ConversionState.IN_PROGRESS
    
    # Определяем кодек субтитров в зависимости от входного контейнера:
    # - MP4/M4V/MOV используют mov_text (tx3g), который НЕ поддерживается MKV
    # - MKV уже содержит совместимые форматы (srt, ass, pgs)
    source_ext = source_path.suffix.lower()
    if source_ext in ['.mp4', '.m4v', '.mov']:
        subtitle_codec = 'srt'  # конвертируем mov_text → srt
    else:
        subtitle_codec = 'copy'  # MKV субтитры копируем как есть
    
    # Определяем, поддерживает ли Vulkan hwaccel декодирование этого кодека
    use_vulkan_decode = video_info.codec.lower() in VULKAN_DECODE_CODECS
    
    # Формируем аргументы ffmpeg (без самого ffmpeg - он в ENTRYPOINT)
    ffmpeg_args = ['-hide_banner']
    
    if use_vulkan_decode:
        # Полный Vulkan pipeline: decode + encode на GPU
        ffmpeg_args.extend([
            '-init_hw_device', 'vulkan=vk:0',
            '-filter_hw_device', 'vk',
            '-hwaccel', 'vulkan',
            '-hwaccel_output_format', 'vulkan',
        ])
        video_filter = None
        pipeline_name = "Vulkan decode + encode"
    else:
        # Гибридный pipeline: software decode → hwupload → Vulkan encode
        # Для кодеков типа MJPEG, которые Vulkan не умеет декодировать
        ffmpeg_args.extend([
            '-init_hw_device', 'vulkan=vk:0',
            '-filter_hw_device', 'vk',
        ])
        video_filter = 'format=nv12,hwupload'
        pipeline_name = f"Software decode ({video_info.codec}) + Vulkan encode"
    
    # Путь к файлу будет подставлен Docker runner
    # Здесь placeholder, который заменится
    ffmpeg_args.extend(['-i', '__INPUT__'])
    
    # Явно маппим только нужные типы потоков.
    # M4V/MOV файлы могут содержать tmcd, chapters и другие data потоки,
    # которые несовместимы с MKV или Vulkan pipeline.
    # Суффикс '?' означает "если есть" — не падать если потока нет.
    # ВАЖНО: 0:V (большая V) — только "настоящее" видео, БЕЗ attached pics,
    # thumbnails, cover art. Эти mjpeg картинки нельзя кодировать через Vulkan.
    ffmpeg_args.extend([
        '-map', '0:V',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map', '0:t?',  # attachments (шрифты) для MKV
        '-map_metadata', '0',
        '-map_chapters', '0',
    ])
    
    # Добавляем видеофильтр если нужен (для software decode pipeline)
    if video_filter:
        ffmpeg_args.extend(['-vf', video_filter])
    
    # Определяем параметры аудио (транскодирование несовместимых кодеков)
    if video_info.audio_codec and video_info.audio_codec.lower() in INCOMPATIBLE_AUDIO_CODECS:
        audio_args = ['-c:a', 'aac', '-b:a', '128k']
    else:
        audio_args = ['-c:a', 'copy']
    
    # AV1 Vulkan encoder (полностью на GPU)
    ffmpeg_args.extend([
        '-c:v', 'av1_vulkan',
        '-b:v', f'{target_br}k',
        '-maxrate', f'{max_br}k',
        '-bufsize', f'{buf_size}k',
        '-g', str(gop_size),
        '-keyint_min', str(keyint_min),
    ])
    ffmpeg_args.extend(audio_args)
    ffmpeg_args.extend([
        '-c:s', subtitle_codec,
        '-c:t', 'copy',
        '-progress', 'pipe:1',
        '-y',
        '__OUTPUT__'
    ])
    
    # Получаем полную команду Docker с маппингом путей
    cmd, container_input, container_output = docker_runner.build_ffmpeg_cmd(
        source_path, temp_path, ffmpeg_args
    )
    
    # Заменяем placeholders на контейнерные пути
    cmd = [container_input if x == '__INPUT__' else x for x in cmd]
    cmd = [container_output if x == '__OUTPUT__' else x for x in cmd]
    
    print(f"\n{Colors.GREEN}▶ Начало конвертации ({pipeline_name})...{Colors.RESET}\n")
    
    try:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # объединяем stderr, чтобы не заблокироваться на полном буфере
            text=True,
            encoding='utf-8',
            errors='replace',  # защищаемся от нечитаемых байт в выводе ffmpeg
            bufsize=1
        )
        
        # Парсим прогресс
        duration_us = video_info.duration * 1_000_000
        last_progress = -1
        last_speed: Optional[float] = None
        
        for line in current_process.stdout:
            if conversion_state == ConversionState.CANCELLED:
                break
                
            line = line.strip()
            log_tail.append(line)
            if line.startswith('out_time_us='):
                try:
                    current_us = int(line.split('=')[1])
                    processed_sec = current_us / 1_000_000
                    if duration_us > 0:
                        progress = min(100, int((current_us / duration_us) * 100))
                        if progress != last_progress:
                            last_progress = progress
                            eta_text = ""
                            speed_text = ""
                            
                            if last_speed is not None:
                                remaining = max(video_info.duration - processed_sec, 0.0)
                                eta_seconds = remaining / max(last_speed, 0.01)
                                eta_text = f" ETA {format_eta(eta_seconds)}"
                                speed_text = f" @ {last_speed:.2f}x"
                            
                            bar_width = 40
                            filled = int(bar_width * progress / 100)
                            bar = '█' * filled + '░' * (bar_width - filled)
                            print(
                                f"\r   {Colors.CYAN}[{bar}] {progress:3d}%{eta_text}{speed_text}{Colors.RESET}",
                                end='',
                                flush=True
                            )
                except (ValueError, IndexError):
                    pass
            elif line.startswith('speed='):
                try:
                    speed_str = line.split('=')[1].strip()
                    if speed_str.endswith('x'):
                        speed_str = speed_str[:-1]
                    last_speed = float(speed_str)
                except (ValueError, IndexError):
                    pass
            elif line.startswith('progress=end'):
                print(f"\r   {Colors.GREEN}[{'█' * 40}] 100%{Colors.RESET}")
        
        current_process.wait()
        
        if conversion_state == ConversionState.CANCELLED:
            cleanup_temp_file()
            return ConversionResult(False, video_info.size_bytes, 0, "Конвертация отменена")
        
        if current_process.returncode != 0:
            cleanup_temp_file()
            # Показываем последние строки лога для диагностики
            error_log = '\n'.join(log_tail)
            return ConversionResult(False, video_info.size_bytes, 0, f"Ошибка ffmpeg: {error_log[-4000:]}")
        
        # Проверяем что файл создан и не пустой
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            cleanup_temp_file()
            return ConversionResult(False, video_info.size_bytes, 0, "Выходной файл пуст или не создан")
        
        # Атомарное перемещение временного файла в целевой
        output_size = temp_path.stat().st_size
        shutil.move(str(temp_path), str(output_path))
        current_temp_file = None
        
        conversion_state = ConversionState.COMPLETED
        return ConversionResult(True, video_info.size_bytes, output_size, "Успешно")
        
    except Exception as e:
        cleanup_temp_file()
        return ConversionResult(False, video_info.size_bytes, 0, f"Исключение: {str(e)}")
    finally:
        current_process = None


def print_result(result: ConversionResult, output_path: Path):
    """Вывод результата конвертации"""
    if result.success:
        saved = result.source_size - result.output_size
        saved_percent = (saved / result.source_size * 100) if result.source_size > 0 else 0
        
        print(f"\n{Colors.GREEN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО{Colors.RESET}")
        print(f"{Colors.GREEN}{'═' * 60}{Colors.RESET}")
        print(f"\n   📁 Результат: {Colors.BOLD}{output_path}{Colors.RESET}")
        print(f"\n   {Colors.BLUE}📊 Статистика:{Colors.RESET}")
        print(f"   ┌─────────────────────────────────────────┐")
        print(f"   │ Исходный размер:  {format_size(result.source_size):>15} │")
        print(f"   │ Новый размер:     {format_size(result.output_size):>15} │")
        print(f"   │{'─' * 41}│")
        
        if saved >= 0:
            print(f"   │ {Colors.GREEN}💾 Сэкономлено:    {format_size(saved):>15} ({saved_percent:.1f}%){Colors.RESET} │")
        else:
            print(f"   │ {Colors.YELLOW}⚠ Увеличение:     {format_size(abs(saved)):>15} ({abs(saved_percent):.1f}%){Colors.RESET} │")
        
        print(f"   └─────────────────────────────────────────┘")
    else:
        print(f"\n{Colors.RED}✗ Ошибка конвертации: {result.message}{Colors.RESET}")


def is_video_file(path: Path) -> bool:
    """Проверка, является ли файл видео (mp4/mkv/m4v/mov/avi)"""
    return path.suffix.lower() in ['.mp4', '.mkv', '.m4v', '.mov', '.avi']


def get_video_files(directory: Path) -> list[Path]:
    """Получение списка видеофайлов в директории"""
    files = []
    for f in directory.iterdir():
        if f.is_file() and is_video_file(f):
            files.append(f)
    return sorted(files)


def generate_output_path(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Генерация пути для выходного файла (контейнер MKV)"""
    stem = input_path.stem
    # Убираем существующий суффикс -av1 если есть
    if stem.endswith('-av1'):
        stem = stem[:-4]
    
    new_name = f"{stem}-av1.mkv"
    
    if output_dir:
        return output_dir / new_name
    else:
        return input_path.parent / new_name


def ensure_mkv_output_path(path: Path) -> Path:
    """
    Принудительно использовать контейнер MKV, даже если пользователь указал другое расширение.
    """
    if path.suffix.lower() != '.mkv':
        new_path = path.with_suffix('.mkv')
        print(f"{Colors.DIM}Используем безопасный контейнер MKV: {path.name} → {new_path.name}{Colors.RESET}")
        return new_path
    return path


def prompt_input(prompt: str, default: str = "") -> str:
    """Запрос ввода с поддержкой значения по умолчанию"""
    if default:
        full_prompt = f"{prompt} [{Colors.DIM}{default}{Colors.RESET}]: "
    else:
        full_prompt = f"{prompt}: "
    
    try:
        value = input(full_prompt).strip()
        return value if value else default
    except EOFError:
        return default


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Запрос да/нет"""
    default_str = "Д/н" if default else "д/Н"
    try:
        answer = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not answer:
            return default
        return answer in ['y', 'yes', 'д', 'да', '1']
    except EOFError:
        return default


def prompt_overwrite_choice(prompt: str, default: str = "n") -> str:
    """
    Подтверждение перезаписи с ускоренными вариантами (RU/EN):
    'y/д' — перезаписать, 'n/н' — пропустить,
    'all/a/в' — перезаписывать все, 'skip_all/s/п' — пропускать все.
    Возвращает одно из: 'y', 'n', 'all', 'skip_all'.
    """
    default = default.lower()
    if default not in {"y", "n", "all", "skip_all"}:
        default = "n"
    
    default_hint = {
        "y": "Д/н/в/п | Y/n/a/s",
        "n": "д/Н/в/п | y/N/a/s",
        "all": "д/н/В/п | y/n/A/s",
        "skip_all": "д/н/в/П | y/n/a/S"
    }[default]
    
    mapping = {
        'y': 'y', 'д': 'y', 'd': 'y', 'yes': 'y',
        'n': 'n', 'н': 'n', 'no': 'n',
        'a': 'all', 'в': 'all', 'all': 'all',
        's': 'skip_all', 'п': 'skip_all', 'skip': 'skip_all',
        'sa': 'skip_all', 'skip_all': 'skip_all', 'skipall': 'skip_all'
    }
    
    try:
        answer = input(f"{prompt} [{default_hint}]: ").strip().lower()
    except EOFError:
        return default
    
    if not answer:
        return default
    
    return mapping.get(answer, default)


def check_docker():
    """Проверка наличия Docker и образа ffmpeg"""
    # Проверяем Docker
    if not shutil.which('docker'):
        print(f"{Colors.RED}Ошибка: Docker не найден. Установите Docker.{Colors.RESET}")
        sys.exit(1)
    
    # Проверяем что Docker daemon запущен
    result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{Colors.RED}Ошибка: Docker daemon не запущен.{Colors.RESET}")
        print(f"{Colors.DIM}Запустите: sudo systemctl start docker{Colors.RESET}")
        sys.exit(1)
    
    # Проверяем наличие образа (или тянем)
    print(f"{Colors.DIM}Проверка Docker образа {DOCKER_IMAGE}...{Colors.RESET}")
    result = subprocess.run(
        ['docker', 'image', 'inspect', DOCKER_IMAGE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"{Colors.YELLOW}Образ не найден, загружаем {DOCKER_IMAGE}...{Colors.RESET}")
        pull_result = subprocess.run(
            ['docker', 'pull', DOCKER_IMAGE],
            capture_output=False
        )
        if pull_result.returncode != 0:
            print(f"{Colors.RED}Ошибка загрузки образа.{Colors.RESET}")
            sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Docker готов{Colors.RESET}")


def main():
    """Главная функция"""
    # Устанавливаем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_banner()
    
    # Проверяем Docker вместо локального ffmpeg
    check_docker()
    
    while True:
        # Запрос пути
        print(f"\n{Colors.BOLD}Введите путь к файлу или папке с видео:{Colors.RESET}")
        print(f"{Colors.DIM}(или 'q' для выхода){Colors.RESET}")
        
        input_path_str = prompt_input("Путь")
        
        if input_path_str.lower() in ['q', 'quit', 'exit', 'в', 'выход']:
            print(f"\n{Colors.CYAN}До свидания!{Colors.RESET}")
            break
        
        if not input_path_str:
            print(f"{Colors.YELLOW}Путь не указан{Colors.RESET}")
            continue
        
        input_path = Path(input_path_str).expanduser().resolve()
        
        if not input_path.exists():
            print(f"{Colors.RED}Путь не существует: {input_path}{Colors.RESET}")
            continue
        
        # Определяем файлы для обработки
        files_to_process: list[Path] = []
        output_dir: Optional[Path] = None
        
        if input_path.is_file():
            if not is_video_file(input_path):
                print(f"{Colors.RED}Файл не является видео (mp4/mkv/m4v/mov/avi): {input_path}{Colors.RESET}")
                continue
            files_to_process = [input_path]
            
            # Предлагаем путь сохранения
            default_output = generate_output_path(input_path)
            print(f"\n{Colors.BOLD}Путь для сохранения:{Colors.RESET}")
            output_str = prompt_input("Выходной файл", str(default_output))
            output_path = Path(output_str).expanduser().resolve()
            output_path = ensure_mkv_output_path(output_path)
            
            # Создаём директорию если нужно
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        elif input_path.is_dir():
            files_to_process = get_video_files(input_path)
            
            if not files_to_process:
                print(f"{Colors.YELLOW}В папке нет видеофайлов (mp4/mkv/m4v/mov/avi){Colors.RESET}")
                continue
            
            print(f"\n{Colors.BLUE}Найдено видеофайлов: {len(files_to_process)}{Colors.RESET}")
            for f in files_to_process[:5]:
                print(f"   • {f.name}")
            if len(files_to_process) > 5:
                print(f"   ... и ещё {len(files_to_process) - 5}")
            
            # Предлагаем папку для сохранения
            default_output_dir = input_path.parent / f"{input_path.name}-av1"
            print(f"\n{Colors.BOLD}Папка для сохранения:{Colors.RESET}")
            output_dir_str = prompt_input("Выходная папка", str(default_output_dir))
            output_dir = Path(output_dir_str).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Подтверждение
        if not prompt_yes_no(f"\n{Colors.YELLOW}Начать конвертацию?{Colors.RESET}"):
            print(f"{Colors.DIM}Отменено{Colors.RESET}")
            continue
        
        # Обрабатываем файлы
        total_files = len(files_to_process)
        successful = 0
        failed = 0
        total_saved = 0
        skipped_files: list[tuple[Path, str]] = []
        overwrite_mode = "ask"  # ask | all_yes | all_no
        
        for idx, file_path in enumerate(files_to_process, 1):
            print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
            print(f"{Colors.HEADER}[{idx}/{total_files}] Обработка: {file_path.name}{Colors.RESET}")
            print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")
            
            # Получаем информацию о видео
            video_info = get_video_info(file_path)
            
            if not video_info:
                print(f"{Colors.RED}Не удалось получить информацию о видео{Colors.RESET}")
                failed += 1
                continue
            
            # Проверяем кодек — пропускаем h265/hevc и av1
            skip_codecs = {'av1', 'hevc', 'h265'}
            if video_info.codec.lower() in skip_codecs:
                codec_name = 'AV1' if video_info.codec.lower() == 'av1' else 'H265/HEVC'
                print(f"{Colors.YELLOW}⏭ Пропуск: файл уже в формате {codec_name}{Colors.RESET}")
                skipped_files.append((file_path, codec_name))
                continue
            
            print_video_info(video_info)
            target_br, _, _ = calculate_av1_bitrate(video_info)
            if target_br >= int(video_info.bitrate * WORTHINESS_THRESHOLD):
                print(
                    f"\n{Colors.YELLOW}⏭ Пропуск: целевой битрейт {target_br} kbps "
                    f"близок к исходному {video_info.bitrate} kbps.{Colors.RESET}"
                )
                skipped_files.append((file_path, "Нецелесообразно"))
                continue
            
            # Определяем выходной путь
            if output_dir:
                out_path = generate_output_path(file_path, output_dir)
            else:
                out_path = output_path  # Для одиночного файла
            
            # Проверяем, не существует ли уже выходной файл
            if out_path.exists():
                if overwrite_mode == "all_no":
                    print(f"{Colors.DIM}Пропуск: {out_path.name} (выбрано 'пропускать все / skip all'){Colors.RESET}")
                    continue
                elif overwrite_mode == "all_yes":
                    pass
                else:
                    print(f"{Colors.YELLOW}Файл уже существует:{Colors.RESET} {out_path}")
                    print(f"{Colors.DIM}Варианты: д/y — перезаписать; н/n — пропустить; в/a — перезаписывать все; п/s — пропускать все.{Colors.RESET}")
                    choice = prompt_overwrite_choice(
                        f"{Colors.YELLOW}Перезаписать файл {out_path.name}?{Colors.RESET}",
                        default="n"
                    )
                    if choice == "all":
                        overwrite_mode = "all_yes"
                        print(f"{Colors.DIM}Выбрано: перезаписывать все последующие (all){Colors.RESET}")
                    elif choice == "skip_all":
                        overwrite_mode = "all_no"
                        print(f"{Colors.DIM}Выбрано: пропускать все последующие (skip all){Colors.RESET}")
                    
                    if choice in {"n", "skip_all"}:
                        reason = "пропуск этого файла" if choice == "n" else "пропуск всех последующих"
                        print(f"{Colors.DIM}Пропуск: {out_path.name} ({reason}){Colors.RESET}")
                        continue
            
            # Конвертируем
            result = convert_video(file_path, out_path, video_info)
            print_result(result, out_path)
            
            if result.success:
                successful += 1
                total_saved += (result.source_size - result.output_size)
            else:
                failed += 1
        
        # Итоговая статистика для пакетной обработки
        if total_files > 1:
            print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
            print(f"{Colors.CYAN}📊 ИТОГОВАЯ СТАТИСТИКА{Colors.RESET}")
            print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
            print(f"   Всего файлов: {total_files}")
            print(f"   {Colors.GREEN}✓ Успешно: {successful}{Colors.RESET}")
            if failed > 0:
                print(f"   {Colors.RED}✗ Ошибок: {failed}{Colors.RESET}")
            if skipped_files:
                print(f"   {Colors.YELLOW}⏭ Пропущено: {len(skipped_files)}{Colors.RESET}")
            
            if total_saved >= 0:
                print(f"\n   {Colors.GREEN}💾 Всего сэкономлено: {format_size(total_saved)}{Colors.RESET}")
            else:
                print(f"\n   {Colors.YELLOW}⚠ Общее увеличение: {format_size(abs(total_saved))}{Colors.RESET}")
        
        # Предложение скопировать пропущенные файлы
        if skipped_files and successful > 0 and output_dir:
            print(f"\n{Colors.CYAN}{'─' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}📁 Обнаружены пропущенные файлы:{Colors.RESET}")
            for skip_path, skip_reason in skipped_files[:5]:
                print(f"   • {skip_path.name} ({skip_reason})")
            if len(skipped_files) > 5:
                print(f"   ... и ещё {len(skipped_files) - 5}")
            
            print(f"\n{Colors.DIM}Скопировать их в выходную папку для полной коллекции?{Colors.RESET}")
            if prompt_yes_no(f"{Colors.YELLOW}Копировать {len(skipped_files)} файл(ов)?{Colors.RESET}", default=False):
                copied = 0
                for skip_path, _ in skipped_files:
                    dest_path = output_dir / skip_path.name
                    try:
                        if dest_path.exists():
                            print(f"   {Colors.DIM}⏭ Пропуск (существует): {skip_path.name}{Colors.RESET}")
                        else:
                            shutil.copy2(str(skip_path), str(dest_path))
                            print(f"   {Colors.GREEN}✓ Скопирован: {skip_path.name}{Colors.RESET}")
                            copied += 1
                    except OSError as e:
                        print(f"   {Colors.RED}✗ Ошибка копирования {skip_path.name}: {e}{Colors.RESET}")
                
                if copied > 0:
                    print(f"\n{Colors.GREEN}✓ Скопировано файлов: {copied}{Colors.RESET}")
            else:
                print(f"{Colors.DIM}Копирование отклонено{Colors.RESET}")
        
        print()


if __name__ == '__main__':
    main()
