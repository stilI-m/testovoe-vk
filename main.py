#!/usr/bin/env python3.12
"""Запускает Android-приложение и нажимает зелёную кнопку через ADB."""

import argparse
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

TIMEOUT_SEC = 10.0
POLL_INTERVAL_SEC = 0.5

GREEN_LOW = np.array([35, 80, 80], dtype=np.uint8)
GREEN_HIGH = np.array([85, 255, 255], dtype=np.uint8)
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.5
MIN_FILL_RATIO = 0.6

PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$")


class AdbError(RuntimeError):
    """Ошибка вызова adb или окружения."""


@dataclass(frozen=True)
class Button:
    x: int
    y: int
    area: int


def adb(*args: str, timeout: float) -> bytes:
    """Вызывает adb с таймаутом и возвращает stdout."""
    try:
        result = subprocess.run(
            ["adb", *args],
            capture_output=True,
            timeout=max(timeout, 0.1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdbError(f"Таймаут команды: adb {' '.join(args)}") from exc

    if result.returncode != 0:
        err = result.stderr.decode(errors="replace").strip()
        raise AdbError(err or f"adb завершился с кодом {result.returncode}")

    return result.stdout


def ensure_device() -> None:
    """Проверяет, что подключено ровно одно авторизованное устройство."""
    state = adb("get-state", timeout=5).decode().strip()
    if state != "device":
        raise AdbError(f"Устройство недоступно, состояние: {state}")


def ensure_installed(package: str) -> None:
    """Проверяет, что пакет установлен на телефоне."""
    try:
        out = adb("shell", "pm", "path", package, timeout=5).decode()
    except AdbError:
        out = ""

    if "package:" not in out:
        raise AdbError(f"Пакет {package} не установлен")


def launch_app(package: str, timeout: float) -> None:
    """Запускает launcher-activity пакета без указания имени activity."""
    out = adb(
        "shell",
        "monkey",
        "-p",
        package,
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
        timeout=timeout,
    ).decode(errors="replace")

    if "No activities found" in out:
        raise AdbError(f"У пакета {package} нет launcher-activity")


def is_foreground(package: str, timeout: float) -> bool:
    """Проверяет, что окно пакета сейчас в фокусе."""
    out = adb("shell", "dumpsys", "window", timeout=timeout).decode(errors="replace")
    for line in out.splitlines():
        if ("mCurrentFocus" in line or "mFocusedApp" in line) and package in line:
            return True
    return False


def take_screenshot(timeout: float) -> np.ndarray:
    """Снимает экран в PNG без записи файла на устройство."""
    png = adb("exec-out", "screencap", "-p", timeout=timeout)
    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AdbError("Не удалось декодировать скриншот")
    return img


def find_green_button(img: np.ndarray) -> Button | None:
    """Ищет самую крупную зелёную область, похожую на кнопку."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screen_area = img.shape[0] * img.shape[1]

    best: Button | None = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h

        if not MIN_AREA_RATIO <= rect_area / screen_area <= MAX_AREA_RATIO:
            continue

        if cv2.contourArea(contour) / rect_area < MIN_FILL_RATIO:
            continue

        if best is None or rect_area > best.area:
            best = Button(x + w // 2, y + h // 2, rect_area)

    return best


def tap(button: Button, timeout: float) -> None:
    """Нажимает на координаты; screencap и input tap используют одни пиксели."""
    adb("shell", "input", "tap", str(button.x), str(button.y), timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Нажать зелёную кнопку в Android-приложении"
    )
    parser.add_argument("package", help="Имя пакета, например com.example.app")
    package: str = parser.parse_args().package

    if not PACKAGE_RE.fullmatch(package):
        print(f"Некорректное имя пакета: {package}", file=sys.stderr)
        return 2

    if shutil.which("adb") is None:
        print("adb не найден в PATH", file=sys.stderr)
        return 2

    deadline = time.monotonic() + TIMEOUT_SEC

    def remaining() -> float:
        return deadline - time.monotonic()

    try:
        ensure_device()
        ensure_installed(package)
        launch_app(package, timeout=remaining())

        while remaining() > 0:
            if is_foreground(package, timeout=remaining()):
                button = find_green_button(take_screenshot(timeout=remaining()))
                if button is not None:
                    tap(button, timeout=remaining())
                    print(f"Нажата зелёная кнопка в точке ({button.x}, {button.y})")
                    return 0

            time.sleep(min(POLL_INTERVAL_SEC, max(remaining(), 0)))

    except AdbError as exc:
        print(f"Ошибка ADB: {exc}", file=sys.stderr)
        return 2

    print(f"Зелёная кнопка не найдена за {TIMEOUT_SEC:.0f} секунд")
    return 1


if __name__ == "__main__":
    sys.exit(main())
