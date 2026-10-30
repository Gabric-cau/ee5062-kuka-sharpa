"""Measure the actual local GLFW/WGL context, separately from CPU physics."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import glfw
import imageio.v2 as imageio
import mujoco
import numpy as np
from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.rollout import GraspRollout
from kuka_sharpa_grasp.scene import build_scene


def executable_and_session() -> dict:
    info = {"python": sys.executable, "base_python": sys._base_executable,
            "python_version": platform.python_version(), "pid": os.getpid(),
            "session_name": os.environ.get("SESSIONNAME", "unknown")}
    if sys.platform == "win32":
        path = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetModuleFileNameW(None, path, len(path))
        session = ctypes.c_uint32()
        ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session))
        info.update(actual_executable=path.value, session_id=session.value)
    else:
        info["actual_executable"] = sys.executable
    return info


def prefer_nvidia(info: dict, backup: Path) -> dict:
    """Change only these Python applications' Windows GPU preferences."""
    if sys.platform != "win32":
        raise ValueError("GPU preferences are Windows-only")
    import winreg

    if backup.exists():
        raise FileExistsError(backup)
    key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
    paths = sorted({info["python"], info["base_python"], info["actual_executable"]})
    previous = {}
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        for path in paths:
            try:
                value, kind = winreg.QueryValueEx(key, path)
                previous[path] = {"value": value, "kind": kind}
            except FileNotFoundError:
                previous[path] = None
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(json.dumps({"key": "HKCU\\" + key_path, "previous": previous}, indent=2), encoding="utf-8")
        for path in paths:
            old = previous[path]
            value = old["value"] if old and old["kind"] == winreg.REG_SZ else ""
            fields = [field for field in value.split(";") if field and not field.startswith("GpuPreference=")]
            winreg.SetValueEx(key, path, 0, winreg.REG_SZ, ";".join([*fields, "GpuPreference=2", ""]))
    return {"applications": paths, "backup": str(backup),
            "note": "Per-application preference only; a fresh-process GL measurement is required."}


def camera_for(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = (0.42, 0, 0.65)
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -22
    return camera


def probe(model: mujoco.MjModel, data: mujoco.MjData, visible: bool,
          seconds: float, screenshot: Path | None) -> dict:
    window = None
    context = None
    if not glfw.init():
        raise RuntimeError("GLFW could not initialize this Windows session")
    try:
        glfw.window_hint(glfw.VISIBLE, int(visible))
        window = glfw.create_window(960, 720, "Dell KUKA-Sharpa | OpenGL hardware measurement", None, None)
        if not window:
            raise RuntimeError("GLFW could not create a WGL context")
        glfw.make_context_current(window)
        decode = lambda value: value.decode("utf-8", errors="replace") if value else "unknown"
        vendor, renderer, version = [decode(glGetString(key)) for key in (GL_VENDOR, GL_RENDERER, GL_VERSION)]
        scene = mujoco.MjvScene(model, maxgeom=10000)
        option = mujoco.MjvOption()
        camera = camera_for(model)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        start = time.perf_counter()
        pixels = None
        while True:
            width, height = glfw.get_framebuffer_size(window)
            if width and height:
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, context)
                pixels = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(pixels, None, viewport, context)
                glfw.swap_buffers(window)
            glfw.poll_events()
            if not visible or glfw.window_should_close(window) or time.perf_counter() - start >= seconds:
                break
            time.sleep(1 / 60)
        if pixels is None:
            raise RuntimeError("No framebuffer pixels were produced")
        if screenshot:
            imageio.imwrite(screenshot, np.flipud(pixels))
        return {"available": True, "opengl_vendor": vendor, "opengl_renderer": renderer,
                "opengl_version": version, "renderer_backend": "native_glfw_wgl" if sys.platform == "win32" else "native_glfw",
                "visible_window": visible, "framebuffer_shape": list(pixels.shape),
                "nonzero_pixels": int(np.any(pixels != 0, axis=2).sum()),
                "pixel_std": float(pixels.std()), "screenshot": str(screenshot) if screenshot else None,
                "nvidia_rtx2060_verified": "NVIDIA" in vendor.upper() and "2060" in renderer,
                "measurement_scope": "This GLFW context in this process; passive viewer uses a separate context."}
    finally:
        if context:
            context.free()
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--prefer-nvidia", action="store_true", help="set only Python application preferences, then exit for a fresh process")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("refusing to overwrite an existing hardware report")
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
              "platform": platform.platform(), "mujoco_version": mujoco.__version__,
              "physics_backend": "classic_mujoco_cpu", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "<default>"),
              **executable_and_session()}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False, timeout=15)
        report["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    if args.prefer_nvidia:
        if not args.output:
            parser.error("--prefer-nvidia requires --output for preference backup")
        report["gpu_preference_change"] = prefer_nvidia(report, args.output.with_suffix(".preferences-backup.json"))
        report["render"] = {"available": False, "reason": "Restart the Python process before verifying preferences."}
    else:
        with tempfile.TemporaryDirectory(prefix="kuka_hardware_") as temporary:
            scene = build_scene(Path(temporary) / "scene.xml", object_kind="ycb_can")
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            GraspRollout(model, data).reset()
            try:
                report["render"] = probe(model, data, args.visible, args.seconds,
                                         args.output.with_suffix(".png") if args.output else None)
            except (RuntimeError, mujoco.FatalError) as exc:
                report["render"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    if not args.prefer_nvidia and not report["render"].get("available"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
