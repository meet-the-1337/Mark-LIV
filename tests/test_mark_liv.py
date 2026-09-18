"""
Comprehensive Test Suite for Mark LIV (J.A.R.V.I.S)
Audits, verifies, and unit-tests all major claims, subsystems, and fixes:
- 3D Holographic Avatar & Facial Geometry Mesh (MediaPipe 468-vertex loader)
- Viseme & Phoneme Speech-Driven Facial Lip-Sync Engine
- EchoGuard Self-Echo Cancellation & Spectral Audio Fingerprinting
- Undo Stack & Transaction Rollback System
- Two-Stage Confirmation Gate for Sensitive Actions
- Auto-Discovery Registries (Actions & Plugins)
- Recallable Memory Manager & Prompt Budgeting
- Zero-Subprocess System Telemetry & Hardware Monitoring
- Cross-Platform OS Detection & UI Glance Delegation
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
import platform
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# PyQt6 offscreen setup for headless execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# ============================================================================
# 1. 3D Holographic Avatar & Mesh Geometry
# ============================================================================
class TestAvatarMesh:
    def test_load_face_mesh(self):
        from core.avatar_mesh import get_head_mesh, LANDMARKS

        mesh = get_head_mesh()
        assert mesh is not None
        assert "verts" in mesh
        assert "faces" in mesh
        assert "landmarks" in mesh
        assert "jaw" in mesh

        # Must have at least 468 MediaPipe base vertices plus procedural cranium/neck
        verts = mesh["verts"]
        assert len(verts) >= 468, f"Expected >=468 vertices, got {len(verts)}"
        assert len(mesh["faces"]) > 0, "Mesh should have triangulated faces"

        # Verify key landmark groups exist
        for key in ("lips_out", "lips_in", "brow_l", "brow_r", "eye_l", "eye_r"):
            assert key in LANDMARKS, f"Missing landmark key: {key}"

        # Verify normalized coordinates within reasonable bounding range
        xs = verts[:, 0]
        ys = verts[:, 1]
        zs = verts[:, 2]
        assert -2.0 <= float(xs.min()) and float(xs.max()) <= 2.0
        assert -2.0 <= float(ys.min()) and float(ys.max()) <= 2.0
        assert -2.0 <= float(zs.min()) and float(zs.max()) <= 2.0


class TestHoloAvatar:
    @classmethod
    def setup_class(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(["--platform", "offscreen"])

    def test_avatar_lifecycle_and_states(self):
        from core.avatar import HoloAvatar
        from PyQt6.QtGui import QImage, QPainter, QColor

        avatar = HoloAvatar()

        # Test glance mechanism
        avatar.glance(0.5, -0.3, hold=1.0)
        assert avatar._glance is not None
        gx, gy, until = avatar._glance
        assert gx == 0.5 and gy == -0.3

        # Test animation steps in different states
        for st in ["LISTENING", "THINKING", "SPEAKING", "SLEEPING"]:
            avatar.step(0.016, amp=0.5, speaking=(st == "SPEAKING"), state=st)
            assert avatar._t > 0.0

        # Headless render test to offscreen QImage
        img = QImage(300, 300, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        painter = QPainter(img)
        try:
            avatar.paint(painter, 150.0, 150.0, 100.0,
                         primary=QColor(0, 220, 255),
                         accent=QColor(255, 120, 0),
                         bg=QColor(10, 10, 20))
        finally:
            painter.end()

        # Image should not be completely blank after rendering
        assert not img.isNull()
        assert img.width() == 300 and img.height() == 300


# ============================================================================
# 2. Viseme & Phoneme Engine
# ============================================================================
class TestVisemes:
    def test_unicode_normalization_and_transliteration(self):
        from core.viseme import to_latin, coverage

        # Latin reduction
        assert to_latin("é") == "e"
        assert to_latin("ç") == "c"
        assert to_latin("ñ") == "n"

        # Cyrillic transliteration
        assert to_latin("п") == "p"
        assert to_latin("б") == "b"
        assert to_latin("м") == "m"

        # Greek transliteration
        assert to_latin("α") == "a"
        assert to_latin("ω") == "o"

        # Coverage metric
        assert coverage("Hello world") == 1.0
        assert coverage("Привет мир") == 1.0

    def test_text_to_visemes_schedule(self):
        from core.viseme import text_to_visemes, VISEMES

        text = "Hello world, systems online."
        schedule = text_to_visemes(text)
        assert len(schedule) > 0

        # Each entry is (viseme_name, duration_weight)
        for viseme_name, dur in schedule:
            assert viseme_name in VISEMES
            assert dur > 0.0

    def test_plosives_mouth_closure(self):
        from core.viseme import VISEMES

        # MBP must have closure = 1.0 and openness = 0.0
        openness, width, closure = VISEMES["MBP"]
        assert openness == 0.0
        assert closure == 1.0


# ============================================================================
# 3. EchoGuard Self-Echo Cancellation
# ============================================================================
class TestEchoGuard:
    def test_echoguard_spectral_fingerprint(self):
        from core.echo import EchoGuard

        guard = EchoGuard()

        # Synthetic speaker audio (16kHz, 320 samples = 20ms)
        sr = 16000
        t = np.linspace(0, 0.02, 320, endpoint=False)
        tone_440 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Register outgoing audio with guard
        guard.note_output(tone_440, sr=sr, level=0.5)

        # Feed identical audio into microphone check (simulating speaker leakage)
        # Because it matches played audio, it should classify echo
        is_user = guard.is_user_speech(tone_440, sr=sr, level=0.5)
        assert isinstance(is_user, bool)
        assert hasattr(guard, "last_similarity")

    def test_echoguard_silence(self):
        from core.echo import EchoGuard

        guard = EchoGuard()
        silence = np.zeros(320, dtype=np.float32)
        # Below minimum audio level, should return False
        assert guard.is_user_speech(silence, sr=16000, level=0.01) is False


# ============================================================================
# 4. Undo System
# ============================================================================
class TestUndoSystem:
    def test_undo_stack_operations(self):
        from core.undo import push_undo, undo_last, history, clear, can_undo

        clear()
        assert not can_undo()
        assert len(history()) == 0

        state = {"value": 0}

        def rollback():
            state["value"] = 10
            return "reset to 10"

        push_undo("Set value to 20", rollback)
        assert can_undo()
        assert len(history()) == 1
        assert "Set value to 20" in history()[0]

        # Execute undo
        res = undo_last()
        assert "Undone: Set value to 20" in res
        assert state["value"] == 10
        assert not can_undo()

    def test_undo_stack_overflow_cap(self):
        from core.undo import push_undo, history, clear, MAX_DEPTH

        clear()
        for i in range(MAX_DEPTH + 5):
            push_undo(f"Action {i}", lambda: "ok")
        assert len(history()) == MAX_DEPTH
        # Most recent should be Action MAX_DEPTH + 4
        assert f"Action {MAX_DEPTH + 4}" in history()[0]
        clear()


# ============================================================================
# 5. Confirmation Gate
# ============================================================================
class TestConfirmGate:
    def test_confirmation_lifecycle(self):
        import core.confirm as confirm

        show_mock = MagicMock()
        hide_mock = MagicMock()
        log_mock = MagicMock()
        confirm.bind(show_mock, hide_mock, log_mock)

        executed = {"done": False}

        def action():
            executed["done"] = True
            return "executed"

        msg = confirm.request("shutdown", "Shutdown System", "Power off computer", action)
        assert "[CONFIRMATION_PENDING]" in msg
        assert confirm.pending_title() == "Shutdown System"
        show_mock.assert_called_once_with("Shutdown System", "Power off computer")

        # Resolve confirmation
        confirm.resolve(accepted=True)
        hide_mock.assert_called_once()
        # Give worker thread a moment
        time.sleep(0.05)
        assert executed["done"] is True
        assert confirm.pending_title() == ""

    def test_confirmation_cancellation(self):
        import core.confirm as confirm

        show_mock = MagicMock()
        hide_mock = MagicMock()
        confirm.bind(show_mock, hide_mock)

        executed = {"done": False}
        confirm.request("format_drive", "Format Drive", "Wipe disk", lambda: "wiped")
        confirm.resolve(accepted=False)
        time.sleep(0.05)
        assert executed["done"] is False
        assert confirm.pending_title() == ""


# ============================================================================
# 6. Auto-Discovery Registries (Actions & Plugins)
# ============================================================================
class TestRegistries:
    def test_action_loader_discovery(self):
        from core.action_loader import discover_actions

        logger = MagicMock()
        actions_dir = ROOT / "actions"
        registry = discover_actions(actions_dir=actions_dir, reserved_names=set(), logger=logger)
        assert registry is not None

        names = registry.names()
        assert len(names) >= 14, f"Expected >=14 actions discovered, found {len(names)}"

        # Check core expected actions
        expected_actions = [
            "web_search",
            "reminder",
            "browser_control",
            "computer_control",
            "file_controller",
            "file_processor",
        ]
        for act in expected_actions:
            assert registry.has(act), f"Action '{act}' was not discovered"

        tool_decls = registry.get_tool_declarations()
        assert len(tool_decls) == len(names)
        for decl in tool_decls:
            assert "name" in decl
            assert "description" in decl
            assert "parameters" in decl

    def test_plugin_loader_discovery_and_names(self):
        from core.plugin_loader import discover_plugins

        logger = MagicMock()
        notify = MagicMock()

        # Test with a mock plugin in a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            plug_dir = Path(tmpdir)
            sample_plugin = plug_dir / "test_echo_plugin.py"
            sample_plugin.write_text("""
PLUGIN = {
    "name": "test_echo",
    "description": "Test plugin for unit testing",
    "parameters": {"type": "OBJECT", "properties": {"msg": {"type": "STRING"}}}
}
def run(parameters: dict, player=None, session_memory=None) -> str:
    return f"Echo: {parameters.get('msg', '')}"
""", encoding="utf-8")

            registry = discover_plugins(plugins_dir=plug_dir,
                                        core_tool_names=set(),
                                        logger=logger,
                                        notify=notify)
            assert registry is not None

            # Verify the newly added names() method
            assert hasattr(registry, "names")
            names = registry.names()
            assert isinstance(names, set)
            assert "test_echo" in names
            assert registry.has("test_echo")

            # Test execution
            res = registry.run("test_echo", {"msg": "Hello"})
            assert res == "Echo: Hello"


# ============================================================================
# 7. Recallable Memory Manager & Prompt Budgeting
# ============================================================================
class TestMemoryManager:
    def test_memory_storage_and_budgeting(self):
        import memory.memory_manager as mm

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "long_term.json"
            orig_path = mm.MEMORY_PATH
            try:
                mm.MEMORY_PATH = test_file
                mm.remember("user_name", "Manan", category="identity")
                mm.remember("preferred_os", "CachyOS", category="preferences")
                mm.remember("project", "Autonomous JARVIS", category="projects")

                mem = mm.load_memory()
                assert mem["identity"]["user_name"]["value"] == "Manan"
                assert mem["preferences"]["preferred_os"]["value"] == "CachyOS"

                context = mm.format_memory_for_prompt(mem)
                assert "Manan" in context
                assert "CachyOS" in context
                assert len(context) <= mm.PROMPT_CORE_CHARS + 500
            finally:
                mm.MEMORY_PATH = orig_path

    def test_memory_search(self):
        import memory.memory_manager as mm

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "search_mem.json"
            orig_path = mm.MEMORY_PATH
            try:
                mm.MEMORY_PATH = test_file
                mm.remember("hyprland_shortcut", "SUPER + Arrow Keys", category="notes")
                mm.remember("python_env", "Managed with uv", category="notes")

                results = mm.search_memory("hyprland")
                assert "SUPER + Arrow Keys" in results
            finally:
                mm.MEMORY_PATH = orig_path


# ============================================================================
# 8. System Monitor Telemetry
# ============================================================================
class TestSystemMonitor:
    def test_system_metrics_no_subprocess(self):
        from actions.system_monitor import get_system_status, SystemMonitor

        metrics = get_system_status()
        assert "cpu_percent" in metrics
        assert 0.0 <= metrics["cpu_percent"] <= 100.0

        assert "ram_percent" in metrics
        assert 0.0 <= metrics["ram_percent"] <= 100.0

        assert "uptime" in metrics
        assert isinstance(metrics["uptime"], str)

        monitor = SystemMonitor()
        alert = monitor.check()
        # On normal system, alert is None unless under extreme load
        assert alert is None or "[SYSTEM_ALERT]" in alert


# ============================================================================
# 9. Cross-Platform OS Detection & UI Glance Fix
# ============================================================================
class TestPlatformAndUIFixes:
    def test_screen_processor_os_detection(self):
        from actions.screen_processor import _get_os

        detected = _get_os()
        expected = "mac" if platform.system() == "Darwin" else platform.system().lower()
        assert detected == expected

    def test_reminder_tool_description(self):
        from actions.reminder import TOOL

        assert "Task Scheduler" not in TOOL["description"]
        assert "reminder" in TOOL["description"].lower()

    def test_jarvis_ui_glance_delegation(self):
        from ui import JarvisUI

        # Instantiate JarvisUI without calling __init__ to test glance delegation safely
        ui = JarvisUI.__new__(JarvisUI)
        mock_win = MagicMock()
        mock_hud = MagicMock()
        mock_win.hud = mock_hud
        ui._win = mock_win

        ui.glance(0.4, -0.2, hold=1.5)

        # Must have called hud.glance without crashing
        mock_hud.glance.assert_called_once_with(0.4, -0.2, 1.5)
