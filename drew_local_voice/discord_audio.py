"""Windows Core Audio capture-session control for Discord.

Muting Discord's capture session silences only the microphone audio delivered to
Discord. GlitchScribe keeps receiving the same microphone, and no window focus,
keyboard shortcut, Discord token, or system-wide endpoint mute is involved.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_float, c_int, c_void_p
from ctypes.wintypes import BOOL, DWORD
from dataclasses import dataclass
from typing import Any

from comtypes import CLSCTX_ALL, COMMETHOD, GUID, HRESULT, IUnknown, CoCreateInstance


class IMMDevice(IUnknown):
    _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "Activate",
            (["in"], POINTER(GUID), "iid"),
            (["in"], DWORD, "clsctx"),
            (["in"], c_void_p, "params"),
            (["out", "retval"], POINTER(c_void_p), "interface"),
        ),
        COMMETHOD([], HRESULT, "OpenPropertyStore"),
        COMMETHOD([], HRESULT, "GetId"),
        COMMETHOD([], HRESULT, "GetState"),
    ]


class IMMDeviceCollection(IUnknown):
    _iid_ = GUID("{0BD7A1BE-7A1A-44DB-8397-C0A18F46BCEF}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "GetCount",
            (["out", "retval"], POINTER(DWORD), "count"),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "Item",
            (["in"], DWORD, "index"),
            (["out", "retval"], POINTER(POINTER(IMMDevice)), "device"),
        ),
    ]


class IMMDeviceEnumerator(IUnknown):
    _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "EnumAudioEndpoints",
            (["in"], c_int, "data_flow"),
            (["in"], DWORD, "state_mask"),
            (
                ["out", "retval"],
                POINTER(POINTER(IMMDeviceCollection)),
                "devices",
            ),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "GetDefaultAudioEndpoint",
            (["in"], c_int, "data_flow"),
            (["in"], c_int, "role"),
            (["out", "retval"], POINTER(POINTER(IMMDevice)), "device"),
        ),
        COMMETHOD([], HRESULT, "GetDevice"),
        COMMETHOD([], HRESULT, "RegisterEndpointNotificationCallback"),
        COMMETHOD([], HRESULT, "UnregisterEndpointNotificationCallback"),
    ]


class IAudioSessionControl(IUnknown):
    _iid_ = GUID("{F4B1A599-7266-4319-A8CA-E70ACB11E8CD}")
    _methods_ = [
        COMMETHOD([], HRESULT, name)
        for name in (
            "GetState",
            "GetDisplayName",
            "SetDisplayName",
            "GetIconPath",
            "SetIconPath",
            "GetGroupingParam",
            "SetGroupingParam",
            "RegisterAudioSessionNotification",
            "UnregisterAudioSessionNotification",
        )
    ]


class IAudioSessionControl2(IAudioSessionControl):
    _iid_ = GUID("{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetSessionIdentifier"),
        COMMETHOD([], HRESULT, "GetSessionInstanceIdentifier"),
        COMMETHOD(
            [],
            HRESULT,
            "GetProcessId",
            (["out", "retval"], POINTER(DWORD), "pid"),
        ),
        COMMETHOD([], HRESULT, "IsSystemSoundsSession"),
        COMMETHOD([], HRESULT, "SetDuckingPreference"),
    ]


class ISimpleAudioVolume(IUnknown):
    _iid_ = GUID("{87CE5498-68D6-44E5-9215-6DA47EF883D8}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "SetMasterVolume",
            (["in"], c_float, "level"),
            (["in"], POINTER(GUID), "event_context"),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "GetMasterVolume",
            (["out", "retval"], POINTER(c_float), "level"),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "SetMute",
            (["in"], BOOL, "muted"),
            (["in"], POINTER(GUID), "event_context"),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "GetMute",
            (["out", "retval"], POINTER(BOOL), "muted"),
        ),
    ]


class IAudioSessionEnumerator(IUnknown):
    _iid_ = GUID("{E2F5BB11-0570-40CA-ACDD-3AA01277DEE8}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "GetCount",
            (["out", "retval"], POINTER(c_int), "count"),
        ),
        COMMETHOD(
            [],
            HRESULT,
            "GetSession",
            (["in"], c_int, "index"),
            (["out", "retval"], POINTER(POINTER(IAudioSessionControl)), "session"),
        ),
    ]


class IAudioSessionManager(IUnknown):
    _iid_ = GUID("{BFA971F1-4D5E-40BB-935E-967039BFBEE4}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetAudioSessionControl"),
        COMMETHOD([], HRESULT, "GetSimpleAudioVolume"),
    ]


class IAudioSessionManager2(IAudioSessionManager):
    _iid_ = GUID("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "GetSessionEnumerator",
            (["out", "retval"], POINTER(POINTER(IAudioSessionEnumerator)), "sessions"),
        ),
        COMMETHOD([], HRESULT, "RegisterSessionNotification"),
        COMMETHOD([], HRESULT, "UnregisterSessionNotification"),
        COMMETHOD([], HRESULT, "RegisterDuckNotification"),
        COMMETHOD([], HRESULT, "UnregisterDuckNotification"),
    ]


@dataclass(frozen=True)
class _DiscordCaptureSession:
    pid: int
    identity: int
    volume: Any


class WindowsDiscordCaptureSessions:
    """Prior-state-aware mute control for Discord microphone capture sessions."""

    _CLSID_MMDEVICE_ENUMERATOR = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
    _E_CAPTURE = 1
    _DEVICE_STATE_ACTIVE = 0x00000001

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Discord capture-session control is available only on Windows")
        import comtypes

        comtypes.CoInitialize()
        self._comtypes = comtypes
        self._restore: dict[int, tuple[Any, bool]] = {}
        self._closed = False
        endpoint_enumerator = CoCreateInstance(
            self._CLSID_MMDEVICE_ENUMERATOR,
            IMMDeviceEnumerator,
            CLSCTX_ALL,
        )
        devices = endpoint_enumerator.EnumAudioEndpoints(
            self._E_CAPTURE,
            self._DEVICE_STATE_ACTIVE,
        )
        self._managers: list[Any] = []
        for index in range(int(devices.GetCount())):
            device = devices.Item(index)
            try:
                raw_manager = device.Activate(
                    byref(IAudioSessionManager2._iid_),
                    CLSCTX_ALL,
                    None,
                )
                self._managers.append(
                    ctypes.cast(raw_manager, POINTER(IAudioSessionManager2))
                )
            except Exception:
                # One unavailable endpoint must not hide sessions on the rest.
                continue

    @staticmethod
    def _process_path(pid: int) -> str:
        if pid <= 0:
            return ""
        kernel32 = ctypes.windll.kernel32
        process = kernel32.OpenProcess(0x1000, False, int(pid))
        if not process:
            return ""
        try:
            size = DWORD(32768)
            path = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(
                process,
                0,
                path,
                byref(size),
            ):
                return path.value
            return ""
        finally:
            kernel32.CloseHandle(process)

    @staticmethod
    def _is_installed_discord(path: str) -> bool:
        normalized = os.path.normcase(os.path.normpath(str(path or "")))
        local_app_data = os.path.normcase(
            os.path.normpath(os.environ.get("LOCALAPPDATA", ""))
        )
        discord_root = os.path.join(local_app_data, "discord") + os.sep
        return bool(
            local_app_data
            and normalized.startswith(discord_root)
            and normalized.endswith(os.sep + "discord.exe")
        )

    def _discord_sessions(self) -> list[_DiscordCaptureSession]:
        matches: list[_DiscordCaptureSession] = []
        seen: set[int] = set()
        for manager in self._managers:
            sessions = manager.GetSessionEnumerator()
            for index in range(sessions.GetCount()):
                session = sessions.GetSession(index)
                control = session.QueryInterface(IAudioSessionControl2)
                pid = int(control.GetProcessId())
                path = self._process_path(pid)
                if self._is_installed_discord(path):
                    volume = session.QueryInterface(ISimpleAudioVolume)
                    identity = int(ctypes.cast(volume, c_void_p).value or 0)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    matches.append(
                        _DiscordCaptureSession(
                            pid=pid,
                            identity=identity,
                            volume=volume,
                        )
                    )
        return matches

    def muted_state(self) -> bool | None:
        sessions = self._discord_sessions()
        if not sessions:
            return None
        return all(bool(session.volume.GetMute()) for session in sessions)

    def reconcile_muted(self) -> tuple[int, int, bool]:
        """Mute every current Discord capture session and own its prior state.

        Returns ``(session_count, newly_owned_count, verified)``.  Calling this
        repeatedly is safe and lets the coordinator catch Discord audio-session
        recreation while Speak Now remains active.
        """

        sessions = self._discord_sessions()
        newly_owned = 0
        verified = True
        for session in sessions:
            if session.identity not in self._restore:
                self._restore[session.identity] = (
                    session.volume,
                    bool(session.volume.GetMute()),
                )
                newly_owned += 1
            try:
                if not bool(session.volume.GetMute()):
                    session.volume.SetMute(True, None)
                verified = verified and bool(session.volume.GetMute())
            except Exception:
                # Ownership was recorded before mutation, so a partial failure
                # can still be rolled back by end/close.
                verified = False
        return len(sessions), newly_owned, verified

    def set_muted(self, muted: bool) -> bool:
        desired = bool(muted)
        if desired:
            session_count, _newly_owned, verified = self.reconcile_muted()
            return bool(session_count) and verified

        restore = self._restore
        if not restore:
            sessions = self._discord_sessions()
            for session in sessions:
                session.volume.SetMute(False, None)
            return bool(sessions) and all(
                not bool(session.volume.GetMute()) for session in sessions
            )
        current_sessions = {
            session.identity: session.volume for session in self._discord_sessions()
        }
        pending: dict[int, tuple[Any, bool]] = {}
        for identity, (saved_volume, prior_muted) in tuple(restore.items()):
            volume = current_sessions.get(identity)
            if volume is None:
                # The exact session no longer exists. A replacement has a new
                # identity and is independently owned by reconciliation.
                continue
            try:
                volume.SetMute(bool(prior_muted), None)
                if bool(volume.GetMute()) != bool(prior_muted):
                    pending[identity] = (volume, prior_muted)
            except Exception:
                pending[identity] = (saved_volume, prior_muted)
        self._restore = pending
        return not pending

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._restore:
                self.set_muted(False)
        finally:
            self._managers = []
            self._comtypes.CoUninitialize()
