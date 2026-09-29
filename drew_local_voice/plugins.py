from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional

from .commands import CommandDefinition, CommandRegistry
from .profiles import DictationProfile, ProfileRegistry, built_in_profiles
from .settings import SettingSpec, SettingsRegistry, built_in_settings


@dataclass(frozen=True)
class PluginFailure:
    plugin: str
    error_type: str
    message: str


@dataclass(frozen=True)
class PluginLoadReport:
    loaded: tuple[str, ...] = ()
    failed: tuple[PluginFailure, ...] = ()

    def combine(self, other: "PluginLoadReport") -> "PluginLoadReport":
        return PluginLoadReport(self.loaded + other.loaded, self.failed + other.failed)


class ExtensionRegistry:
    """Single extension surface shared by built-ins, local files, and packages."""

    def __init__(self) -> None:
        self.commands = CommandRegistry()
        self.profiles = ProfileRegistry(built_in_profiles())
        self.settings = SettingsRegistry(built_in_settings())
        self.metadata: dict[str, Any] = {}
        for phrase, replacement in {
            "dot slash": "./",
            "double greater than": ">>",
            "double less than": "<<",
            "or or": "||",
            "home directory": "~/",
        }.items():
            self.register_spoken_replacement(
                phrase,
                replacement,
                source="core.terminal",
            )

    def register_spoken_replacement(
        self,
        phrase: str,
        replacement: str,
        *,
        source: str = "plugin",
        replace: bool = True,
    ) -> None:
        self.commands.register_replacement(
            phrase,
            replacement,
            source=source,
            replace=replace,
        )

    def register_command(self, command: CommandDefinition, *, replace: bool = True) -> None:
        self.commands.register_command(command, replace=replace)

    def register_profile(
        self,
        profile: Optional[DictationProfile] = None,
        *,
        id: str = "",
        label: str = "",
        mode: str = "general",
        **kwargs: Any,
    ) -> None:
        profile = profile or DictationProfile(id=id, label=label, mode=mode, **kwargs)
        self.profiles.register(profile)

    def register_setting(self, spec: SettingSpec, *, replace: bool = False) -> None:
        self.settings.register(spec, replace=replace)


class PluginManager:
    def __init__(self, registry: ExtensionRegistry) -> None:
        self.registry = registry

    def _register_module(self, name: str, module: ModuleType) -> None:
        register = getattr(module, "register", None)
        if not callable(register):
            raise TypeError(f"plugin {name!r} must expose register(registry)")
        register(self.registry)

    def load_directory(self, directory: Path | str) -> PluginLoadReport:
        root = Path(directory)
        if not root.exists():
            return PluginLoadReport()
        loaded: list[str] = []
        failed: list[PluginFailure] = []
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            name = path.stem
            module_name = f"drew_local_voice_user_plugin_{name}_{abs(hash(path.resolve()))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot build import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self._register_module(name, module)
                loaded.append(name)
            except Exception as exc:
                failed.append(
                    PluginFailure(name, type(exc).__name__, str(exc)[:500])
                )
            finally:
                sys.modules.pop(module_name, None)
        return PluginLoadReport(tuple(loaded), tuple(failed))

    def load_entry_points(
        self,
        group: str = "drew_local_voice.plugins",
    ) -> PluginLoadReport:
        loaded: list[str] = []
        failed: list[PluginFailure] = []
        try:
            entry_points: Iterable[Any] = importlib.metadata.entry_points().select(group=group)
        except AttributeError:
            entry_points = importlib.metadata.entry_points().get(group, ())
        for entry_point in entry_points:
            try:
                plugin = entry_point.load()
                if isinstance(plugin, ModuleType):
                    self._register_module(entry_point.name, plugin)
                elif callable(plugin):
                    plugin(self.registry)
                else:
                    raise TypeError("entry point must be a module or callable")
                loaded.append(entry_point.name)
            except Exception as exc:
                failed.append(
                    PluginFailure(entry_point.name, type(exc).__name__, str(exc)[:500])
                )
        return PluginLoadReport(tuple(loaded), tuple(failed))


def load_extensions(
    local_directory: Optional[Path | str] = None,
    *,
    include_entry_points: bool = True,
) -> tuple[ExtensionRegistry, PluginLoadReport]:
    registry = ExtensionRegistry()
    manager = PluginManager(registry)
    report = PluginLoadReport()
    if include_entry_points:
        report = report.combine(manager.load_entry_points())
    if local_directory is not None:
        report = report.combine(manager.load_directory(local_directory))
    return registry, report
