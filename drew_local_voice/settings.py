from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Iterable, Optional

from .models import NEMOTRON_MODEL_ID


DEFAULT_AUXILIARY_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


class SettingKind(str, Enum):
    BOOLEAN = "boolean"
    NUMBER = "number"
    TEXT = "text"
    CHOICE = "choice"
    PATH = "path"
    STRING_LIST = "string_list"


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    section: str
    kind: SettingKind
    default: Any
    description: str = ""
    guidance: str = ""
    choices: tuple[Any, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    restart_required: bool = False
    advanced: bool = False
    sensitive: bool = False
    presentation: str = "entry"
    step: Optional[float] = None


@dataclass(frozen=True)
class EffectRecipe:
    key: str
    title: str
    outcome: str
    changes: tuple[tuple[str, Any], ...]


def built_in_effect_recipes() -> tuple[EffectRecipe, ...]:
    """Coordinated starting points for user-visible dictation outcomes."""
    return (
        EffectRecipe(
            "protect_every_final_word",
            "Protect Every Final Word",
            "Preview speech in the bubble, then commit one complete final transcript with conservative tail protection.",
            (
                ("performance.preset", "reliable_stream"),
                ("typing.live_insert", False),
                ("endpoint.capture_tail_grace_seconds", 0.40),
                ("finalization.decoding_mode", "fast_full_buffer"),
                ("finalization.allow_quiet_tail_reuse", False),
            ),
        ),
        EffectRecipe(
            "fast_bubble_preview",
            "Fast Bubble Preview",
            "Make provisional words appear faster while retaining a complete final recognition pass on stop.",
            (
                ("streaming.enabled", True),
                ("streaming.interval_seconds", 0.18),
                ("streaming.minimum_audio_seconds", 0.45),
                ("streaming.minimum_new_audio_seconds", 0.14),
                ("streaming.stability_lag_seconds", 0.32),
                ("streaming.holdback_words", 1),
                ("ui.streaming_bubble", True),
                ("typing.live_insert", False),
            ),
        ),
        EffectRecipe(
            "rapid_speech_accuracy",
            "Rapid-Speech Accuracy",
            "Keep Nemotron 3.5 ASR active and use a complete final decode when speaking quickly or using long sentences.",
            (
                ("recognition.model", NEMOTRON_MODEL_ID),
                ("recognition.two_pass", True),
                ("streaming.maximum_partial_audio_seconds", 45.0),
                ("finalization.decoding_mode", "accurate_full_buffer"),
                ("finalization.allow_quiet_tail_reuse", False),
            ),
        ),
        EffectRecipe(
            "terminal_safe_dictation",
            "Terminal-Safe Dictation",
            "Use literal spoken symbols without inferred prose endings or automatic command execution.",
            (
                ("general.active_profile", "terminal"),
                ("typing.automatic_punctuation", False),
                ("typing.suppress_inferred_trailing_period", True),
                ("typing.spoken_punctuation", True),
                ("typing.trailing_space", False),
                ("commands.confirm_execution", True),
            ),
        ),
        EffectRecipe(
            "bubble_near_cursor",
            "Bubble Near Cursor",
            "Anchor the transcript bubble once near the text caret, fall back to the mouse, and preserve a draggable offset.",
            (
                ("ui.streaming_bubble", True),
                ("ui.preview_near_caret", True),
                ("ui.bubble_width", 420.0),
                ("ui.bubble_offset_x", 18.0),
                ("ui.bubble_offset_y", -88.0),
            ),
        ),
        EffectRecipe(
            "low_glare_streaming",
            "Low-Glare Streaming",
            "Use the Glitch Factor control with soft dark surfaces and restrained violet highlights.",
            (
                ("ui.theme", "dark"),
                ("ui.accent_color", "violet"),
                ("ui.overlay", True),
                ("ui.streaming_bubble", True),
                ("ui.bubble_opacity", 0.78),
            ),
        ),
        EffectRecipe(
            "streamer_large_text",
            "Streamer Large Text",
            "Make the live bubble easy to read from a distance during a stream with a wide, bold transcript surface.",
            (
                ("ui.streaming_bubble", True),
                ("ui.bubble_width", 720.0),
                ("ui.bubble_height", 120.0),
                ("ui.bubble_font_size", 18.0),
                ("ui.bubble_font_weight", "bold"),
                ("ui.bubble_max_characters", 280.0),
            ),
        ),
        EffectRecipe(
            "low_glare_chat",
            "Low-Glare Chat",
            "Keep chat previews calm and readable with a restrained dark surface that stays out of the way.",
            (
                ("ui.theme", "dark"),
                ("ui.streaming_bubble", True),
                ("ui.bubble_visual_style", "minimal"),
                ("ui.bubble_text_color_mode", "theme"),
                ("ui.bubble_border_width", 0.0),
                ("ui.bubble_opacity", 0.72),
            ),
        ),
        EffectRecipe(
            "neon_high_contrast",
            "Neon High-Contrast",
            "Use a vivid cyan edge and bright heavy text when the preview must remain unmistakable against a busy background.",
            (
                ("ui.theme", "dark"),
                ("ui.accent_color", "cyan"),
                ("ui.streaming_bubble", True),
                ("ui.bubble_visual_style", "neon_edge"),
                ("ui.bubble_text_color_mode", "bright"),
                ("ui.bubble_font_weight", "bold"),
                ("ui.bubble_border_width", 3.0),
            ),
        ),
        EffectRecipe(
            "compact_corner_bubble",
            "Compact Corner Bubble",
            "Reduce the preview to a small rounded bubble near the caret for focused, low-obstruction dictation.",
            (
                ("ui.streaming_bubble", True),
                ("ui.preview_near_caret", True),
                ("ui.bubble_width", 320.0),
                ("ui.bubble_height", 64.0),
                ("ui.bubble_font_size", 9.0),
                ("ui.bubble_corner_radius", 28.0),
                ("ui.bubble_offset_x", 24.0),
                ("ui.bubble_offset_y", -64.0),
            ),
        ),
        EffectRecipe(
            "rapid_preview_long_sample",
            "Rapid Preview / Long Sample",
            "Tune a fast, spacious preview for rapid testing; select the existing long synthetic sample to check wrapping and visible text capacity.",
            (
                ("ui.streaming_bubble", True),
                ("ui.bubble_width", 680.0),
                ("ui.bubble_height", 112.0),
                ("ui.bubble_font_size", 12.0),
                ("ui.bubble_max_characters", 1000.0),
                ("ui.bubble_visual_style", "glitch_glass"),
                ("ui.bubble_opacity", 0.80),
            ),
        ),
    )


class SettingsRegistry:
    def __init__(self, specs: Iterable[SettingSpec] = ()) -> None:
        self._specs: dict[str, SettingSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SettingSpec, *, replace: bool = False) -> None:
        if not spec.key or "." not in spec.key:
            raise ValueError("setting keys must be namespaced, for example typing.live_insert")
        if spec.kind is SettingKind.CHOICE and not spec.choices:
            raise ValueError(f"choice setting has no choices: {spec.key}")
        if not replace and spec.key in self._specs:
            raise KeyError(f"setting already registered: {spec.key}")
        self._specs[spec.key] = spec

    def get(self, key: str) -> SettingSpec:
        return self._specs[key]

    def validate(self, key: str, value: Any) -> Any:
        spec = self.get(key)
        if spec.kind is SettingKind.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            return value
        if spec.kind is SettingKind.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            numeric = float(value)
            if spec.minimum is not None and numeric < spec.minimum:
                raise ValueError(f"{key} must be at least {spec.minimum}")
            if spec.maximum is not None and numeric > spec.maximum:
                raise ValueError(f"{key} must be at most {spec.maximum}")
            return numeric
        if spec.kind is SettingKind.CHOICE:
            if value not in spec.choices:
                raise ValueError(f"{key} must be one of {spec.choices}")
            return value
        if spec.kind is SettingKind.STRING_LIST:
            if isinstance(value, str):
                return tuple(item.strip() for item in value.split(",") if item.strip())
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                return tuple(value)
            raise ValueError(f"{key} must be a list of strings")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        return value

    def defaults(self) -> dict[str, Any]:
        return {key: spec.default for key, spec in self._specs.items()}

    def to_schema(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self._specs.values():
            row = asdict(spec)
            row["kind"] = spec.kind.value
            row["choices"] = list(spec.choices)
            rows.append(row)
        return rows

    def sections(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.section for spec in self._specs.values()))


_SETTING_HELP: dict[str, tuple[str, str]] = {
    "general.active_profile": ("Chooses rules optimized for prose, chat, documents, terminals, code, or search.", "Use Automatic unless one application is repeatedly classified incorrectly."),
    "general.auto_profile": ("Selects a profile from the focused application's executable and window metadata.", "Leave enabled for mixed-app use; disable to force Active profile everywhere."),
    "general.hotkey": ("The system-wide key combination that starts or stops dictation.", "Ctrl + Alt + Space is the tested default; avoid shortcuts already owned by another app."),
    "general.hold_to_talk_enabled": ("Enables a physical hold-to-record shortcut alongside the existing toggle shortcut; Escape cancels active recording or transcription.", "Keep disabled until you choose a free function key; hold the key to talk, release it to finalize, or press Escape to discard."),
    "general.hold_shortcut": ("Chooses the function key used for hold-to-talk.", "Use F8 unless another application owns it; do not choose the same unmodified key as the toggle shortcut."),
    "general.hold_latch_modifier": ("Adds a modifier that latches an active hold recording without restarting the microphone.", "Hold F8, press Shift to continue hands-free, release the chord, then press F8 + Shift again to stop."),
    "performance.preset": ("Loads a tested bundle of streaming, endpoint, and finalization values.", "Use Reliable Stream for protected final words; use Balanced or Instant only after live testing."),
    "performance.allow_experimental": ("Unlocks speed options that intentionally trade away reliability checks.", "Keep disabled when sentence endings matter."),
    "recognition.model": ("The one explicit local speech-recognition model GlitchScribe is allowed to load.", "Nemotron 3.5 ASR Streaming 0.6B is the default. Every other selectable model remains visible here and is never loaded as an automatic fallback."),
    "model.idle_unload_minutes": ("Unloads speech model weights from VRAM/RAM after this many minutes without dictation so other local agents (e.g. Hermes) can use the GPU; 0 disables.", "Use 10 minutes for shared-GPU desktops; the process and tray stay alive and reload on the next dictation."),
    "recognition.additional_models": ("Adds explicitly named local model IDs or folders to the visible picker without loading them.", "Only add a model you intentionally installed. GlitchScribe will not download it or select it as a fallback."),
    "recognition.device": ("Chooses automatic placement, any detected NVIDIA GPU, or CPU inference.", "Use Automatic to select only a detected GPU with measured headroom; explicit CUDA choices are generated from the live NVIDIA inventory."),
    "recognition.vram_reserve_mb": ("Keeps this much GPU memory free after the selected speech model's estimated allocation.", "The Nemotron default is 512 MiB so the model can coexist with the current desktop load; increase it when more GPU headroom is available."),
    "recognition.compute_type": ("Requests numerical precision and memory use; the resource panel shows the backend's effective precision.", "Nemotron uses float16 on CUDA and float32 on CPU. Faster-Whisper options retain their supported CTranslate2 precision."),
    "recognition.language": ("Constrains decoding to a language instead of detecting it on every pass.", "Use en for English dictation; fixed language is faster and reduces language-switch mistakes."),
    "recognition.transcription_mode": ("Selects verbatim speech or intended clean speech for CrisperWhisper.", "Verbatim preserves fillers, repetitions, cut-offs, and vocal sounds; Intended produces a cleaner transcript. Other ASR backends ignore this setting."),
    "recognition.crisperwhisper_model_path": ("Local CrisperWhisper model directory.", "Point this at a downloaded CrisperWhisper 2.0 model directory. Leave blank to use models/crisperwhisper/<size>; the app never downloads weights automatically."),
    "recognition.custom_vocabulary": ("Biases recognition toward names, products, acronyms, and uncommon terms.", "Add short comma-separated terms you actually say; overly long prompts can reduce general accuracy."),
    "recognition.two_pass": ("Uses fast streaming recognition followed by complete final recognition.", "Keep enabled to protect sentence endings and revise uncertain streamed words."),
    "ai.cleanup_enabled": ("Runs an optional OpenRouter AI pass over the complete raw transcript before final insertion.", "Keep disabled for deterministic dictation; enable for prose cleanup only after reviewing the cloud data boundary."),
    "ai.cleanup_nearby_context": ("Includes up to 2,000 recent same-session characters as spelling-only context for OpenRouter cleanup.", "Enable for names and project terms in prose; the guard forbids introducing context that was not spoken."),
    "ai.cleanup_timeout_seconds": ("Limits the optional OpenRouter cleanup request independently from speech recognition.", "Use 20 seconds normally; timeout always falls back to raw text."),
    "ai.model": ("Names the OpenRouter Nemotron model used by AI cleanup and AI Action Wheel actions.", "This model stays blocked unless Allow auxiliary model requests is enabled and OPENROUTER_API_KEY is available."),
    "resources.allow_auxiliary_models": ("Controls whether GlitchScribe may send non-ASR text to OpenRouter.", "Keep disabled for speech-only operation. Enable deliberately after reviewing the exact model and cloud data boundary."),
    "streaming.enabled": ("Runs repeated recognition while you are still speaking.", "Keep enabled for live bubble preview; Lean Core and Instant turn this on by default."),
    "streaming.interval_seconds": ("Minimum cadence between partial recognition attempts.", "Lower feels faster but increases GPU work; 0.15–0.22 seconds is the snappy preview range."),
    "streaming.minimum_audio_seconds": ("Audio required before the first partial recognition pass.", "Lower starts preview sooner; about 0.40–0.50 seconds balances first words vs noise."),
    "streaming.minimum_new_audio_seconds": ("Fresh audio required before another partial pass.", "Lower updates faster but repeats more inference; 0.12–0.18 seconds is responsive."),
    "streaming.stability_lag_seconds": ("Keeps the newest timestamped words provisional until more audio confirms them.", "Affects live typing stability more than bubble preview; bubble shows full provisional text."),
    "streaming.holdback_words": ("Keeps this many trailing words out of direct insertion while still showing them in preview.", "Bubble always shows full partials; holdback only gates live insertion."),
    "streaming.timestamp_tolerance_seconds": ("Allows small timestamp movement when deciding whether a word is stable across passes.", "Increase if stable words rarely commit; decrease if mismatched words commit too early."),
    "streaming.maximum_partial_audio_seconds": ("Limits how much recent audio each live pass reprocesses during long dictation.", "30–45 seconds controls latency; finalization still uses the complete recording."),
    "streaming.quiet_tail_rms_threshold": ("Defines how quiet the newest audio must be for quiet-tail logic.", "Leave near 0.003 unless microphone diagnostics show a substantially different noise floor."),
    "typing.live_insert": ("Types timestamp-stable words into the original focused field during speech.", "Leave disabled for the reliable review-in-bubble workflow: press the trigger again to commit one complete final transcript. Enable only if you accept occasional live/final reconciliation conflicts."),
    "typing.smart_continuation": ("Uses recent same-target context to choose spacing and capitalization after stop/start.", "Leave enabled for natural chat and document continuation."),
    "typing.continuation_window_seconds": ("How long a stopped session can still be treated as continuation in the same field.", "30–45 seconds works for conversation; lower it if unrelated phrases are joined."),
    "typing.focus_recovery_seconds": ("Retains a finished transcript when a popup temporarily steals focus and retries only when the exact original field returns.", "Use 30 seconds for ordinary interruptions; increase toward 120 for slower authentication or notification popups."),
    "typing.automatic_punctuation": ("Renders spoken punctuation and applies prose punctuation rules.", "Enable for prose; terminal and code profiles suppress unsafe inferred endings."),
    "typing.suppress_inferred_trailing_period": ("Removes an unspoken final period while preserving explicitly spoken punctuation.", "Leave enabled if Whisper keeps adding periods you did not say."),
    "typing.spoken_punctuation": ("Converts phrases such as slash, comma, and open parenthesis into literal symbols.", "Enable for hands-free terminal and code work."),
    "typing.trailing_space": ("Adds a separator after completed prose so the next insertion does not stick to it.", "Enable for prose; terminal symbol-only phrases deliberately bypass it."),
    "typing.replace_selection": ("Allows final insertion to replace text that was selected when dictation began.", "Disable if selected text should never be overwritten."),
    "insertion.clipboard_fallback": ("When Unicode typing fails or text is very long, paste via the clipboard and restore prior clipboard contents.", "Keep enabled so stubborn controls and long transcripts still insert; clipboard is restored after paste."),
    "insertion.clipboard_threshold_chars": ("Character count at which insertion prefers clipboard paste over Unicode SendInput.", "400 balances short keystroke typing with reliable long-paste; raise it if you prefer keystrokes for longer phrases."),
    "general.paste_last_hotkey": ("Optional secondary chord that re-inserts the last successful transcript into the current focus.", "Use Ctrl + Shift + V or set None to disable; works even when hold-to-talk is off."),
    "endpoint.auto_stop": ("Stops recording only after speech has begun and sustained silence follows.", "Enable for hands-free use; the hotkey can always stop manually."),
    "endpoint.silence_seconds": ("Length of silence required before automatic stop begins finalization.", "Increase to 2.5–3.0 seconds when pauses or the final word are being cut off; lower it for faster stopping."),
    "endpoint.minimum_speech_seconds": ("Sustained above-threshold audio required before silence can stop a session.", "Increase to ignore clicks/noise; decrease only if very short words fail to establish speech."),
    "endpoint.rms_threshold": ("Audio energy threshold used to distinguish speech from background noise.", "Lower if quiet speech is missed; raise if room noise falsely counts as speech."),
    "endpoint.require_recognized_speech": ("Prevents automatic stop until recognition has produced non-empty text.", "Keep enabled so startup noise or silence does not immediately end dictation."),
    "endpoint.capture_tail_grace_seconds": ("Continues draining microphone callbacks briefly after stop before taking the final snapshot.", "Increase to 0.35–0.50 seconds for clipped final consonants or words; higher values add equal stop latency."),
    "endpoint.quiet_tail_reuse_seconds": ("Defines how much terminal quiet is needed before a cached partial may be considered reusable.", "This matters only with quiet-tail reuse; keep short and leave reuse disabled for reliability."),
    "finalization.decoding_mode": ("Chooses fast full-buffer, accurate full-buffer, or cached quiet-partial finalization.", "Use fast_full_buffer normally; accurate_full_buffer may help difficult audio; avoid reuse_when_quiet for missing endings."),
    "finalization.allow_quiet_tail_reuse": ("Allows finalization to skip full decoding when the latest audio appears quiet.", "Keep disabled to ensure every final word receives a complete-buffer pass."),
    "finalization.worker_wait_seconds": ("Maximum wait for an in-flight live inference call before the final pass proceeds.", "Increase only if logs show finalization waiting on a busy model; 3 seconds is normally ample."),
    "audio.microphone": ("Preferred capture device name used for reconnect and startup selection.", "Select the physical headset microphone rather than a truncated duplicate endpoint."),
    "audio.auto_reconnect": ("Attempts to restore the preferred microphone after disconnect or device changes.", "Keep enabled for an always-running app."),
    "audio.noise_calibration": ("Learns a local noise floor for adaptive speech thresholds.", "Leave disabled until calibrated diagnostics are implemented and verified for your room."),
    "audio.input_gain": ("Multiplies captured samples before endpointing and recognition.", "Raise gradually for a quiet microphone; avoid clipping and prefer hardware/Windows gain when available."),
    "audio.callback_block_ms": ("Requested audio block size delivered by the capture callback.", "60–100 ms balances latency and stability; very small blocks increase scheduling pressure."),
    "audio.preroll_ms": ("Keeps a short ring of microphone audio before official record start so the first phonemes are not clipped.", "300–500 ms is typical; 400 ms is the default. Set 0 to disable ambient pre-roll capture."),
    "integrations.discord_mute_while_recording": ("Isolates Discord's Windows microphone capture session while GlitchScribe is actively recording, then restores its exact prior state.", "Keep enabled to prevent dictation from being heard in a Discord voice call without muting GlitchScribe or the physical microphone. Speak Now stops safely if an active Discord session cannot be isolated."),
    "cleanup.hallucination_filter_enabled": ("Drops known empty-room recognition ghosts such as thank-you-for-watching before insertion.", "Keep enabled; it is deterministic text filtering and never loads another model."),
    "cleanup.hallucination_blocklist": ("Extra ghost phrases to strip when they are the whole utterance or a trailing outro.", "Comma-separated phrases; defaults already cover common YouTube-style hallucinations."),
    "commands.enabled": ("Enables exact spoken command and replacement processing.", "Leave enabled for symbols; commands trigger only on constrained phrase matches."),
    "commands.wait_seconds": ("Waits briefly for a potential command phrase to stabilize.", "Lower for faster symbols; increase if commands execute before the phrase is complete."),
    "commands.literal_mode": ("Treats speech as literal text rather than interpreting command phrases.", "Enable temporarily when dictating words that resemble commands."),
    "commands.confirm_execution": ("Requires confirmation before future commands with broader side effects execute.", "Keep enabled; plain text and symbol replacements do not need intrusive confirmation."),
    "commands.custom_replacements": ("Stores local wrong phrase to intended text corrections for recurring recognition mistakes.", "Enter what GlitchScribe heard and the intended spelling or phrase; save settings to apply it to future dictation."),
    "ui.theme": ("Chooses a polished light appearance, a low-glare dark appearance, or follows the Windows app theme.", "Use dark for the softest low-glare workspace, light for daylight, or system to follow Windows after each app start."),
    "ui.accent_color": ("Changes interactive highlights, selected tabs, progress, and bubble outlines without changing readability colors.", "Violet is the balanced default; cyan, blue, emerald, and rose provide distinct alternatives."),
    "ui.overlay": ("Shows the compact non-activating availability and recognition flyout.", "Enable to see readiness and live recognition without opening settings."),
    "ui.overlay_compact": ("Uses the small flyout rather than an expanded status panel.", "Keep enabled for minimal screen obstruction."),
    "ui.overlay_width": ("Sets the width of the persistent READY/listening control rectangle.", "Drag its lower-right grip for direct resizing, or use 260–500 pixels for a compact control."),
    "ui.overlay_height": ("Sets the height of the branded Glitch Factor READY/listening control.", "Use 92–120 pixels to keep the identity strip, controls, and meter readable; drag the lower-right grip for direct resizing."),
    "ui.microphone_meter": ("Displays local microphone energy while listening.", "Use it to confirm input activity; it never exposes transcript text."),
    "ui.meter_interval_ms": ("Refresh cadence for the meter and endpoint checks.", "60–80 ms feels responsive; lower values add UI and audio polling overhead."),
    "ui.sounds": ("Plays local cues for start, stop, and errors.", "Enable when visual feedback is easy to miss; disable during streams if cues leak into capture."),
    "ui.sound_profile": ("Chooses the local tone recipe used for feedback cues.", "Try Pulse, Chime, Synth Rise, Arcade Pop, Soft Tap, or Silent; all tones are generated locally and carry no product branding."),
    "ui.cursor_fx_enabled": ("Shows the animated cursor halo and event bursts.", "Keep enabled for a visible ring around the pointer while recording and a short burst when a cue fires."),
    "ui.cursor_fx_style": ("Chooses the cursor halo animation style.", "Orbit Halo is the default; Ripple Rings, Comet Sweep, Spectrum Burst, and Minimal Pulse change only status geometry."),
    "ui.cursor_fx_intensity": ("Controls cursor halo and burst strength.", "Use 1.0 normally; lower it for a quieter desktop or increase it for a clearer visual signal."),
    "ui.sound_start": ("Plays the selected cue when recording starts.", "Disable if you want only the visual start burst."),
    "ui.sound_stop": ("Plays the selected cue when recording stops.", "Leave disabled to avoid a second cue before final transcription completes."),
    "ui.sound_complete": ("Plays the selected cue when recording/transcription completes.", "Disable if completion should remain visual-only."),
    "ui.sound_error": ("Plays the selected cue when recording or transcription fails.", "Keep enabled if errors are easy to miss; disable for silent failure feedback."),
    "ui.sound_preview": ("Allows future preview/action events to use the selected cue.", "Leave enabled to make the sound catalog useful for later local UI actions."),
    "ui.microphone_ring_enabled": ("Shows the cursor halo during active microphone capture.", "Disable to keep event bursts while hiding the persistent recording halo."),
    "ui.cursor_fx_reduced_motion": ("Reduces cursor animation motion.", "Use this for motion sensitivity or accessibility; the ring remains visible but orbit movement is removed."),
    "ui.ring_start": ("Cursor ring on recording start", "Disable if start should use sound only; the persistent microphone ring remains separately controlled."),
    "ui.ring_stop": ("Cursor ring on recording stop", "Leave disabled for a quieter transition into final transcription, or enable for immediate stop feedback."),
    "ui.ring_complete": ("Cursor ring on completion", "Disable if completion should use sound or status text only."),
    "ui.ring_error": ("Cursor ring on error", "Keep enabled when an error should be visible without relying on audio."),
    "ui.ring_preview": ("Cursor ring on preview/action events", "Leave enabled for future local preview and action feedback."),
    "ui.streaming_bubble": ("Shows recognized speech in a separate translucent bubble while you talk.", "Keep enabled for visual feedback without putting transcript text in the persistent control rectangle."),
    "ui.bubble_width": ("Sets the width of the temporary translucent speech bubble.", "Use 360–520 pixels for normal chat; increase it if long provisional phrases feel cramped."),
    "ui.bubble_height": ("Sets the height of the temporary speech bubble.", "Use 68–120 pixels for one or two readable lines; increase it when using larger text or longer previews."),
    "ui.bubble_font_size": ("Sets the size of provisional transcript text inside the speech bubble.", "Use 10–14 points for normal viewing or increase it when monitoring from farther away."),
    "ui.bubble_font_family": ("Chooses the typeface used for provisional transcript text.", "Use Segoe UI for a native Windows look, Cascadia Code or Consolas for coding, and Bahnschrift for a compact display style."),
    "ui.bubble_font_weight": ("Chooses regular, semibold, or bold transcript text.", "Use Semibold for the clearest everyday monitoring or Bold when the bubble is small or viewed from farther away."),
    "ui.bubble_visual_style": ("Chooses the bubble surface, outline, and state-color recipe.", "Use Glitch Glass for the branded default, Neon Edge for a stronger coding/stream look, Minimal for low distraction, or High Contrast for maximum legibility."),
    "ui.bubble_text_color_mode": ("Chooses whether transcript text follows the theme, accent, or maximum-contrast color.", "Use Theme for balanced reading, Accent for a stylized display, or Bright when visibility matters most."),
    "ui.bubble_corner_radius": ("Controls how rounded the transcript bubble corners appear.", "Use 16–24 pixels for a soft glass panel, lower values for a technical rectangle, or higher values for a pill-like shape."),
    "ui.bubble_border_width": ("Controls the strength of the transcript bubble outline.", "Use 0 for borderless Minimal, 1 for subtle Glass, or 2–4 for Neon Edge and high-contrast monitoring."),
    "ui.bubble_max_characters": ("Limits how much of the newest provisional transcript remains visible in the bubble.", "Use 120–240 characters for normal dictation; increase it for longer context or reduce it for a quieter display."),
    "ui.bubble_opacity": ("Controls how solid the temporary speech bubble appears.", "0.70–0.85 remains readable without blocking much of the application behind it."),
    "ui.preview_near_caret": ("Anchors the speech bubble near the caret once when recording starts.", "The anchor remains fixed for that recording and does not chase the caret as text is inserted."),
    "ui.preview_offset_pixels": ("Vertical gap between the initial text caret and the speech bubble.", "12–20 pixels usually clears the text line; increase if the bubble covers nearby content."),
    "ui.bubble_offset_x": ("Horizontal offset from the session's caret or mouse anchor to the speech bubble.", "Drag the visible bubble to update this automatically; positive values move it right."),
    "ui.bubble_offset_y": ("Vertical offset from the session's caret or mouse anchor to the speech bubble.", "Drag the visible bubble to update this automatically; negative values move it above the anchor."),
    "action_wheel.enabled": ("Shows the local graphic Action Wheel from right-click or its configured hotkey.", "Keep enabled for Cursor-style prompt upgrades and last-transcript actions; disable if the wheel conflicts with another app."),
    "action_wheel.hotkey": ("The global shortcut that opens or closes the Action Wheel.", "Ctrl + Shift + A is the default; choose a free function key if another application owns the chord."),
    "privacy.transcript_history": ("Would retain private local transcript recovery data between sessions.", "Keep disabled unless you explicitly want recovery history; current diagnostics remain transcript-free."),
    "privacy.raw_audio_retention": ("Would save microphone recordings locally for review or opt-in adaptation.", "Keep disabled unless you explicitly approve storage and cleanup rules."),
    "privacy.runtime_diagnostics": ("Stores timing, counts, target safety, and error metadata without transcript content.", "Keep enabled because it makes reliability bugs diagnosable without recording what you said."),
    "plugins.directory": ("Folder scanned for local Python extension modules.", "Use a project-relative folder and install only code you trust."),
    "plugins.enabled": ("Loads extensions from the configured local plugin directory; plugin code can allocate resources during import.", "Keep disabled unless the Resource Use panel lists the exact trusted plugins you intend to run."),
    "plugins.entry_points": ("Loads installed packages that register GlitchScribe entry points and may execute code at startup.", "Keep disabled unless you explicitly trust and need an installed extension."),
    "compatibility.custom_text_controls": ("Allows UI Automation Custom controls only when they expose supported text patterns.", "Keep enabled for Electron and browser editors; safety checks still reject unknown targets."),
    "compatibility.fail_closed": ("Refuses insertion when focus, integrity, password state, or text support is uncertain.", "Keep enabled; disabling would risk typing into the wrong or unsafe window."),
}


def built_in_settings() -> tuple[SettingSpec, ...]:
    B, N, T, C, P, L = (
        SettingKind.BOOLEAN,
        SettingKind.NUMBER,
        SettingKind.TEXT,
        SettingKind.CHOICE,
        SettingKind.PATH,
        SettingKind.STRING_LIST,
    )
    specs = (
        SettingSpec("general.active_profile", "Active profile", "General", C, "automatic", choices=("automatic", "general", "chat", "document", "terminal", "code", "search")),
        SettingSpec("general.auto_profile", "Choose profile from active application", "General", B, True),
        SettingSpec("general.hotkey", "Global dictation hotkey", "General", T, "Ctrl + Alt + Space"),
        SettingSpec("general.hold_to_talk_enabled", "Enable hold-to-talk", "General", B, False, restart_required=True),
        SettingSpec("general.hold_shortcut", "Hold-to-talk shortcut", "General", C, "F8", choices=("F8", "F9", "F10", "F11", "F12"), restart_required=True),
        SettingSpec("general.hold_latch_modifier", "Latch modifier", "General", C, "Shift", choices=("Shift", "Ctrl", "Alt"), restart_required=True),
        SettingSpec("performance.preset", "Performance preset", "Performance Lab", C, "lean_core", choices=("lean_core", "reliable_stream", "balanced", "instant", "experimental_turbo"), description="Use Lean Core for fast bubble partials plus a full final pass, or pick Instant for even snappier live cadence."),
        SettingSpec("performance.allow_experimental", "Allow experimental speed trade-offs", "Performance Lab", B, False, description="Required before selecting options that can omit final verification."),
        SettingSpec("recognition.model", "Recognition model ID or local path", "Recognition", T, NEMOTRON_MODEL_ID, restart_required=True),
        SettingSpec("model.idle_unload_minutes", "Idle unload after (minutes)", "Recognition", N, 10.0, minimum=0.0, maximum=1440.0, advanced=True, description="0 keeps the model resident; default 10 frees VRAM for Hermes coexistence."),
        SettingSpec("recognition.additional_models", "Additional model IDs or local paths", "Recognition", L, ()),
        SettingSpec("recognition.device", "Compute device", "Recognition", C, "automatic", choices=("automatic", "cuda:0", "cuda:1", "cpu"), restart_required=True),
        SettingSpec("recognition.vram_reserve_mb", "VRAM reserve (MiB)", "Recognition", N, 512, minimum=0, maximum=8192, restart_required=True, advanced=True, presentation="slider", step=256),
        SettingSpec("recognition.compute_type", "Compute precision", "Recognition", C, "float16", choices=("int8", "int8_float16", "float16", "float32"), restart_required=True, advanced=True, presentation="select"),
        SettingSpec("recognition.language", "Recognition language", "Recognition", T, "en"),
        SettingSpec("recognition.transcription_mode", "CrisperWhisper transcription mode", "Recognition", C, "verbatim", choices=("verbatim", "intended")),
        SettingSpec("recognition.crisperwhisper_model_path", "CrisperWhisper local model directory", "Recognition", P, "", advanced=True, description="Local-only model directory; the application never downloads CrisperWhisper weights automatically."),
        SettingSpec("recognition.custom_vocabulary", "Personal vocabulary", "Recognition", L, ()),
        SettingSpec("recognition.two_pass", "Accurate final recognition pass", "Recognition", B, True),
        SettingSpec("ai.cleanup_enabled", "Cloud AI transcript cleanup", "Cloud AI", B, False),
        SettingSpec("ai.cleanup_nearby_context", "Use bounded nearby context for spelling", "Cloud AI", B, True),
        SettingSpec("ai.cleanup_timeout_seconds", "Cleanup timeout", "Cloud AI", N, 20.0, minimum=1.0, maximum=120.0, advanced=True),
        SettingSpec("ai.model", "OpenRouter Nemotron model", "Cloud AI", T, DEFAULT_AUXILIARY_MODEL),
        SettingSpec("resources.allow_auxiliary_models", "Allow OpenRouter requests", "Cloud AI", B, False),
        SettingSpec("streaming.enabled", "Progressive live recognition", "Streaming", B, True),
        SettingSpec("streaming.interval_seconds", "Partial recognition interval", "Streaming", N, 0.18, minimum=0.08, maximum=3.0, advanced=True),
        SettingSpec("streaming.minimum_audio_seconds", "Audio before first pass", "Streaming", N, 0.45, minimum=0.15, maximum=5.0, advanced=True),
        SettingSpec("streaming.minimum_new_audio_seconds", "New audio before another pass", "Streaming", N, 0.14, minimum=0.02, maximum=3.0, advanced=True),
        SettingSpec("streaming.stability_lag_seconds", "Stable-word lag", "Streaming", N, 0.32, minimum=0.0, maximum=3.0, advanced=True),
        SettingSpec("streaming.holdback_words", "Provisional trailing words", "Streaming", N, 1, minimum=0, maximum=12, advanced=True),
        SettingSpec("streaming.timestamp_tolerance_seconds", "Timestamp agreement tolerance", "Streaming", N, 0.45, minimum=0.0, maximum=2.0, advanced=True),
        SettingSpec("streaming.maximum_partial_audio_seconds", "Rolling partial window", "Streaming", N, 45.0, minimum=5.0, maximum=300.0, advanced=True),
        SettingSpec("streaming.quiet_tail_rms_threshold", "Quiet-tail RMS threshold", "Streaming", N, 0.003, minimum=0.00001, maximum=0.1, advanced=True),
        SettingSpec("typing.live_insert", "Type stable words while speaking (advanced)", "Typing", B, False),
        SettingSpec("typing.smart_continuation", "Smart continuation across sessions", "Typing", B, True),
        SettingSpec("typing.continuation_window_seconds", "Continuation memory", "Typing", N, 30.0, minimum=0.0, maximum=600.0),
        SettingSpec("typing.focus_recovery_seconds", "Popup focus recovery window", "Typing", N, 30.0, minimum=1.0, maximum=120.0),
        SettingSpec("typing.automatic_punctuation", "Automatic punctuation", "Typing", B, True),
        SettingSpec("typing.suppress_inferred_trailing_period", "Suppress inferred final period", "Typing", B, True, description="Preserve explicitly spoken punctuation while preventing Whisper from appending an unwanted final period."),
        SettingSpec("typing.spoken_punctuation", "Spoken punctuation commands", "Typing", B, True),
        SettingSpec("typing.trailing_space", "Add a separator after prose", "Typing", B, True),
        SettingSpec("typing.replace_selection", "Replace selected text", "Typing", B, True),
        SettingSpec("insertion.clipboard_fallback", "Clipboard paste fallback", "Typing", B, True, description="Paste via clipboard with restore when Unicode SendInput fails or text exceeds the threshold."),
        SettingSpec("insertion.clipboard_threshold_chars", "Clipboard paste threshold", "Typing", N, 400.0, minimum=1.0, maximum=20000.0, advanced=True),
        SettingSpec("general.paste_last_hotkey", "Paste-last transcript hotkey", "General", T, "Ctrl + Shift + V", description="Secondary chord to re-insert the last successful transcript. Use None to disable."),
        SettingSpec("endpoint.auto_stop", "Stop after speech and silence", "Endpointing", B, True),
        SettingSpec("endpoint.silence_seconds", "Silence before stopping", "Endpointing", N, 3.0, minimum=0.5, maximum=10.0),
        SettingSpec("endpoint.minimum_speech_seconds", "Minimum sustained speech", "Endpointing", N, 0.30, minimum=0.1, maximum=3.0, advanced=True),
        SettingSpec("endpoint.rms_threshold", "Speech RMS threshold", "Endpointing", N, 0.002, minimum=0.00001, maximum=0.1, advanced=True),
        SettingSpec("endpoint.require_recognized_speech", "Require recognized text before auto-stop", "Endpointing", B, True, advanced=True),
        SettingSpec("endpoint.capture_tail_grace_seconds", "Capture drain grace", "Endpointing", N, 0.40, minimum=0.0, maximum=1.0, advanced=True),
        SettingSpec("endpoint.quiet_tail_reuse_seconds", "Reusable quiet tail", "Endpointing", N, 0.05, minimum=0.0, maximum=0.5, advanced=True),
        SettingSpec("finalization.decoding_mode", "Final decoding mode", "Finalization", C, "fast_full_buffer", choices=("fast_full_buffer", "accurate_full_buffer", "reuse_when_quiet"), description="Full-buffer modes protect the last words; reuse_when_quiet is the fastest and least reliable."),
        SettingSpec("finalization.allow_quiet_tail_reuse", "Reuse a quiet rolling partial", "Finalization", B, False, description="Experimental latency optimization. Disabled in Reliable Stream because a quiet audio tail does not prove the final word was recognized."),
        SettingSpec("finalization.worker_wait_seconds", "Wait for live inference before final pass", "Finalization", N, 3.0, minimum=0.1, maximum=15.0, advanced=True),
        SettingSpec("audio.microphone", "Preferred microphone", "Audio", T, "automatic"),
        SettingSpec("audio.auto_reconnect", "Reconnect preferred microphone", "Audio", B, True),
        SettingSpec("audio.noise_calibration", "Adaptive noise calibration", "Audio", B, False),
        SettingSpec("audio.input_gain", "Software input gain", "Audio", N, 1.0, minimum=0.25, maximum=4.0, advanced=True),
        SettingSpec("audio.callback_block_ms", "Capture callback block", "Audio", N, 100.0, minimum=10.0, maximum=500.0, advanced=True, description="Smaller blocks can reduce capture latency but increase callback overhead."),
        SettingSpec("audio.preroll_ms", "Pre-roll buffer", "Audio", N, 400.0, minimum=0.0, maximum=1000.0, advanced=True, description="Milliseconds of ambient PCM prepended when dictation starts (0 disables)."),
        SettingSpec("integrations.discord_mute_while_recording", "Isolate Discord audio while dictating", "Integrations", B, True, description="Silence only Discord's Windows microphone capture session during Speak Now, then restore its exact prior state."),
        SettingSpec("cleanup.hallucination_filter_enabled", "Recognition hallucination blocklist", "Recognition", B, True, description="Strip common empty-room recognition ghosts from final transcripts without another model."),
        SettingSpec("cleanup.hallucination_blocklist", "Extra hallucination phrases", "Recognition", L, (), advanced=True, description="Additional whole-utterance ghost phrases to drop."),
        SettingSpec("commands.enabled", "Voice commands", "Commands", B, True),
        SettingSpec("commands.wait_seconds", "Command stability wait", "Commands", N, 0.6, minimum=0.0, maximum=3.0),
        SettingSpec("commands.literal_mode", "Start in literal mode", "Commands", B, False),
        SettingSpec("commands.confirm_execution", "Confirm command execution", "Commands", B, True),
        SettingSpec("commands.custom_replacements", "Pronunciation and recognition corrections", "Corrections", T, "", presentation="correction_editor"),
        SettingSpec("action_wheel.enabled", "Enable Action Wheel", "Interface", B, True, description="Right-click Control Lab or the configured Action Wheel hotkey opens the local pie menu."),
        SettingSpec("action_wheel.hotkey", "Action Wheel hotkey", "Interface", C, "Ctrl + Shift + A", choices=("Ctrl + Shift + A", "F8", "F9", "F10", "F11", "F12"), restart_required=True, description="Global shortcut that opens or closes the local pie Action Wheel."),
        SettingSpec("ui.theme", "Theme", "Interface", C, "dark", choices=("system", "dark", "light")),
        SettingSpec("ui.accent_color", "Accent color", "Interface", C, "violet", choices=("violet", "cyan", "blue", "emerald", "rose")),
        SettingSpec("ui.overlay", "Floating availability flyout", "Interface", B, True),
        SettingSpec("ui.overlay_compact", "Compact flyout", "Interface", B, True),
        SettingSpec("ui.overlay_width", "Control rectangle width", "Interface", N, 300.0, minimum=260.0, maximum=720.0),
        SettingSpec("ui.overlay_height", "Branded control height", "Interface", N, 92.0, minimum=92.0, maximum=240.0),
        SettingSpec("ui.microphone_meter", "Microphone activity meter", "Interface", B, True),
        SettingSpec("ui.meter_interval_ms", "Meter and endpoint refresh interval", "Interface", N, 80.0, minimum=20.0, maximum=500.0, advanced=True),
        SettingSpec("ui.sounds", "Start, stop, and error sounds", "Interface", B, True),
        SettingSpec("ui.sound_profile", "Sound profile", "Interface", C, "pulse", choices=("pulse", "chime", "synth", "arcade", "soft", "silent")),
        SettingSpec("ui.cursor_fx_enabled", "Cursor halo and event bursts", "Interface", B, True),
        SettingSpec("ui.cursor_fx_style", "Cursor halo style", "Interface", C, "orbit", choices=("orbit", "ripple", "comet", "spectrum", "minimal")),
        SettingSpec("ui.cursor_fx_intensity", "Cursor effect intensity", "Interface", N, 1.0, minimum=0.35, maximum=1.5, presentation="slider", step=0.05),
        SettingSpec("ui.sound_start", "Sound on recording start", "Interface", B, True),
        SettingSpec("ui.sound_stop", "Sound on recording stop", "Interface", B, False),
        SettingSpec("ui.sound_complete", "Sound on completion", "Interface", B, True),
        SettingSpec("ui.sound_error", "Sound on error", "Interface", B, True),
        SettingSpec("ui.sound_preview", "Sound on preview/action events", "Interface", B, True),
        SettingSpec("ui.microphone_ring_enabled", "Persistent microphone halo", "Interface", B, True),
        SettingSpec("ui.cursor_fx_reduced_motion", "Reduce cursor effect motion", "Interface", B, False),
        SettingSpec("ui.ring_start", "Cursor ring on recording start", "Interface", B, True),
        SettingSpec("ui.ring_stop", "Cursor ring on recording stop", "Interface", B, False),
        SettingSpec("ui.ring_complete", "Cursor ring on completion", "Interface", B, True),
        SettingSpec("ui.ring_error", "Cursor ring on error", "Interface", B, True),
        SettingSpec("ui.ring_preview", "Cursor ring on preview/action events", "Interface", B, True),
        SettingSpec("ui.streaming_bubble", "Separate translucent speech bubble", "Speech Bubble", B, True),
        SettingSpec("ui.bubble_width", "Bubble width", "Speech Bubble", N, 420.0, minimum=280.0, maximum=900.0, presentation="slider", step=10.0),
        SettingSpec("ui.bubble_height", "Bubble height", "Speech Bubble", N, 68.0, minimum=56.0, maximum=300.0, presentation="slider", step=4.0),
        SettingSpec("ui.bubble_font_size", "Transcript text size", "Speech Bubble", N, 10.0, minimum=8.0, maximum=28.0, presentation="slider", step=1.0),
        SettingSpec("ui.bubble_font_family", "Transcript font", "Speech Bubble", C, "Segoe UI", choices=("Segoe UI", "Cascadia Code", "Consolas", "Bahnschrift", "Arial")),
        SettingSpec("ui.bubble_font_weight", "Transcript font weight", "Speech Bubble", C, "normal", choices=("normal", "semibold", "bold")),
        SettingSpec("ui.bubble_visual_style", "Bubble visual style", "Speech Bubble", C, "glitch_glass", choices=("minimal", "glitch_glass", "neon_edge", "high_contrast")),
        SettingSpec("ui.bubble_text_color_mode", "Transcript color", "Speech Bubble", C, "theme", choices=("theme", "accent", "bright")),
        SettingSpec("ui.bubble_corner_radius", "Corner radius", "Speech Bubble", N, 20.0, minimum=4.0, maximum=32.0, presentation="slider", step=1.0),
        SettingSpec("ui.bubble_border_width", "Border width", "Speech Bubble", N, 1.0, minimum=0.0, maximum=4.0, presentation="slider", step=1.0),
        SettingSpec("ui.bubble_max_characters", "Visible transcript characters", "Speech Bubble", N, 120.0, minimum=40.0, maximum=1000.0, presentation="slider", step=10.0),
        SettingSpec("ui.bubble_opacity", "Bubble opacity", "Speech Bubble", N, 0.82, minimum=0.35, maximum=0.98, presentation="slider", step=0.01),
        SettingSpec("ui.preview_near_caret", "Anchor bubble near caret at recording start", "Speech Bubble", B, True),
        SettingSpec("ui.preview_offset_pixels", "Caret preview gap", "Speech Bubble", N, 16.0, minimum=0.0, maximum=120.0, advanced=True),
        SettingSpec("ui.bubble_offset_x", "Bubble horizontal anchor offset", "Speech Bubble", N, 18, minimum=-1200, maximum=1200, advanced=True),
        SettingSpec("ui.bubble_offset_y", "Bubble vertical anchor offset", "Speech Bubble", N, -88, minimum=-1200, maximum=1200, advanced=True),
        SettingSpec("privacy.transcript_history", "Private local transcript recovery", "Privacy", B, False),
        SettingSpec("privacy.raw_audio_retention", "Retain raw audio", "Privacy", B, False, sensitive=True),
        SettingSpec("privacy.runtime_diagnostics", "Transcript-free diagnostics", "Privacy", B, True),
        SettingSpec("plugins.directory", "Local plugin directory", "Plugins", P, "plugins"),
        SettingSpec("plugins.enabled", "Enable local plugins", "Plugins", B, False),
        SettingSpec("plugins.entry_points", "Enable installed entry-point plugins", "Plugins", B, False, advanced=True),
        SettingSpec("compatibility.custom_text_controls", "Allow pattern-backed Custom controls", "Compatibility", B, True),
        SettingSpec("compatibility.fail_closed", "Fail closed when target safety is uncertain", "Compatibility", B, True),
    )
    return tuple(
        replace(
            spec,
            description=_SETTING_HELP[spec.key][0],
            guidance=_SETTING_HELP[spec.key][1],
        )
        for spec in specs
    )
