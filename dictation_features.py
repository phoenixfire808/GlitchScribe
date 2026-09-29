from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class VoiceCommand:
    kind: str
    value: Optional[str] = None


@dataclass(frozen=True)
class CommitDecision:
    text_to_insert: str
    committed_text: str
    conflict: bool = False


class CommandDebouncer:
    def __init__(self, wait_seconds: float = 0.6, clock=time.monotonic):
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.clock = clock
        self._candidate: Optional[VoiceCommand] = None
        self._candidate_text = ""
        self._started_at = 0.0

    def reset(self) -> None:
        self._candidate = None
        self._candidate_text = ""
        self._started_at = 0.0

    def update(self, text: str) -> Optional[VoiceCommand]:
        command = parse_voice_command(text)
        normalized = _normalized_command(text)
        if command is None:
            self.reset()
            return None
        now = self.clock()
        if command != self._candidate or normalized != self._candidate_text:
            self._candidate = command
            self._candidate_text = normalized
            self._started_at = now
            return command if self.wait_seconds == 0 else None
        if now - self._started_at >= self.wait_seconds:
            return command
        return None


_STOP_COMMANDS = {
    "pause voice typing",
    "pause dictation",
    "stop voice typing",
    "stop dictation",
    "stop listening",
    "stop dictating",
    "stop voice mode",
    "pause voice mode",
}
_DELETE_COMMANDS = {"delete that", "erase that", "scratch that"}
_SELECT_COMMANDS = {"select that"}
_UNDO_COMMANDS = {"undo that", "revert"}
_KEY_COMMANDS = {
    "press enter": "enter",
    "enter": "enter",
    "backspace": "backspace",
    "press backspace": "backspace",
    "tab": "tab",
    "press tab": "tab",
    "insert space": "space",
    "press space": "space",
}
_SIMPLE_COMMANDS = {
    "redo that": VoiceCommand("redo"),
    "cut that": VoiceCommand("cut_that"),
    "copy that": VoiceCommand("copy_that"),
    "paste": VoiceCommand("paste"),
    "bold that": VoiceCommand("format_that", "bold"),
    "italicize that": VoiceCommand("format_that", "italic"),
    "underline that": VoiceCommand("format_that", "underline"),
    "capitalize that": VoiceCommand("shape_that", "capitalize"),
    "uppercase that": VoiceCommand("shape_that", "uppercase"),
    "lowercase that": VoiceCommand("shape_that", "lowercase"),
    "no space that": VoiceCommand("shape_that", "no_space"),
    "correct that": VoiceCommand("correct_that"),
    "spell that": VoiceCommand("spell_that"),
    "spell out": VoiceCommand("spell_out"),
    "capture selection": VoiceCommand("capture_selection"),
    "explain selection": VoiceCommand("explain_selection"),
}
_IDENTIFIER_COMMANDS = {
    "snake case ": "snake",
    "camel case ": "camel",
    "pascal case ": "pascal",
    "kebab case ": "kebab",
    "constant case ": "constant",
}


def _normalized_command(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip().lower())
    return value.rstrip(".,!?;:")


def _render_identifier(text: str, style: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).casefold())
    if not words:
        return ""
    if style == "snake":
        return "_".join(words)
    if style == "kebab":
        return "-".join(words)
    if style == "constant":
        return "_".join(words).upper()
    titled = [word[:1].upper() + word[1:] for word in words]
    if style == "pascal":
        return "".join(titled)
    if style == "camel":
        return words[0] + "".join(titled[1:])
    return ""


def parse_voice_command(text: str) -> Optional[VoiceCommand]:
    compact = re.sub(r"\s+", " ", str(text).strip())
    compact_folded = compact.casefold()
    for prefix, kind in (
        ("literal translate selection to ", "translate_selection_literal"),
        ("translate selection to ", "translate_selection_natural"),
    ):
        if compact_folded.startswith(prefix) and len(compact) > len(prefix):
            language = compact[len(prefix) :].strip().rstrip(".,!?;:")
            if 0 < len(language) <= 80:
                return VoiceCommand(kind, language)
    for prefix, kind in (
        ("edit selection ", "edit_selection"),
        ("compile prompt ", "compile_prompt"),
        ("snippet ", "snippet"),
    ):
        if compact_folded.startswith(prefix) and len(compact) > len(prefix):
            payload = compact[len(prefix) :]
            if kind != "edit_selection" or len(payload) <= 2_000:
                return VoiceCommand(kind, payload)

    normalized = _normalized_command(text)
    if normalized in _STOP_COMMANDS:
        return VoiceCommand("stop")
    if normalized in _DELETE_COMMANDS:
        return VoiceCommand("delete_that")
    if normalized in _SELECT_COMMANDS:
        return VoiceCommand("select_that")
    if normalized in _UNDO_COMMANDS:
        return VoiceCommand("undo")
    if normalized in _KEY_COMMANDS:
        return VoiceCommand("key", _KEY_COMMANDS[normalized])
    if normalized in _SIMPLE_COMMANDS:
        return _SIMPLE_COMMANDS[normalized]

    for prefix, style in _IDENTIFIER_COMMANDS.items():
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            identifier = _render_identifier(normalized[len(prefix) :], style)
            if identifier:
                return VoiceCommand("insert", identifier)

    for prefix in ("type ", "dictate "):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return VoiceCommand("insert", normalized[len(prefix) :])
    if normalized.startswith("caps ") and len(normalized) > 5:
        return VoiceCommand("insert", normalized[5:].title())
    if normalized.startswith("no space ") and len(normalized) > 9:
        return VoiceCommand("insert", re.sub(r"\s+", "", normalized[9:]))
    return None


# Longest aliases are applied first. These are the English names documented by
# Microsoft Voice Typing and Voice Access, plus common singular variants.
_PUNCTUATION_ALIASES = {
    "double ampersand": " && ",
    "double dash": " --",
    "back slash": "\\",
    "vertical bar": " | ",
    "pipe": " | ",
    "slash": "/",
    "begin single quote": "‘",
    "open single quote": "‘",
    "end single quote": "'",
    "close single quote": "'",
    "left square bracket": "[",
    "open square bracket": "[",
    "right square bracket": "]",
    "close square bracket": "]",
    "left curly brace": "{",
    "open curly brace": "{",
    "right curly brace": "}",
    "close curly brace": "}",
    "left angle bracket": "<",
    "open angle bracket": "<",
    "right angle bracket": ">",
    "close angle bracket": ">",
    "left parenthesis": "(",
    "open parenthesis": "(",
    "left parentheses": "(",
    "open parentheses": "(",
    "right parenthesis": ")",
    "close parenthesis": ")",
    "right parentheses": ")",
    "close parentheses": ")",
    "vertical bar character": "|",
    "vertical bar sign": "|",
    "pipe character": "|",
    "registered sign": "®",
    "copyright sign": "©",
    "copyright mark": "©",
    "paragraph sign": "¶",
    "paragraph mark": "¶",
    "section sign": "§",
    "degree symbol": "°",
    "degree sign": "°",
    "pound sterling sign": "£",
    "multiplication sign": "×",
    "division sign": "÷",
    "plus or minus sign": "±",
    "greater than sign": ">",
    "less than sign": "<",
    "exclamation mark": "!",
    "exclamation point": "!",
    "question mark": "?",
    "number sign": "#",
    "pound sign": "#",
    "dollar sign": "$",
    "percent sign": "%",
    "and sign": "&",
    "forward slash": "/",
    "at sign": "@",
    "at mention": "@",
    "caret symbol": "^",
    "back quote": "`",
    "back tick": "`",
    "backquote": "`",
    "backtick": "`",
    "tilde symbol": "~",
    "open quotes": '“',
    "close quotes": '”',
    "euro sign": "€",
    "yen sign": "¥",
    "plus sign": "+",
    "equal sign": "=",
    "minus sign": "−",
    "heart emoji": "<3",
    "frowny face": ":(",
    "smiley face": ":)",
    "winky face": ";)",
    "apostrophe s": "'s",
    "apostrophe-s": "'s",
    "new paragraph": "\n\n",
    "next paragraph": "\n\n",
    "new line": "\n",
    "next line": "\n",
    "full stop": ".",
    "dot dot dot": "...",
    "n dash": "–",
    "n-dash": "–",
    "m dash": "—",
    "m-dash": "—",
    "hyphen": "-",
    "asterisk": "*",
    "comma": ",",
    "period": ".",
    "ellipsis": "...",
    "ellipses": "...",
    "colon": ":",
    "semicolon": ";",
    "backslash": "\\",
    "underscore": "_",
    "ampersand": "&",
}


def parse_custom_replacements(specification: str) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for item in re.split(r"[;\n]+", str(specification or "")):
        if "=" not in item:
            continue
        phrase, replacement = item.split("=", 1)
        phrase = re.sub(r"\s+", " ", phrase.strip().lower())
        if phrase:
            replacements[phrase] = replacement.strip()
    return replacements


def serialize_custom_replacements(replacements: Mapping[str, str]) -> str:
    """Serialize local correction mappings in the existing one-per-line format."""
    return "\n".join(
        f"{phrase}={replacement}"
        for phrase, replacement in replacements.items()
        if str(phrase).strip()
    )


def _validated_custom_replacement(heard: str, intended: str) -> tuple[str, str]:
    phrase = re.sub(r"\s+", " ", str(heard or "").strip().lower())
    replacement = str(intended or "").strip()
    if not phrase:
        raise ValueError("Enter what GlitchScribe heard.")
    if not replacement:
        raise ValueError("Enter what GlitchScribe should write instead.")
    if any(delimiter in phrase for delimiter in ("=", ";", "\n", "\r")):
        raise ValueError("The misunderstood phrase cannot contain =, ;, or a line break.")
    if any(delimiter in replacement for delimiter in (";", "\n", "\r")):
        raise ValueError("The intended text cannot contain ; or a line break.")
    if len(phrase) > 200 or len(replacement) > 500:
        raise ValueError(
            "Keep corrections under 200 heard characters and 500 intended characters."
        )
    return phrase, replacement


def upsert_custom_replacement(
    specification: str,
    heard: str,
    intended: str,
) -> str:
    """Add or update one persistent wrong-to-intended local correction."""
    phrase, replacement = _validated_custom_replacement(heard, intended)
    replacements = parse_custom_replacements(specification)
    replacements[phrase] = replacement
    return serialize_custom_replacements(replacements)


def remove_custom_replacement(specification: str, heard: str) -> str:
    """Remove one persistent local correction by its normalized heard phrase."""
    phrase = re.sub(r"\s+", " ", str(heard or "").strip().lower())
    replacements = parse_custom_replacements(specification)
    replacements.pop(phrase, None)
    return serialize_custom_replacements(replacements)


def _combined_spoken_replacements(
    custom_replacements: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    replacements = dict(_PUNCTUATION_ALIASES)
    if custom_replacements:
        replacements.update(
            {
                re.sub(r"\s+", " ", str(phrase).strip().lower()): str(replacement)
                for phrase, replacement in custom_replacements.items()
                if str(phrase).strip()
            }
        )
    return replacements


def spoken_replacement_is_exact(
    text: str,
    custom_replacements: Optional[Mapping[str, str]] = None,
) -> bool:
    normalized = _normalized_command(text)
    return normalized in _combined_spoken_replacements(custom_replacements)


def transcript_needs_trailing_separator(
    prepared_text: str,
    *,
    exact_literal: bool = False,
) -> bool:
    if exact_literal:
        return False
    value = str(prepared_text or "")
    if not value or value.endswith(("\n", " ", "\t")):
        return False
    return not value.rstrip().endswith(("/", "\\", "@", "$", "_", "-", "|", "&", "="))


def render_spoken_punctuation(
    text: str,
    custom_replacements: Optional[Mapping[str, str]] = None,
) -> str:
    rendered = re.sub(r"\s+", " ", text.strip())
    placeholders: list[tuple[str, str]] = []
    for index, (phrase, symbol) in enumerate(
        sorted(
            _combined_spoken_replacements(custom_replacements).items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    ):
        marker = f"\ue000{index}\ue001"
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)(?:[.,!?])?"
        updated, count = re.subn(pattern, marker, rendered, flags=re.IGNORECASE)
        if count:
            rendered = updated
            placeholders.append((marker, symbol))
    for marker, symbol in placeholders:
        rendered = rendered.replace(marker, symbol)

    rendered = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*", "\n\n", rendered)
    rendered = re.sub(r"[ \t]*\n[ \t]*", "\n", rendered)
    rendered = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", rendered)
    rendered = re.sub(r"([\(\[\{])\s+", r"\1", rendered)
    rendered = re.sub(r"\s*(['’]s)\b", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\s*([@/_\\])\s*", r"\1", rendered)
    rendered = re.sub(r"[ \t]*\|[ \t]*", " | ", rendered)
    rendered = re.sub(r"[ \t]*&&[ \t]*", " && ", rendered)
    rendered = re.sub(r"(?<!-)\s*([-–—])\s*(?!-)", r"\1", rendered)
    rendered = re.sub(r"--[ \t]+(?=\w)", "--", rendered)
    rendered = re.sub(
        r"\.\s+(com|org|net|edu|gov|io|ai|co|us|uk)\b",
        r".\1",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip()


def strip_automatic_punctuation(text: str) -> str:
    stripped = re.sub(r"[.!?,;:…]+", "", text)
    return re.sub(r"\s+", " ", stripped).strip()


def suppress_inferred_trailing_period(
    prepared_text: str,
    recognized_text: str,
    *,
    enabled: bool = True,
) -> str:
    """Remove Whisper's sentence-final period unless the speaker named it.

    Internal dots are preserved, so shell paths, filenames, versions, and URLs
    remain usable. Question marks, exclamation marks, and ellipses are not
    changed.
    """
    prepared = str(prepared_text or "").rstrip()
    if not enabled or not prepared or not prepared.endswith("."):
        return prepared
    if prepared.endswith("..."):
        return prepared
    raw = re.sub(r"\s+", " ", str(recognized_text or "").strip())
    if re.search(r"\b(?:period|full stop)\s*[.!?]*$", raw, flags=re.IGNORECASE):
        return prepared
    return prepared[:-1].rstrip()


def mask_profanity(text: str, terms: Iterable[str]) -> str:
    rendered = text
    for term in sorted({item.strip() for item in terms if item.strip()}, key=len, reverse=True):
        rendered = re.sub(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            lambda match: "*" * len(match.group(0)),
            rendered,
            flags=re.IGNORECASE,
        )
    return rendered


def prepare_transcript(
    text: str,
    *,
    automatic_punctuation: bool = True,
    profanity_terms: Iterable[str] = (),
    custom_replacements: Optional[Mapping[str, str]] = None,
) -> str:
    prepared = text if automatic_punctuation else strip_automatic_punctuation(text)
    prepared = render_spoken_punctuation(prepared, custom_replacements)
    return mask_profanity(prepared, profanity_terms)


@dataclass(frozen=True)
class ContextualTranscript:
    text: str
    prefix: str = ""


def contextualize_transcript(
    text: str,
    preceding_text: str = "",
    *,
    protected_terms: Iterable[str] = (),
) -> ContextualTranscript:
    """Shape a new dictation fragment for the text immediately before the caret."""
    fragment = str(text or "")
    before = str(preceding_text or "")
    if not fragment or not before:
        return ContextualTranscript(fragment, "")

    meaningful_before = before.rstrip()
    if not meaningful_before:
        return ContextualTranscript(fragment, "")

    first_character = fragment[0]
    no_space_before = set(",.;:!?%)]}'’”/@_\\-–—")
    no_space_after = set("([{‘“/@_\\-–—")
    prefix = ""
    if (
        not before[-1].isspace()
        and first_character not in no_space_before
        and meaningful_before[-1] not in no_space_after
    ):
        prefix = " "

    sentence_boundary = meaningful_before.endswith((".", "!", "?", "\n"))
    if sentence_boundary:
        return ContextualTranscript(fragment, prefix)

    word_match = re.search(r"[^\W\d_][\w'’.-]*", fragment, flags=re.UNICODE)
    if not word_match:
        return ContextualTranscript(fragment, prefix)
    first_word = word_match.group(0)
    protected = first_word == "I" or (
        len(first_word) > 1 and first_word.isupper()
    )
    if not protected:
        lowered_fragment = fragment.casefold()
        for term in protected_terms:
            candidate = str(term).strip()
            if candidate and lowered_fragment.startswith(candidate.casefold()):
                boundary = len(candidate)
                if boundary == len(fragment) or not fragment[boundary].isalnum():
                    protected = True
                    break
    if protected:
        return ContextualTranscript(fragment, prefix)

    index = word_match.start()
    fragment = fragment[:index] + fragment[index].lower() + fragment[index + 1 :]
    return ContextualTranscript(fragment, prefix)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\n\n|\n|[\w]+(?:['’][\w]+)?|[^\w\s]", text, flags=re.UNICODE)


def _format_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    result = ""
    closing = set(",.;:!?%)]}’”)\n")
    opening = set("([{‘“")
    tight = {"@", "/", "\\", "_", "-", "–", "—"}
    for index, token in enumerate(tokens):
        if token == "\n\n":
            result = result.rstrip() + "\n\n"
            continue
        if token == "\n":
            result = result.rstrip() + "\n"
            continue
        previous = tokens[index - 1] if index else ""
        no_space = (
            not result
            or token in closing
            or token in tight
            or previous in opening
            or previous in tight
            or result.endswith(("\n", "\n\n"))
            or (previous == "." and "@" in tokens[:index])
        )
        if not no_space:
            result += " "
        result += token
    return result


class LiveCommitter:
    def __init__(self, holdback_words: int = 3):
        if holdback_words < 0:
            raise ValueError("holdback_words must be non-negative")
        self.holdback_words = holdback_words
        self._previous_tokens: list[str] = []
        self._committed_tokens: list[str] = []

    @property
    def committed_text(self) -> str:
        return _format_tokens(self._committed_tokens)

    def reset(self) -> None:
        self._previous_tokens = []
        self._committed_tokens = []

    def _prepare(self, text: str) -> list[str]:
        return _tokens(render_spoken_punctuation(text))

    @staticmethod
    def _semantic_word(token: str) -> Optional[str]:
        if re.fullmatch(r"[\w]+(?:['’][\w]+)?", token, flags=re.UNICODE):
            return token.casefold()
        return None

    def _format_compatible_final(self, current: list[str]) -> Optional[CommitDecision]:
        """Append only the new tail when earlier words differ in formatting only."""
        allowed_formatting = {",", ".", ";", ":", "!", "?", "…"}
        if any(
            self._semantic_word(token) is None and token not in allowed_formatting
            for token in self._committed_tokens
        ):
            return None
        committed_words = [
            word
            for token in self._committed_tokens
            if (word := self._semantic_word(token)) is not None
        ]
        if not committed_words:
            return None

        current_words: list[str] = []
        boundary = 0
        for index, token in enumerate(current):
            word = self._semantic_word(token)
            if word is not None:
                current_words.append(word)
                if len(current_words) == len(committed_words):
                    boundary = index + 1
                    break
            elif token not in allowed_formatting:
                return None
        if current_words != committed_words:
            return None

        suffix = current[boundary:]
        return self._decision_for(self._committed_tokens + suffix)

    def _decision_for(self, new_committed: list[str]) -> CommitDecision:
        before = _format_tokens(self._committed_tokens)
        after = _format_tokens(new_committed)
        delta = after[len(before) :] if after.startswith(before) else ""
        self._committed_tokens = list(new_committed)
        return CommitDecision(delta, after, conflict=False)

    def update(self, partial_text: str) -> CommitDecision:
        current = self._prepare(partial_text)
        committed_count = len(self._committed_tokens)
        if current[:committed_count] != self._committed_tokens:
            self._previous_tokens = current
            return CommitDecision("", self.committed_text, conflict=True)

        common = 0
        for previous, latest in zip(self._previous_tokens, current):
            if previous != latest:
                break
            common += 1
        stable_count = max(committed_count, common - self.holdback_words)
        decision = self._decision_for(current[:stable_count])
        self._previous_tokens = current
        return decision

    def accept_stable(self, stable_text: str) -> CommitDecision:
        """Commit text already stabilized by timestamp agreement upstream."""
        current = self._prepare(stable_text)
        committed_count = len(self._committed_tokens)
        if current[:committed_count] != self._committed_tokens:
            self._previous_tokens = current
            return CommitDecision("", self.committed_text, conflict=True)
        self._previous_tokens = current
        return self._decision_for(current)

    def finalize(self, final_text: str) -> CommitDecision:
        current = self._prepare(final_text)
        committed_count = len(self._committed_tokens)
        if current[:committed_count] != self._committed_tokens:
            compatible = self._format_compatible_final(current)
            if compatible is not None:
                return compatible
            return CommitDecision("", self.committed_text, conflict=True)
        self._previous_tokens = current
        return self._decision_for(current)
