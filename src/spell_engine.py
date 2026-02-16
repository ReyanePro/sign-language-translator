"""
Spell Engine
Converts detected letters into words in real-time.
Handles letter confirmation, word building, and basic auto-correction.
"""

import time


class SpellEngine:
    def __init__(self, hold_time=1.0, cooldown=0.5):
        """
        Args:
            hold_time: seconds a letter must be held to be confirmed
            cooldown: seconds to wait between letter confirmations
        """
        self.hold_time = hold_time
        self.cooldown = cooldown

        self.current_word = ""
        self.words = []
        self.current_letter = None
        self.letter_start_time = None
        self.last_confirm_time = 0
        self.confirmed = False

        # Callbacks
        self.on_letter_confirmed = None
        self.on_word_completed = None

    def update(self, letter, confidence):
        """
        Update with a new detected letter.
        Returns dict with current state.
        """
        now = time.time()
        result = {
            "current_letter": letter,
            "confidence": confidence,
            "hold_progress": 0.0,
            "confirmed": False,
            "current_word": self.current_word,
            "full_text": self.get_full_text(),
        }

        if letter is None:
            self.current_letter = None
            self.letter_start_time = None
            self.confirmed = False
            return result

        # Cooldown check
        if now - self.last_confirm_time < self.cooldown:
            return result

        # Same letter being held
        if letter == self.current_letter and not self.confirmed:
            elapsed = now - self.letter_start_time
            progress = min(elapsed / self.hold_time, 1.0)
            result["hold_progress"] = progress

            if elapsed >= self.hold_time:
                # Letter confirmed!
                self.current_word += letter
                self.confirmed = True
                self.last_confirm_time = now
                result["confirmed"] = True
                result["current_word"] = self.current_word
                result["full_text"] = self.get_full_text()

                if self.on_letter_confirmed:
                    self.on_letter_confirmed(letter)

        elif letter != self.current_letter:
            # New letter detected
            self.current_letter = letter
            self.letter_start_time = now
            self.confirmed = False

        return result

    def add_space(self):
        """Add a space (complete current word)."""
        if self.current_word:
            self.words.append(self.current_word)
            if self.on_word_completed:
                self.on_word_completed(self.current_word)
            self.current_word = ""

    def backspace(self):
        """Delete last letter."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.words:
            self.current_word = self.words.pop()

    def clear(self):
        """Clear everything."""
        self.current_word = ""
        self.words = []
        self.current_letter = None
        self.letter_start_time = None
        self.confirmed = False

    def get_full_text(self):
        """Get the complete text (all words + current word)."""
        parts = self.words.copy()
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def get_display_text(self):
        """Get text formatted for display with cursor."""
        text = self.get_full_text()
        return text + "▌"
