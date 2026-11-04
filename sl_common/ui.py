"""How capture scripts talk to the operator: the console by default, the GUI from capture_gui.py.

A capture's run(args, ui) calls these instead of print()/input(), so the same code drives both.
"""


class Stopped(Exception):
    """Raised by check_stop() when the operator asks the capture to stop."""


class ConsoleUI:
    def log(self, msg):
        print(msg)

    def prompt(self, msg):
        """Block until the operator has done something by hand (e.g. switched the LED)."""
        input(f"{msg} Press Enter to continue... ")

    def frame(self, frames, label):
        """A capture just arrived (uint16 frame stack); the GUI previews it."""

    def progress(self, done, total):
        pass

    def check_stop(self):
        """Raise Stopped if the operator asked to stop. Called between captures."""
