"""
Module for displaying the agent's output on the console.
"""

import logging
import sys
import threading
from datetime import datetime

from colorama import Fore, Style, init

# Initialize colorama
init()

logger = logging.getLogger(__name__)


class ConsoleDisplay:
    """
    Console display for the agent's input and output.
    Provides a more user-friendly display than raw logging.
    """

    def __init__(self):
        """Initialize the console display."""
        self.lock = threading.Lock()
        logger.info("Console display initialized")

    def display_header(self):
        """Display the application header."""
        self._clear_console()
        print(f"{Fore.CYAN}╔{'═' * 78}╗{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}║{Style.RESET_ALL}{Fore.WHITE}{' System Design Interview AI Agent ':^78}{Style.RESET_ALL}{Fore.CYAN}║{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}╚{'═' * 78}╝{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}►{Style.RESET_ALL} {Fore.YELLOW}Listening for interview audio...{Style.RESET_ALL}"
        )
        print()

    def display_detected_audio(self, speaker, confidence):
        """
        Display information about detected audio.

        Args:
            speaker: The detected speaker (interviewer, candidate, or unknown)
            confidence: Confidence level of the speaker detection
        """
        timestamp = self._get_timestamp()

        with self.lock:
            if speaker == "interviewer":
                speaker_text = f"{Fore.GREEN}Interviewer{Style.RESET_ALL}"
            elif speaker == "candidate":
                speaker_text = f"{Fore.BLUE}Candidate{Style.RESET_ALL}"
            else:
                speaker_text = f"{Fore.YELLOW}Unknown{Style.RESET_ALL}"

            confidence_percentage = int(confidence * 100)
            print(
                f"{timestamp} {Fore.CYAN}►{Style.RESET_ALL} Detected {speaker_text} speaking ({confidence_percentage}% confidence)"
            )

    def display_transcription(self, speaker, text):
        """
        Display transcribed text from a speaker.

        Args:
            speaker: The speaker (interviewer, candidate, or agent)
            text: The transcribed text
        """
        timestamp = self._get_timestamp()

        with self.lock:
            if speaker == "interviewer":
                print(f"{timestamp} {Fore.GREEN}Interviewer:{Style.RESET_ALL} {text}")
            elif speaker == "candidate":
                print(f"{timestamp} {Fore.BLUE}Candidate:{Style.RESET_ALL} {text}")
            elif speaker == "agent":
                print(f"{timestamp} {Fore.MAGENTA}AI Agent:{Style.RESET_ALL} {text}")
            else:
                print(f"{timestamp} {Fore.YELLOW}Unknown:{Style.RESET_ALL} {text}")

    def display_thinking(self, message):
        """
        Display a thinking/processing message.

        Args:
            message: The thinking/processing message
        """
        timestamp = self._get_timestamp()

        with self.lock:
            print(f"{timestamp} {Fore.CYAN}[Processing]{Style.RESET_ALL} {message}")

    def display_diagram_generated(self, diagram_path, diagram_type):
        """
        Display information about a generated diagram.

        Args:
            diagram_path: Path to the generated diagram
            diagram_type: Type of the diagram (high-level or low-level)
        """
        timestamp = self._get_timestamp()

        with self.lock:
            print(
                f"{timestamp} {Fore.CYAN}[Generated]{Style.RESET_ALL} {Fore.YELLOW}{diagram_type}{Style.RESET_ALL} diagram at {Fore.GREEN}{diagram_path}{Style.RESET_ALL}"
            )

    def display_system_design(self, title, components_count):
        """
        Display information about a generated system design.

        Args:
            title: Title of the system design
            components_count: Number of components in the design
        """
        timestamp = self._get_timestamp()

        with self.lock:
            print(
                f"{timestamp} {Fore.CYAN}[Design]{Style.RESET_ALL} System design '{Fore.YELLOW}{title}{Style.RESET_ALL}' with {components_count} components"
            )

    def display_error(self, error_message):
        """
        Display an error message.

        Args:
            error_message: The error message
        """
        timestamp = self._get_timestamp()

        with self.lock:
            print(f"{timestamp} {Fore.RED}[Error]{Style.RESET_ALL} {error_message}")

    def display_status(self, status_message):
        """
        Display a status message.

        Args:
            status_message: The status message
        """
        timestamp = self._get_timestamp()

        with self.lock:
            print(f"{timestamp} {Fore.CYAN}[Status]{Style.RESET_ALL} {status_message}")

    def _get_timestamp(self):
        """Get a formatted timestamp for display."""
        return f"{Fore.LIGHTBLACK_EX}[{datetime.now().strftime('%H:%M:%S')}]{Style.RESET_ALL}"

    def _clear_console(self):
        """Clear the console screen."""
        # This is a cross-platform way to clear the console
        if sys.platform.startswith("win"):
            # For Windows
            _ = os.system("cls")
        else:
            # For Linux/macOS
            _ = os.system("clear")
