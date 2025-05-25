# main_gui.py

import logging
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk  # Keep messagebox import
from typing import Any, Dict, List, Optional  # Added Optional

import numpy as np
import torch  # Keep if VAD uses it
from dotenv import load_dotenv
from ttkthemes import ThemedTk  # Keep if used

# Load environment variables first
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from modules.ai_processor import AIProcessor
from modules.audio_capture import AudioCapture
from modules.diagram_generator import DiagramGenerator
from modules.gui import AgentGUI  # AgentGUI is in modules.gui
from modules.speaker_diarization import SpeakerDiarization
from modules.speech_recognition import SpeechRecognition

# --- Centralized Logging Setup (Assuming this is your main entry point) ---
log_file_name = "agent_main_gui_debug.log"
# Ensure logging is configured only once
if not logging.getLogger().hasHandlers():  # Check if root logger already configured
    logging.basicConfig(
        level=logging.INFO,  # Adjusted default level for less noise, can be DEBUG for dev
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_name, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
logger = logging.getLogger(__name__)  # Logger for this file
# Silence noisy libraries
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("pydub").setLevel(logging.INFO)
logging.getLogger("torchaudio").setLevel(logging.INFO)
logging.getLogger("speechbrain").setLevel(logging.INFO)
logging.getLogger("sounddevice").setLevel(logging.INFO)
logging.getLogger("pyaudiowpatch").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(
    logging.DEBUG
)  # OpenAI can be verbose for debugging
logging.getLogger("httpx").setLevel(logging.INFO)
# --- End of Logging Setup ---


class SystemDesignAgentGUI:
    """Main agent class that integrates all components with the GUI."""

    def __init__(self, root_tk_instance: tk.Tk):
        logger.info("Initializing SystemDesignAgentGUI...")

        self.root = root_tk_instance
        self.gui = AgentGUI(self.root, agent=self)  # Pass self to AgentGUI

        self.after_id_process_audio: Optional[str] = None

        self._initialization_failed = False
        try:
            self.audio_capture = AudioCapture(
                chunk_duration=0.2, sample_rate=16000
            )  # VAD works well with 16kHz
            self.speaker_diarization = SpeakerDiarization()
            self.speech_recognition = SpeechRecognition()
            self.ai_processor = AIProcessor()
            self.diagram_generator = DiagramGenerator()
        except ValueError as e:
            logger.critical(
                f"Critical Initialization Error (check .env): {e}", exc_info=True
            )
            if hasattr(self.gui, "display_error"):
                self.gui.display_error(
                    f"Initialization Error: {e}\nPlease check .env file and restart."
                )
            if hasattr(self.gui, "start_button"):
                self.gui.start_button.config(state=tk.DISABLED)
            self._initialization_failed = True
            return
        except Exception as e:
            logger.critical(
                f"A critical error occurred during component initialization: {e}",
                exc_info=True,
            )
            if hasattr(self.gui, "display_error"):
                self.gui.display_error(
                    f"Critical Initialization Error: {e}\nAgent may not function correctly."
                )
            if hasattr(self.gui, "start_button"):
                self.gui.start_button.config(state=tk.DISABLED)
            self._initialization_failed = True
            return

        self.is_running = False
        self.interview_context: List[Dict[str, Any]] = []
        self.current_speaker = "unknown"

        # VAD components
        self.vad_buffer: Dict[str, List[np.ndarray]] = {}
        self.vad_speech_started: Dict[str, bool] = {}
        self.vad_silence_timeout_seconds = 1.2  # Tuned: How long to wait in silence
        self.vad_min_speech_duration_ms = 250  # Min length for a *final* utterance
        self.vad_last_speech_time: Dict[str, float] = {}

        # << --- FIX: Store the target device for VAD --- >>
        self.vad_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SystemDesignAgentGUI: VAD will use device: {self.vad_device}")

        self.vad_model: Optional[torch.nn.Module] = (
            None  # Can be torch.jit.ScriptModule
        )
        self.vad_iterator: Any = None
        self._initialize_vad()

        self._refresh_and_populate_devices()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("SystemDesignAgentGUI initialized with VAD and close handler.")

    def on_closing(self):
        logger.info(
            "WM_DELETE_WINDOW: Close button clicked. Initiating clean shutdown."
        )
        if hasattr(self.gui, "shutdown_gui_updates") and callable(
            self.gui.shutdown_gui_updates
        ):
            logger.info("on_closing: Calling self.gui.shutdown_gui_updates().")
            self.gui.shutdown_gui_updates()
        else:
            logger.warning(
                "on_closing: self.gui.shutdown_gui_updates method not found."
            )

        if self.is_running:
            logger.info(
                "on_closing: Agent is running, calling self.stop_agent_processing()."
            )
            self.stop_agent_processing()
        else:
            logger.info("on_closing: Agent was not running.")
            if (
                hasattr(self, "audio_capture")
                and self.audio_capture
                and self.audio_capture.is_capturing
            ):
                logger.info(
                    "on_closing: Agent not running, but audio_capture is active. Stopping audio_capture."
                )
                self.audio_capture.stop()

        if self.after_id_process_audio:
            logger.info(
                f"on_closing: Ensuring _process_audio is cancelled (ID: {self.after_id_process_audio})."
            )
            try:
                if self.root.winfo_exists():
                    self.root.after_cancel(self.after_id_process_audio)
            except tk.TclError as e:
                logger.debug(f"on_closing: TclError cancelling _process_audio: {e}")
            except Exception as e_cancel:
                logger.error(
                    f"on_closing: Error cancelling _process_audio: {e_cancel}",
                    exc_info=True,
                )
            self.after_id_process_audio = None

        if hasattr(self.gui, "destroy_matplotlib_resources") and callable(
            self.gui.destroy_matplotlib_resources
        ):
            logger.info("on_closing: Destroying GUI's Matplotlib resources.")
            self.gui.destroy_matplotlib_resources()

        logger.info(
            "on_closing: Allowing a brief moment (0.2s) for final cleanup before destroying root."
        )
        time.sleep(0.2)

        logger.info("on_closing: Calling root.quit() to stop mainloop.")
        if self.root.winfo_exists():
            self.root.quit()  # Attempt to make mainloop return

        logger.info("on_closing: Destroying Tkinter root window.")
        if self.root.winfo_exists():
            self.root.destroy()
        logger.info(
            "on_closing: Shutdown sequence fully complete from SystemDesignAgentGUI."
        )

    def _initialize_vad(self):
        try:
            # << --- FIX: Use self.vad_device for logging --- >>
            logger.info(f"Loading Silero VAD model to device: {self.vad_device}...")
            if (
                not hasattr(self, "audio_capture")
                or self.audio_capture.target_sample_rate != 16000
            ):
                logger.warning(
                    f"AudioCapture target SR is {getattr(self.audio_capture, 'target_sample_rate', 'N/A')}Hz. "
                    "Silero VAD expects 16kHz. Ensure AudioCapture provides 16kHz for VAD."
                )

            # << --- FIX: Load model to CPU first, then move --- >>
            vad_model_cpu, loaded_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,  # Added trust_repo
            )
            self.vad_model = vad_model_cpu.to(self.vad_device)  # Move to target device

            if isinstance(loaded_utils, tuple) and len(loaded_utils) >= 4:
                VADIterator_class = loaded_utils[3]  # Usually get_vad_iterator
                self.vad_iterator = VADIterator_class(
                    self.vad_model, threshold=0.5
                )  # Pass the moved model
            elif hasattr(loaded_utils, "VADIterator"):
                self.vad_iterator = loaded_utils.VADIterator(
                    self.vad_model, threshold=0.5
                )  # Pass the moved model
            else:
                logger.error(
                    f"Failed to unpack VADIterator from torch.hub.load utils. Type: {type(loaded_utils)}"
                )
                self.vad_model = None  # Ensure model is None if iterator fails
                raise AttributeError(
                    "Could not find VADIterator in loaded Silero VAD utilities."
                )

            logger.info(
                f"Silero VAD model moved to {self.vad_device} and VADIterator initialized successfully."
            )

        except AttributeError as ae:  # For VADIterator issues
            logger.error(
                f"Failed to initialize VADIterator: {ae}. VAD segmentation disabled.",
                exc_info=True,
            )
            self.vad_model, self.vad_iterator = None, None
            if hasattr(self, "gui"):
                self.gui.display_error(
                    f"Failed to init VAD tools: {ae}. Speech segmentation impaired."
                )
        except Exception as e:  # For general torch.hub.load or other issues
            logger.error(
                f"Failed to load Silero VAD model: {e}. VAD segmentation disabled.",
                exc_info=True,
            )
            self.vad_model, self.vad_iterator = None, None
            if hasattr(self, "gui"):
                self.gui.display_error(
                    "Failed to load Voice Activity Detection model. Speech segmentation impaired."
                )

    def _reset_vad_for_speaker(self, speaker_label: str):
        logger.debug(f"Resetting VAD state for speaker: {speaker_label}")
        self.vad_buffer[speaker_label] = []
        self.vad_speech_started[speaker_label] = False
        # self.vad_last_speech_time[speaker_label] = 0 # Don't reset, managed by timeout logic
        if self.vad_iterator:
            self.vad_iterator.reset_states()
        logger.debug(
            f"VAD state reset for {speaker_label}. Buffer cleared, speech_started set to False."
        )

    def _refresh_and_populate_devices(self):
        if self._initialization_failed:
            logger.warning(
                "_refresh_and_populate_devices: Aborted due to prior init failure."
            )
            return
        try:
            if hasattr(self, "audio_capture") and self.audio_capture:
                devices = self.audio_capture.get_device_list()
                if (
                    hasattr(self, "gui")
                    and self.gui
                    and hasattr(self.gui, "update_queue")
                ):
                    self.gui.update_queue.put(("devices", devices))
                else:
                    logger.warning(
                        "GUI or GUI update queue not available for device list update."
                    )
            else:
                logger.warning(
                    "Audio capture module not available for refreshing devices."
                )
        except Exception as e:
            logger.error(f"Error refreshing devices: {e}", exc_info=True)
            if hasattr(self, "gui") and self.gui:
                self.gui.display_error(f"Failed to refresh audio devices: {e}")

    def run(self):
        if self._initialization_failed:
            logger.error(
                "SystemDesignAgentGUI.run(): Initialization failed. Cannot start GUI."
            )
            # Simple error display if GUI elements aren't fully up
            if not (
                self.root and self.root.winfo_exists()
            ):  # Check if root even exists
                err_root = tk.Tk()
                err_root.withdraw()
                messagebox.showerror(
                    "Fatal Init Error", "Agent components failed. Check logs."
                )
                err_root.destroy()
            elif hasattr(self.gui, "display_error"):  # If GUI is somewhat up
                self.gui.display_error("Agent components failed. Check logs.")
            return
        logger.info("Starting GUI main loop via SystemDesignAgentGUI.run()...")
        if hasattr(self.gui, "start") and callable(self.gui.start):
            self.gui.start()  # This typically calls self.root.mainloop()
        else:
            logger.error("GUI object or its start method is not available.")

    def start_agent_processing(
        self, candidate_device_index=None, interviewer_device_index=None
    ) -> bool:
        if self._initialization_failed:
            logger.error("Start Aborted: Initialization failed.")
            return False
        if not all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in [
                "audio_capture",
                "speech_recognition",
                "ai_processor",
                "diagram_generator",
                "gui",
            ]
        ):
            logger.error("Start Aborted: Agent components not fully initialized.")
            if hasattr(self.gui, "display_error"):
                self.gui.display_error("Agent components not fully initialized.")
                return False
        if candidate_device_index is None or interviewer_device_index is None:
            if hasattr(self.gui, "display_error"):
                self.gui.display_error("Devices must be selected.")
                return False

        logger.info(
            f"Starting agent processing: Cand Idx {candidate_device_index}, Int Idx {interviewer_device_index}"
        )
        logger.debug("Resetting state for new session...")
        self.interview_context = []
        if hasattr(self, "speaker_diarization") and self.speaker_diarization:
            self.speaker_diarization.reset_calibration()

        # Reset VAD states for all potential speakers that might have been tracked
        for speaker_key in list(self.vad_buffer.keys()):
            self._reset_vad_for_speaker(speaker_key)
        self.vad_buffer.clear()
        self.vad_speech_started.clear()
        self.vad_last_speech_time.clear()
        if self.vad_iterator:
            self.vad_iterator.reset_states()  # Global VAD iterator reset

        if hasattr(self.gui, "update_queue"):
            self.gui.update_queue.put(("clear_chat", {}))
            self.gui.update_queue.put(("clear_diagram", {}))

        self.is_running = True  # Set running flag
        if hasattr(self.gui, "update_status"):
            self.gui.update_status("Agent starting audio capture...")

        try:
            self.audio_capture.start(
                candidate_device_index=candidate_device_index,
                interviewer_device_index=interviewer_device_index,
                online_interview=True,
            )
            if (
                self.after_id_process_audio
            ):  # Cancel any previous pending _process_audio calls
                try:
                    if self.root.winfo_exists():
                        self.root.after_cancel(self.after_id_process_audio)
                except tk.TclError:
                    pass  # Can happen if root is closing
                self.after_id_process_audio = None

            if self.root.winfo_exists():  # Schedule the first call to _process_audio
                self.after_id_process_audio = self.root.after(
                    100, self._process_audio
                )  # Start polling for audio
                logger.info(
                    f"Audio capture initiated. _process_audio scheduled (ID: {self.after_id_process_audio})."
                )
                if hasattr(self.gui, "update_status"):
                    self.gui.update_status("Agent running. Listening...")
                return True
            else:  # Root window gone before scheduling
                logger.error("Root window gone before scheduling _process_audio.")
                self.is_running = False
                if hasattr(self, "audio_capture") and self.audio_capture.is_capturing:
                    self.audio_capture.stop()
                return False
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}", exc_info=True)
            if hasattr(self.gui, "display_error"):
                self.gui.display_error(f"Failed to start audio: {e}")
            self.is_running = False
            if hasattr(self, "audio_capture") and self.audio_capture.is_capturing:
                try:
                    self.audio_capture.stop()
                except Exception as stop_e:
                    logger.error(f"Error stopping audio after start failure: {stop_e}")
            return False

    def stop_agent_processing(self):
        logger.info("Attempting to stop agent processing...")
        self.is_running = False  # Primary flag to stop loops

        if self.after_id_process_audio:
            logger.debug(
                f"Stopping: Cancelling _process_audio (ID: {self.after_id_process_audio})"
            )
            try:
                if self.root.winfo_exists():
                    self.root.after_cancel(self.after_id_process_audio)
            except tk.TclError as e:
                logger.warning(
                    f"TclError cancelling _process_audio in stop: {e} (can be normal if root destroying)"
                )
            except Exception as e_cancel:
                logger.error(
                    f"Error cancelling _process_audio in stop: {e_cancel}",
                    exc_info=True,
                )
            self.after_id_process_audio = None  # Clear the ID

        if hasattr(self, "audio_capture") and self.audio_capture:
            try:
                logger.info("Calling audio_capture.stop()...")
                self.audio_capture.stop()
                logger.info("Audio capture stopped.")
            except Exception as e:
                logger.error(f"Error during audio_capture.stop(): {e}", exc_info=True)

        if hasattr(self.gui, "update_status"):
            self.gui.update_status("Agent stopped")
        logger.info("Agent processing and audio capture signaled to stop.")

    def _execute_stop_async(self):
        thread_id = threading.get_ident()
        logger.info(
            f"Agent Async Stop Thread [{thread_id}]: Executing agent stop processing."
        )
        self.stop_agent_processing()
        if hasattr(self.gui, "update_queue"):
            self.gui.update_queue.put(("set_stopped_ui_state", {}))
            self.gui.update_queue.put(
                ("status", {"text": "Agent successfully stopped."})
            )
        logger.info(f"Agent Async Stop Thread [{thread_id}]: Stop processing complete.")

    def _process_audio(self):
        if not self.is_running:
            logger.debug("_process_audio: Agent not running. Loop terminated.")
            self.after_id_process_audio = None
            return

        try:
            audio_data_dict = self.audio_capture.get_audio_tensor()
            if audio_data_dict:
                # Update audio visualization if GUI elements are valid
                if "audio" in audio_data_dict and audio_data_dict["audio"] is not None:
                    if hasattr(self.gui, "update_audio_visualization") and callable(
                        self.gui.update_audio_visualization
                    ):
                        if (
                            self.gui.root.winfo_exists()
                            and hasattr(self.gui, "canvas")
                            and self.gui.canvas
                            and hasattr(self.gui.canvas, "get_tk_widget")
                            and self.gui.canvas.get_tk_widget().winfo_exists()
                        ):
                            self.gui.update_audio_visualization(
                                audio_data_dict["audio"]
                            )
                        else:
                            logger.debug(
                                "Skipping audio viz update, GUI root or canvas widget not ready/exists."
                            )

                # Process the audio chunk with VAD for speech segmentation and transcription
                self._process_audio_chunk_with_vad(audio_data_dict)
        except Exception as e:
            logger.error(f"Error in _process_audio loop: {e}", exc_info=True)
        finally:
            if (
                self.is_running and self.root.winfo_exists()
            ):  # Check if root still valid before rescheduling
                self.after_id_process_audio = self.root.after(
                    100, self._process_audio
                )  # Reschedule
            else:
                logger.info(
                    f"_process_audio: Not rescheduling (is_running: {self.is_running}, root_exists: {self.root.winfo_exists() if hasattr(self.root, 'winfo_exists') else 'N/A'})."
                )
                self.after_id_process_audio = None  # Clear the ID if not rescheduling

    def _prepare_audio_bytes_for_stt(self, audio_np_mono: np.ndarray) -> bytes:
        """Prepares a mono numpy audio array (float32) into WAV-formatted bytes for STT."""
        if audio_np_mono is None or audio_np_mono.size == 0:
            logger.debug("STT_PREPARE: Empty audio segment.")
            return b""

        processed_audio = audio_np_mono.astype(np.float32)

        max_abs_val = np.max(np.abs(processed_audio))
        if max_abs_val > 1.0 + 1e-6:  # Add a small epsilon for float comparisons
            logger.debug(
                f"STT_PREPARE: Normalizing audio with max_abs_val {max_abs_val:.3f}"
            )
            processed_audio = processed_audio / max_abs_val
        elif max_abs_val == 0:
            logger.debug("STT_PREPARE: Pure silence segment.")
            return b""

        audio_int16 = (processed_audio * 32767).astype(np.int16)

        # Create WAV-formatted bytes in memory
        import wave
        from io import BytesIO

        wav_buffer = BytesIO()
        try:
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit (since audio_int16 is np.int16)
                wf.setframerate(
                    self.audio_capture.target_sample_rate
                )  # Use actual capture rate
                wf.writeframes(audio_int16.tobytes())

            wav_bytes = wav_buffer.getvalue()
            logger.debug(
                f"STT_PREPARE: Successfully created WAV-formatted bytes, size: {len(wav_bytes)}"
            )
            return wav_bytes
        except Exception as e:
            logger.error(f"STT_PREPARE: Error creating WAV bytes: {e}", exc_info=True)
            return b""
        finally:
            wav_buffer.close()

    def _process_audio_chunk_with_vad(self, audio_data_dict: dict):
        """
        Processes an audio chunk from AudioCapture using Voice Activity Detection (VAD).
        It buffers speech segments and processes them after a defined period of silence.
        """
        try:
            audio_np_raw = audio_data_dict.get("audio")
            captured_speaker_label = audio_data_dict.get("speaker")

            if (
                audio_np_raw is None
                or audio_np_raw.size == 0
                or not captured_speaker_label
            ):
                return

            self.current_speaker = captured_speaker_label

            # << --- ADDED LOG 1 --- >>
            current_time_ms = int(time.time() * 1000)
            max_val_raw_chunk = (
                np.max(np.abs(audio_np_raw)) if audio_np_raw.size > 0 else 0
            )
            logger.debug(
                f"VAD_PIPE [{current_time_ms}ms] Speaker: {self.current_speaker}, "
                f"Received AudioCapture chunk, raw_max_abs: {max_val_raw_chunk:.4f}, "
                f"size: {audio_np_raw.shape}"
            )

            if not self.vad_model or not self.vad_iterator:
                logger.debug(
                    f"VAD: VAD model/iterator not available. Processing chunk directly for {self.current_speaker}."
                )
                audio_for_fixed_chunk = audio_np_raw
                if audio_for_fixed_chunk.ndim == 2:
                    if audio_for_fixed_chunk.shape[1] == 1:
                        audio_for_fixed_chunk = audio_for_fixed_chunk.squeeze(axis=1)
                    elif audio_for_fixed_chunk.shape[0] == 1:
                        audio_for_fixed_chunk = audio_for_fixed_chunk.squeeze(axis=0)
                    else:
                        audio_for_fixed_chunk = np.mean(audio_for_fixed_chunk, axis=1)
                if audio_for_fixed_chunk.dtype != np.float32:
                    audio_for_fixed_chunk = audio_for_fixed_chunk.astype(np.float32)
                    max_abs = np.max(np.abs(audio_for_fixed_chunk))
                    if max_abs > 1.0:
                        audio_for_fixed_chunk /= max_abs
                self._process_fixed_chunk_for_stt(
                    audio_for_fixed_chunk, self.current_speaker
                )
                return

            audio_for_vad_input_chunk = audio_np_raw
            if audio_for_vad_input_chunk.ndim == 2:
                if audio_for_vad_input_chunk.shape[1] == 1:
                    audio_for_vad_input_chunk = audio_for_vad_input_chunk.squeeze(
                        axis=1
                    )
                elif audio_for_vad_input_chunk.shape[0] == 1:
                    audio_for_vad_input_chunk = audio_for_vad_input_chunk.squeeze(
                        axis=0
                    )
                else:
                    audio_for_vad_input_chunk = np.mean(
                        audio_for_vad_input_chunk, axis=1
                    )

            if audio_for_vad_input_chunk.dtype != np.float32:
                audio_for_vad_input_chunk = audio_for_vad_input_chunk.astype(np.float32)
                max_abs_val = np.max(np.abs(audio_for_vad_input_chunk))
                if max_abs_val > 1.0:
                    audio_for_vad_input_chunk = audio_for_vad_input_chunk / max_abs_val

            if (
                self.current_speaker not in self.vad_buffer
                or self.current_speaker not in self.vad_speech_started
                or self.current_speaker not in self.vad_last_speech_time
            ):
                self._reset_vad_for_speaker(self.current_speaker)

            samples_in_input_chunk = len(audio_for_vad_input_chunk)
            vad_window_size_samples = 512

            # << --- ADDED LOG 2 --- >>
            logger.debug(
                f"VAD_PIPE [{current_time_ms}ms] Speaker: {self.current_speaker}, "
                f"Processing VAD windows. speech_started: {self.vad_speech_started.get(self.current_speaker, False)}, "
                f"buffer_len: {len(self.vad_buffer.get(self.current_speaker, []))}"
            )

            for i in range(0, samples_in_input_chunk, vad_window_size_samples):
                window_end = min(i + vad_window_size_samples, samples_in_input_chunk)
                audio_window_np_original = audio_for_vad_input_chunk[i:window_end]

                if len(audio_window_np_original) == 0:
                    continue

                audio_window_for_vad = audio_window_np_original
                if len(audio_window_for_vad) < vad_window_size_samples:
                    padding = np.zeros(
                        vad_window_size_samples - len(audio_window_for_vad),
                        dtype=np.float32,
                    )
                    audio_window_for_vad = np.concatenate(
                        (audio_window_for_vad, padding)
                    )

                # << --- FIX: Use self.vad_device --- >>
                audio_tensor_for_vad_window = torch.from_numpy(audio_window_for_vad).to(
                    self.vad_device
                )

                if audio_tensor_for_vad_window.ndim > 1:
                    audio_tensor_for_vad_window = audio_tensor_for_vad_window.squeeze()
                if audio_tensor_for_vad_window.ndim == 0:
                    audio_tensor_for_vad_window = audio_tensor_for_vad_window.unsqueeze(
                        0
                    )

                speech_info_in_window = self.vad_iterator(
                    audio_tensor_for_vad_window, return_seconds=False
                )
                current_window_has_speech = bool(speech_info_in_window)

                # << --- ADDED LOG 3 --- >>
                window_max_abs = (
                    np.max(np.abs(audio_window_np_original))
                    if audio_window_np_original.size > 0
                    else 0
                )
                logger.debug(
                    f"VAD_WINDOW [{current_time_ms}ms] Speaker: {self.current_speaker}, "
                    f"Win_idx: {i//vad_window_size_samples}, has_speech: {current_window_has_speech}, "
                    f"win_max_abs: {window_max_abs:.4f}, "
                    f"speech_info: {speech_info_in_window}"
                )

                if current_window_has_speech:
                    if not self.vad_speech_started.get(self.current_speaker, False):
                        logger.info(
                            f"VAD: Speech START detected for '{self.current_speaker}'"
                        )
                        self.vad_speech_started[self.current_speaker] = True
                        if hasattr(self.gui, "update_queue"):
                            self.gui.update_queue.put(
                                ("speaker", {"speaker": self.current_speaker})
                            )

                    self.vad_buffer.setdefault(self.current_speaker, []).append(
                        audio_window_np_original
                    )
                    self.vad_last_speech_time[self.current_speaker] = time.time()

                elif self.vad_speech_started.get(self.current_speaker, False):
                    self.vad_buffer.setdefault(self.current_speaker, []).append(
                        audio_window_np_original
                    )

                    silence_duration = time.time() - self.vad_last_speech_time.get(
                        self.current_speaker, time.time()
                    )
                    if silence_duration >= self.vad_silence_timeout_seconds:
                        logger.info(
                            f"VAD: Silence TIMEOUT for '{self.current_speaker}' "
                            f"({silence_duration:.2f}s >= {self.vad_silence_timeout_seconds}s). Processing segment."
                        )
                        self._process_vad_segment(self.current_speaker)
                        self._reset_vad_for_speaker(self.current_speaker)
                        break

            # After iterating all small VAD windows within the current AudioCapture chunk:
            if self.vad_speech_started.get(
                self.current_speaker, False
            ) and self.vad_buffer.get(
                self.current_speaker
            ):  # Check if buffer has content

                current_silence_for_speaker = (
                    time.time()
                    - self.vad_last_speech_time.get(self.current_speaker, time.time())
                )
                # << --- ADDED LOG 4 --- >>
                logger.debug(
                    f"VAD_PIPE_END [{current_time_ms}ms] Speaker: {self.current_speaker}, "
                    f"End of AudioCapture chunk. Current silence: {current_silence_for_speaker:.2f}s. "
                    f"speech_started: {self.vad_speech_started.get(self.current_speaker, False)}, "
                    f"buffer_len: {len(self.vad_buffer.get(self.current_speaker, []))}"
                )

                if current_silence_for_speaker >= self.vad_silence_timeout_seconds:
                    logger.info(
                        f"VAD: End of AudioCapture chunk check for '{self.current_speaker}' at {current_time_ms}ms, "
                        f"silence TIMEOUT met ({current_silence_for_speaker:.2f}s). Processing segment."
                    )
                    self._process_vad_segment(
                        self.current_speaker
                    )  # This will log internally
                    self._reset_vad_for_speaker(self.current_speaker)

        except Exception as e:
            logger.error(
                f"Error in _process_audio_chunk_with_vad for speaker '{audio_data_dict.get('speaker', 'N/A')}': {e}",
                exc_info=True,
            )
            current_processing_speaker = audio_data_dict.get("speaker")
            if current_processing_speaker:
                self._reset_vad_for_speaker(current_processing_speaker)

    def _process_vad_segment(self, speaker_label: str):
        """Concatenates buffered audio for a speaker, checks duration, and sends for STT."""
        buffered_audio_list = self.vad_buffer.get(speaker_label)
        if not buffered_audio_list:
            # logger.debug(f"VAD: No buffered audio to process for '{speaker_label}'.")
            return

        num_buffered_windows = len(buffered_audio_list)
        if num_buffered_windows == 0:
            # logger.debug(f"VAD: Buffer for '{speaker_label}' is empty (list of len 0).")
            return

        logger.info(
            f"VAD: Concatenating speech segment for '{speaker_label}' from {num_buffered_windows} VAD windows."
        )
        try:
            complete_speech_segment_np = np.concatenate(buffered_audio_list)
        except ValueError as ve:  # Should not happen if list contains only np.ndarrays
            logger.error(
                f"VAD: Error concatenating audio for '{speaker_label}': {ve}. Buffer items: {[type(b) for b in buffered_audio_list]}",
                exc_info=True,
            )
            return

        segment_duration_samples = len(complete_speech_segment_np)
        if segment_duration_samples == 0:
            logger.info(
                f"VAD: Concatenated segment for '{speaker_label}' is empty. Discarding."
            )
            return

        segment_duration_ms = (
            segment_duration_samples / self.audio_capture.target_sample_rate
        ) * 1000

        logger.info(
            f"VAD_SEGMENT: Concatenated speech segment for '{speaker_label}' from {num_buffered_windows} VAD windows. Duration: {segment_duration_ms:.0f}ms."
        )  # Enhanced log

        logger.info(
            f"VAD: Segment for '{speaker_label}' has duration {segment_duration_ms:.0f}ms."
        )
        if segment_duration_ms < self.vad_min_speech_duration_ms:
            logger.info(
                f"VAD_SEGMENT: Segment for '{speaker_label}' too short ({segment_duration_ms:.0f}ms vs min {self.vad_min_speech_duration_ms}ms). Discarding."
            )
            return

        logger.info(
            f"VAD_SEGMENT: Segment for '{speaker_label}' ({segment_duration_ms:.0f}ms) meets minimum. Proceeding to STT."
        )  # Clear indication
        self._process_fixed_chunk_for_stt(complete_speech_segment_np, speaker_label)

    def _process_fixed_chunk_for_stt(
        self, audio_np_segment: np.ndarray, speaker_label: str
    ):
        """Processes a finalized audio segment (numpy array) for STT and AI."""
        try:
            if audio_np_segment is None or audio_np_segment.size == 0:
                logger.debug(
                    f"STT: Empty audio segment for '{speaker_label}', skipping."
                )
                return

            # Prepare audio bytes (e.g. normalize, convert to int16 bytes)
            audio_bytes_for_stt = self._prepare_audio_bytes_for_stt(audio_np_segment)
            if not audio_bytes_for_stt:
                logger.debug(
                    f"STT: Audio bytes preparation failed or resulted in empty bytes for '{speaker_label}'."
                )
                return

            # Perform Speech-to-Text
            transcribed_text = self.speech_recognition.transcribe(audio_bytes_for_stt)
            if not transcribed_text or not transcribed_text.strip():
                logger.info(
                    f"STT: No transcription result or empty text for '{speaker_label}'."
                )
                return

            # Update GUI and context
            if hasattr(self.gui, "update_chat"):
                self.gui.update_chat(speaker_label, transcribed_text)

            self.interview_context.append(
                {
                    "speaker": speaker_label,
                    "text": transcribed_text,
                    "timestamp": time.time(),
                }
            )
            logger.info(
                f"{speaker_label.capitalize()} (VAD Segment): {transcribed_text}"
            )

            # If the speaker is the interviewer, process with AI
            # Ensure 'interviewer' is the exact string used by speaker diarization/calibration for the interviewer role
            is_primary_speaker_for_ai = (
                speaker_label
                == self.speaker_diarization.speaker_map.get(
                    "interviewer_raw_id", "interviewer"
                )
            ) or (
                speaker_label == "interviewer"
            )  # Fallback if raw_id mapping is complex

            if is_primary_speaker_for_ai:
                if hasattr(self.gui, "update_status"):
                    self.gui.update_status("AI Processing query...")

                # Create a copy of the context for the thread to prevent race conditions if context is modified
                context_copy = list(self.interview_context)

                # Offload AI processing to a separate thread to keep GUI responsive
                ai_thread = threading.Thread(
                    target=self._handle_ai_processing_and_diagrams,
                    args=(transcribed_text, context_copy),
                    daemon=True,
                    name=f"AIProc-{transcribed_text[:10]}",
                )
                ai_thread.start()
        except Exception as e:
            logger.error(
                f"Error processing fixed chunk STT/AI for '{speaker_label}': {e}",
                exc_info=True,
            )
            if hasattr(self.gui, "display_error"):
                self.gui.display_error(
                    f"Error in segment STT/AI for {speaker_label}: {e}"
                )

    def _handle_ai_processing_and_diagrams(
        self, query_text: str, context_history: List[Dict[str, Any]]
    ):
        try:
            thread_id = threading.get_ident()
            logger.info(
                f"AI Thread [{thread_id}]: Processing query: '{query_text[:50]}...'"
            )

            if not hasattr(self, "ai_processor") or not self.ai_processor:
                logger.error(
                    f"AI Thread [{thread_id}]: AI Processor component is missing."
                )
                if hasattr(self.gui, "update_queue"):
                    self.gui.update_queue.put(
                        (
                            "error_message_box",
                            {"message": "AI Processor component is missing."},
                        )
                    )
                    self.gui.update_queue.put(
                        ("status", {"text": "Error: AI Processor missing."})
                    )
                return

            response = self.ai_processor.process_query(
                query_text, context_history
            )  # Assumes AIProcessor has updated prompt
            logger.info(
                f"AI Thread [{thread_id}]: AI processing complete. Response received: {response.keys() if isinstance(response, dict) else 'Non-dict response'}"
            )

            if hasattr(self.gui, "update_queue"):
                self.gui.update_queue.put(("status", {"text": "AI response received."}))

            if response.get("generate_diagram"):
                diagram_type = response.get("diagram_type", "high_level")
                diagram_content = response.get("diagram_content", {})
                title = diagram_content.get(
                    "title", "System Diagram"
                )  # Extract title from content

                logger.info(
                    f"AI Thread [{thread_id}]: Diagram generation requested: Type='{diagram_type}', Title='{title}'"
                )
                if hasattr(self.gui, "update_queue"):
                    self.gui.update_queue.put(
                        (
                            "status",
                            {"text": f"Generating {diagram_type} diagram: {title}"},
                        )
                    )

                if not hasattr(self, "diagram_generator") or not self.diagram_generator:
                    logger.error(
                        f"AI Thread [{thread_id}]: Diagram Generator component is missing."
                    )
                    if hasattr(self.gui, "update_queue"):
                        self.gui.update_queue.put(
                            (
                                "error_message_box",
                                {"message": "Diagram Generator missing."},
                            )
                        )
                        self.gui.update_queue.put(
                            ("status", {"text": "Error: Diagram Generator missing."})
                        )
                else:
                    diagram_path = self.diagram_generator.generate_diagram(
                        diagram_content, diagram_type
                    )

                    if diagram_path:
                        logger.info(
                            f"AI Thread [{thread_id}]: Diagram generated successfully at {diagram_path}"
                        )

                        design_details_for_gui = {
                            "title": diagram_content.get("title", "System Design"),
                            "image_path": diagram_path,
                            "main_explanation": response.get(
                                "text", "No textual explanation provided by AI."
                            ),
                            "api_specifications": response.get("api_specifications"),
                            "data_models": response.get("data_models"),
                            "quantitative_analysis": response.get(
                                "quantitative_analysis"
                            ),
                            "security_architecture": response.get(
                                "security_architecture"
                            ),
                            "tradeoff_analysis_summary": response.get(
                                "tradeoff_analysis_summary"
                            ),
                            "failure_analysis_summary": response.get(
                                "failure_analysis_summary"
                            ),
                            # <<<< ADD THIS >>>>
                            "diagram_content_components": diagram_content.get(
                                "components", []
                            ),
                            # <<<< END OF ADDITION >>>>
                        }

                        if hasattr(self.gui, "update_queue"):
                            # Send the enriched details to the GUI
                            self.gui.update_queue.put(
                                ("diagram_and_details", design_details_for_gui)
                            )  # New queue type

                            status_msg = f"Diagram '{design_details_for_gui['title']}' displayed."
                            if "design_file_path" in response:
                                status_msg += f" Design spec saved: {os.path.basename(str(response['design_file_path']))}"
                            self.gui.update_queue.put(("status", {"text": status_msg}))
                    else:
                        logger.warning(
                            f"AI Thread [{thread_id}]: Diagram generation failed (returned None path)."
                        )
                        if hasattr(self.gui, "update_queue"):
                            self.gui.update_queue.put(
                                (
                                    "error_message_box",
                                    {
                                        "message": f"Failed to generate '{diagram_type}' diagram for '{title}'. Check logs and Graphviz installation."
                                    },
                                )
                            )

            # Handle main textual response from AI (for chat display)
            # The 'text' field from the response is the main explanation.
            # It's already part of design_details_for_gui.main_explanation for the details panel.
            # Decide if you also want to put this full explanation in the chat.
            # For brevity, we could put a summary or skip if it's in details.
            # Or, keep it for a complete chat log. Let's assume we still want it in chat for now.
            response_text_for_chat = response.get("text", "")
            if response_text_for_chat:
                logger.info(
                    f"AI Thread [{thread_id}]: Agent Chat Response (from main 'text' field): '{response_text_for_chat[:100]}...'"
                )
                self.interview_context.append(
                    {
                        "speaker": "agent",
                        "text": response_text_for_chat,
                        "timestamp": time.time(),
                    }
                )
                if hasattr(self.gui, "update_queue"):
                    self.gui.update_queue.put(
                        ("chat", {"speaker": "agent", "text": response_text_for_chat})
                    )

            if hasattr(self.gui, "update_queue"):
                self.gui.update_queue.put(("status", {"text": "Ready. Listening..."}))

        except Exception as e:
            thread_id = threading.get_ident()  # Get thread_id again if not already set
            logger.error(
                f"AI Thread [{thread_id}]: Error during AI processing or diagram generation: {e}",
                exc_info=True,
            )
            if hasattr(self.gui, "update_queue"):
                self.gui.update_queue.put(
                    ("error_message_box", {"message": f"AI Processing Error: {e}"})
                )
                self.gui.update_queue.put(
                    ("status", {"text": "Error during AI processing."})
                )


def main():
    logger.info("Application starting (main_gui.py with VAD)...")
    root_tk_instance = None
    agent_app_instance = None  # To have it in scope for finally block if needed
    try:
        # Attempt to use a themed Tk instance
        try:
            root_tk_instance = ThemedTk(theme="plastik")
            # Some themes might need explicit setting for toplevels too
            root_tk_instance.set_theme("plastik", toplevel=True, themebg=True)
        except tk.TclError:
            logger.warning(
                "ThemedTk 'plastik' theme not found or failed to apply, using standard tk.Tk()."
            )
            root_tk_instance = tk.Tk()

        agent_app_instance = SystemDesignAgentGUI(root_tk_instance)

        if agent_app_instance._initialization_failed:
            logger.critical(
                "Main: Agent initialization failed. The application will now exit."
            )
            # If root exists, destroy it. Otherwise, it might not have been created.
            if root_tk_instance and root_tk_instance.winfo_exists():
                root_tk_instance.destroy()
            return  # Exit if initialization failed

        agent_app_instance.run()  # This should call root.mainloop() via gui.start()

    except Exception as e:
        logger.critical(
            f"Fatal error in GUI application main function: {e}", exc_info=True
        )
        error_message = f"A fatal application error occurred: {e}\nPlease check the log file: {log_file_name}"
        # Try to show an error message box, even if the main GUI is broken
        try:
            # Check if the GUI's root window or the agent's GUI instance is usable
            if not (
                agent_app_instance
                and hasattr(agent_app_instance, "gui")
                and hasattr(agent_app_instance.gui, "root")
                and agent_app_instance.gui.root.winfo_exists()
            ):
                # If main GUI structure is not available, create a temporary root for messagebox
                temp_error_root = tk.Tk()
                temp_error_root.withdraw()  # Hide the temp root window
                messagebox.showerror(
                    "Fatal Application Error", error_message, parent=None
                )  # No parent if main root is gone
                temp_error_root.destroy()
            elif (
                agent_app_instance
                and hasattr(agent_app_instance, "gui")
                and hasattr(agent_app_instance.gui, "display_error")
            ):
                # If AgentGUI's display_error is available, use it
                agent_app_instance.gui.display_error(error_message)
            else:  # Fallback if specific display_error not available but root might exist
                messagebox.showerror(
                    "Fatal Application Error",
                    error_message,
                    parent=(
                        root_tk_instance
                        if root_tk_instance and root_tk_instance.winfo_exists()
                        else None
                    ),
                )

        except Exception as display_err_ex:
            # If even showing the messagebox fails, print to stderr
            logger.error(
                f"Failed to display the fatal error message box: {display_err_ex}"
            )
            print(f"FATAL APPLICATION ERROR: {error_message}", file=sys.stderr)
    finally:
        logger.info(
            "Application main function's 'finally' block reached. Process should exit if all non-daemon threads are done."
        )


if __name__ == "__main__":
    main()
