"""
Main application for the System Design Interview AI Agent.
This orchestrates all components of the application.
"""

import logging
import os
import tempfile
import threading
import time
import uuid
import wave

import numpy as np
from dotenv import load_dotenv

from modules.ai_processor import AIProcessor
from modules.audio_capture import AudioCapture
from modules.diagram_generator import DiagramGenerator
from modules.speaker_diarization import SpeakerDiarization
from modules.speech_recognition import SpeechRecognition

# Configure specific loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Keep application logger at DEBUG

# Disable matplotlib debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)  # Also silence Pillow logs

# Set levels for our own modules to maintain debugging capabilities
logging.getLogger("modules.audio_capture").setLevel(logging.DEBUG)
logging.getLogger("modules.speaker_diarization").setLevel(logging.DEBUG)
logging.getLogger("modules.speech_recognition").setLevel(logging.DEBUG)
logging.getLogger("modules.ai_processor").setLevel(logging.DEBUG)
logging.getLogger("modules.diagram_generator").setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()


class SystemDesignAgent:
    """Main agent class that orchestrates all components."""

    def __init__(self):
        """Initialize the agent and its components."""
        logger.info("Initializing System Design Interview AI Agent...")

        # Initialize components
        self.audio_capture = AudioCapture()
        self.speaker_diarization = SpeakerDiarization()
        self.audio_capture.diarization = self.speaker_diarization
        self.speech_recognition = SpeechRecognition()
        self.ai_processor = AIProcessor()
        self.diagram_generator = DiagramGenerator()

        # State variables
        self.is_running = False
        self.interview_context = []
        self.current_speaker = None

        logger.info("Agent initialized successfully")

    def start(self):
        """Start the agent."""
        logger.info("Starting the agent...")
        self.is_running = True

        # Start audio capture in a separate thread
        self.audio_thread = threading.Thread(target=self._process_audio_stream)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        try:
            # Keep the main thread alive
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping the agent...")
            self.stop()

    def stop(self):
        """Stop the agent."""
        self.is_running = False
        if hasattr(self, "audio_thread") and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        self.audio_capture.stop()
        logger.info("Agent stopped")

    def _process_audio_stream(self):
        """Process the audio stream and handle the interview."""
        logger.info("Processing audio stream...")

        # Configure speech recognition to be more verbose
        import logging

        logging.getLogger("modules.speech_recognition").setLevel(logging.DEBUG)

        self.audio_capture.start()

        try:
            while self.is_running:
                # Get audio chunk as torch tensor
                audio_tensor = self.audio_capture.get_audio_tensor()
                if audio_tensor is None:
                    time.sleep(0.1)
                    continue

                # Determine the speaker (interviewer or candidate)
                speaker_info = self.speaker_diarization.identify_speaker(audio_tensor)
                self.current_speaker = speaker_info["speaker"]

                if self.current_speaker != audio_tensor["speaker"]:
                    logger.debug(
                        f"Speaker mismatch: capture says {audio_tensor['speaker']}, diarization says {self.current_speaker}"
                    )

                # Only process interviewer's speech
                if self.current_speaker == "interviewer":
                    # Get the raw audio data and convert it to bytes
                    audio_data = audio_tensor["audio"]
                    logger.debug(
                        f"Processing audio chunk from interviewer, shape: {audio_data.shape}"
                    )

                    try:
                        # Generate a unique ID for this chunk for tracking
                        chunk_id = str(uuid.uuid4())[:8]
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Starting processing for interviewer"
                        )

                        # Create temporary WAV file with a unique name
                        temp_filename = os.path.join(
                            tempfile.gettempdir(), f"interviewer_audio_{chunk_id}.wav"
                        )

                        # Ensure audio_data is an actual numpy array
                        if not isinstance(audio_data, np.ndarray):
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Expected numpy array, got {type(audio_data)}"
                            )
                            audio_data = np.array(audio_data)

                        # Check for NaN or inf values
                        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Audio contains NaN or inf values, cleaning up"
                            )
                            audio_data = np.nan_to_num(audio_data)

                        # Ensure audio_data is normalized and in the right format
                        max_val = np.max(np.abs(audio_data))
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Audio max value: {max_val}"
                        )

                        if max_val > 0:  # Avoid division by zero
                            # Normalize to -1 to 1 range first
                            audio_data = audio_data / max_val

                        # Convert to int16
                        audio_data = (audio_data * 32767).astype(np.int16)
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Converted to int16, shape: {audio_data.shape}"
                        )

                        # Write audio data to WAV file
                        with wave.open(temp_filename, "wb") as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(audio_data.tobytes())

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - WAV file written: {temp_filename}"
                        )

                        # Verify the file was created
                        if not os.path.exists(temp_filename):
                            logger.error(
                                f"CHUNK ID: {chunk_id} - WAV file was not created"
                            )
                            raise FileNotFoundError(
                                f"Failed to create WAV file at {temp_filename}"
                            )

                        file_size = os.path.getsize(temp_filename)
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - WAV file size: {file_size} bytes"
                        )

                        # Read the WAV file back as bytes
                        with open(temp_filename, "rb") as f:
                            audio_bytes = f.read()

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Audio bytes read from file: {len(audio_bytes)} bytes"
                        )

                        # Explicitly log what we're sending to transcribe
                        logger.info(
                            f"CHUNK ID: {chunk_id} - Sending to transcribe: Type={type(audio_bytes)}, Size={len(audio_bytes)} bytes, ID={id(audio_bytes)}"
                        )

                        # Convert speech to text
                        text = self.speech_recognition.transcribe(audio_bytes)

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Transcription result: '{text}'"
                        )

                        if text:
                            logger.info(f"Interviewer: {text}")

                            # Add to interview context
                            self.interview_context.append(
                                {
                                    "speaker": "interviewer",
                                    "text": text,
                                    "timestamp": time.time(),
                                }
                            )

                        # Clean up the temp file
                        try:
                            if os.path.exists(temp_filename):
                                os.unlink(temp_filename)
                                logger.debug(
                                    f"CHUNK ID: {chunk_id} - Cleaned up temp file"
                                )
                        except Exception as e:
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Could not delete temp file: {e}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error converting interviewer audio to bytes: {e}",
                            exc_info=True,
                        )

                        # Process the query and generate response
                        response = self.ai_processor.process_query(
                            text, self.interview_context
                        )

                        # Check if diagram generation is requested
                        if "generate_diagram" in response:
                            diagram_type = response.get("diagram_type", "high_level")
                            diagram_path = self.diagram_generator.generate_diagram(
                                response["diagram_content"], diagram_type
                            )
                            logger.info(f"Generated diagram: {diagram_path}")

                        # Add AI response to context
                        self.interview_context.append(
                            {
                                "speaker": "agent",
                                "text": response.get("text", ""),
                                "timestamp": time.time(),
                            }
                        )

                        logger.info(f"Agent response: {response.get('text', '')}")

                # Also capture candidate's speech for context
                elif self.current_speaker == "candidate":
                    # Get the raw audio data and convert it to bytes
                    audio_data = audio_tensor["audio"]
                    logger.debug(
                        f"Processing audio chunk from candidate, shape: {audio_data.shape}"
                    )

                    try:
                        # Generate a unique ID for this chunk for tracking
                        chunk_id = str(uuid.uuid4())[:8]
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Starting processing for candidate"
                        )

                        # Create temporary WAV file with a unique name
                        temp_filename = os.path.join(
                            tempfile.gettempdir(), f"candidate_audio_{chunk_id}.wav"
                        )

                        # Ensure audio_data is an actual numpy array
                        if not isinstance(audio_data, np.ndarray):
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Expected numpy array, got {type(audio_data)}"
                            )
                            audio_data = np.array(audio_data)

                        # Check for NaN or inf values
                        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Audio contains NaN or inf values, cleaning up"
                            )
                            audio_data = np.nan_to_num(audio_data)

                        # Ensure audio_data is normalized and in the right format
                        max_val = np.max(np.abs(audio_data))
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Audio max value: {max_val}"
                        )

                        if max_val > 0:  # Avoid division by zero
                            # Normalize to -1 to 1 range first
                            audio_data = audio_data / max_val

                        # Convert to int16
                        audio_data = (audio_data * 32767).astype(np.int16)
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Converted to int16, shape: {audio_data.shape}"
                        )

                        # Write audio data to WAV file
                        with wave.open(temp_filename, "wb") as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(audio_data.tobytes())

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - WAV file written: {temp_filename}"
                        )

                        # Verify the file was created
                        if not os.path.exists(temp_filename):
                            logger.error(
                                f"CHUNK ID: {chunk_id} - WAV file was not created"
                            )
                            raise FileNotFoundError(
                                f"Failed to create WAV file at {temp_filename}"
                            )

                        file_size = os.path.getsize(temp_filename)
                        logger.debug(
                            f"CHUNK ID: {chunk_id} - WAV file size: {file_size} bytes"
                        )

                        # Read the WAV file back as bytes
                        with open(temp_filename, "rb") as f:
                            audio_bytes = f.read()

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Audio bytes read from file: {len(audio_bytes)} bytes"
                        )

                        # Explicitly log what we're sending to transcribe
                        logger.info(
                            f"CHUNK ID: {chunk_id} - Sending to transcribe: Type={type(audio_bytes)}, Size={len(audio_bytes)} bytes, ID={id(audio_bytes)}"
                        )

                        # Convert speech to text
                        text = self.speech_recognition.transcribe(audio_bytes)

                        logger.debug(
                            f"CHUNK ID: {chunk_id} - Transcription result: '{text}'"
                        )

                        if text:
                            logger.info(f"Candidate: {text}")
                            self.interview_context.append(
                                {
                                    "speaker": "candidate",
                                    "text": text,
                                    "timestamp": time.time(),
                                }
                            )

                        # Clean up the temp file
                        try:
                            if os.path.exists(temp_filename):
                                os.unlink(temp_filename)
                                logger.debug(
                                    f"CHUNK ID: {chunk_id} - Cleaned up temp file"
                                )
                        except Exception as e:
                            logger.warning(
                                f"CHUNK ID: {chunk_id} - Could not delete temp file: {e}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error converting candidate audio to bytes: {e}",
                            exc_info=True,
                        )

        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            self.stop()


if __name__ == "__main__":
    agent = SystemDesignAgent()
    agent.start()
