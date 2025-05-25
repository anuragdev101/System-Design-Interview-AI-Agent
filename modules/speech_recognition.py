"""
Module for speech recognition to convert audio to text.
"""

import logging
import os
import tempfile
from io import BytesIO

import speech_recognition as sr
import whisper
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class SpeechRecognition:
    """
    Converts speech to text using OpenAI's Whisper model.
    Falls back to other speech recognition engines if needed.
    """

    def __init__(self, model_size="base"):
        """
        Initialize the speech recognition module.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
        """
        logger.info(f"Initializing speech recognition with Whisper model: {model_size}")

        try:
            # Initialize Whisper model
            self.whisper_model = whisper.load_model(model_size)

            # Initialize SpeechRecognition for fallback
            self.recognizer = sr.Recognizer()

            logger.info("Speech recognition initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            raise

    def transcribe(self, audio_bytes):
        """
        Transcribe speech to text.

        Args:
            audio_bytes: Audio data as bytes

        Returns:
            Transcribed text or empty string if transcription fails
        """
        try:
            # EXPLICIT DEBUGGING: Log incoming data type and details
            logger.info(
                f"TRANSCRIBE RECEIVED: Type={type(audio_bytes)}, ID={id(audio_bytes)}"
            )

            # Check if we're getting a NumPy array and convert it
            import numpy as np

            if isinstance(audio_bytes, np.ndarray):
                logger.info(
                    f"Converting NumPy array to bytes, shape={audio_bytes.shape}"
                )
                # Convert NumPy array to WAV bytes
                try:
                    import os
                    import tempfile
                    import wave

                    # Create temporary WAV file
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as temp_file:
                        temp_filename = temp_file.name

                    # Scale to 16-bit int range
                    max_val = np.max(np.abs(audio_bytes))
                    if max_val > 0:  # Avoid division by zero
                        # Normalize to -1 to 1 range first
                        audio_bytes = audio_bytes / max_val

                    # Convert to int16
                    audio_bytes = (audio_bytes * 32767).astype(np.int16)

                    # Write to WAV file
                    with wave.open(temp_filename, "wb") as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)  # 16kHz
                        wf.writeframes(audio_bytes.tobytes())

                    # Read back as bytes
                    with open(temp_filename, "rb") as f:
                        audio_bytes = f.read()

                    # Clean up
                    os.unlink(temp_filename)

                    logger.info(
                        f"Successfully converted NumPy array to {len(audio_bytes)} bytes"
                    )
                except Exception as e:
                    logger.error(f"Failed to convert NumPy array to bytes: {e}")
                    return ""

            # First check if we have enough audio data to process
            if audio_bytes is None or (
                isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) < 100
            ):  # Arbitrary minimum size
                audio_length = len(audio_bytes) if audio_bytes is not None else 0
                logger.error(
                    f"Received audio data is too small: {audio_length} bytes. Minimum required is 100 bytes."
                )
                # Save the received bytes for analysis if we have any
                if audio_bytes is not None and len(audio_bytes) > 0:
                    try:
                        debug_filename = os.path.join(
                            tempfile.gettempdir(),
                            f"invalid_audio_{len(audio_bytes)}bytes.bin",
                        )
                        with open(debug_filename, "wb") as f:
                            f.write(audio_bytes)
                        logger.info(
                            f"Saved problematic audio data to {debug_filename} for analysis"
                        )
                    except Exception as save_error:
                        logger.warning(
                            f"Could not save diagnostic audio data: {save_error}"
                        )
                return ""

            # Log audio data characteristics for debugging
            logger.info(f"Processing audio data: {len(audio_bytes)} bytes")
            # Log first few bytes as hex for debugging
            if len(audio_bytes) >= 16:
                hex_preview = " ".join([f"{b:02x}" for b in audio_bytes[:16]])
                logger.info(f"Audio data preview (first 16 bytes): {hex_preview}")
                # Check for common audio file signatures
                if isinstance(audio_bytes, (bytes, bytearray)):
                    if audio_bytes.startswith(b"RIFF"):
                        logger.info(
                            "Audio appears to be in WAV format (RIFF header detected)"
                        )
                    elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(
                        b"\xff\xfb"
                    ):
                        logger.info("Audio appears to be in MP3 format")
                    elif audio_bytes.startswith(b"fLaC"):
                        logger.info("Audio appears to be in FLAC format")
                    elif audio_bytes.startswith(b"OggS"):
                        logger.info("Audio appears to be in OGG format")
                else:
                    logger.warning(
                        f"Audio is not in a recognized format: type={type(audio_bytes)}"
                    )

            # First try with Whisper
            text = self._transcribe_with_whisper(audio_bytes)

            # If Whisper fails, fall back to other engines
            if not text:
                logger.info(
                    "Whisper transcription failed, falling back to other engines"
                )
                text = self._transcribe_with_fallback(audio_bytes)

            return text

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""

    def _transcribe_with_whisper(self, audio_bytes):
        """
        Transcribe audio using Whisper model.

        Args:
            audio_bytes: Audio data as bytes

        Returns:
            Transcribed text or empty string if transcription fails
        """
        try:
            # Check if audio_bytes is valid
            if not isinstance(audio_bytes, (bytes, bytearray)):
                logger.error(
                    f"Invalid audio data type: {type(audio_bytes)}, expected bytes or bytearray"
                )
                return ""

            # Use pydub to handle audio conversion more robustly
            from io import BytesIO

            import numpy as np

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            try:
                # Try to load as an audio segment first
                audio_buffer = BytesIO(audio_bytes)
                audio_buffer.seek(0)
                audio_segment = AudioSegment.from_file(audio_buffer, format="wav")
                logger.info(
                    f"Successfully loaded audio: {len(audio_segment)}ms, channels={audio_segment.channels}, sample_width={audio_segment.sample_width}, frame_rate={audio_segment.frame_rate}Hz"
                )

                # Ensure consistent format: 16kHz, 16-bit, mono
                audio_segment = (
                    audio_segment.set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)
                )

                # Export to the temporary WAV file
                audio_segment.export(temp_filename, format="wav")

            except Exception as pydub_error:
                logger.warning(
                    f"Could not process with pydub: {pydub_error}, trying numpy approach"
                )

                try:
                    # Convert audio bytes to numpy array (assuming 16-bit PCM)
                    # Ensure audio_bytes has even length for 16-bit samples
                    if len(audio_bytes) % 2 != 0:
                        logger.warning(
                            f"Audio bytes length ({len(audio_bytes)}) is not even, truncating last byte"
                        )
                        audio_bytes = audio_bytes[:-1]

                    # Only proceed if we have valid data
                    if len(audio_bytes) >= 2:  # At least one 16-bit sample
                        # Convert to numpy array of 16-bit integers
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                        # Create a WAV file with proper format
                        import wave

                        with wave.open(temp_filename, "wb") as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(audio_np.tobytes())
                    else:
                        logger.error(f"Audio data too short: {len(audio_bytes)} bytes")
                        return ""

                except Exception as np_error:
                    logger.error(f"Failed to process audio with numpy: {np_error}")
                    return ""

            # Verify the file exists and has content
            if (
                not os.path.exists(temp_filename) or os.path.getsize(temp_filename) < 44
            ):  # 44 bytes is minimum WAV header
                logger.error(
                    f"Invalid WAV file created: {os.path.getsize(temp_filename) if os.path.exists(temp_filename) else 'file not found'}"
                )
                return ""

            # Transcribe using Whisper
            try:
                result = self.whisper_model.transcribe(temp_filename)
                transcribed_text = result["text"].strip()
                logger.info(f"Whisper transcription successful: '{transcribed_text}'")
                return transcribed_text
            except Exception as whisper_error:
                logger.error(f"Whisper transcription failed: {whisper_error}")
                return ""
            finally:
                # Clean up the temporary file
                try:
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary file: {cleanup_error}"
                    )

        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            return ""

    def _transcribe_with_fallback(self, audio_bytes):
        """
        Transcribe audio using fallback engines.

        Args:
            audio_bytes: Audio data as bytes

        Returns:
            Transcribed text or empty string if transcription fails
        """
        try:
            # Check if audio_bytes is valid
            if not isinstance(audio_bytes, (bytes, bytearray)):
                logger.error(
                    f"Invalid audio data type for fallback: {type(audio_bytes)}, expected bytes or bytearray"
                )
                return ""

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Use pydub for more robust audio handling
            try:
                # Try to load as an audio segment
                audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))

                # Ensure consistent format: 16kHz, 16-bit, mono
                audio_segment = (
                    audio_segment.set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)
                )

                # Export to the temporary WAV file
                audio_segment.export(temp_filename, format="wav")
                logger.info(
                    f"Created WAV file for fallback recognition: {len(audio_segment)}ms"
                )

            except Exception as pydub_error:
                logger.warning(
                    f"Could not process with pydub for fallback: {pydub_error}"
                )

                # Fallback to direct WAV creation if pydub fails
                try:
                    # Ensure audio_bytes has even length for 16-bit samples
                    if len(audio_bytes) % 2 != 0:
                        audio_bytes = audio_bytes[:-1]

                    # Only proceed if we have valid data
                    if len(audio_bytes) >= 2:  # At least one 16-bit sample
                        import wave

                        import numpy as np

                        # Convert to numpy array of 16-bit integers
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                        # Create a WAV file with proper format
                        with wave.open(temp_filename, "wb") as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(audio_np.tobytes())
                    else:
                        logger.error(
                            f"Audio data too short for fallback: {len(audio_bytes)} bytes"
                        )
                        return ""

                except Exception as np_error:
                    logger.error(f"Failed to process audio for fallback: {np_error}")
                    return ""

            # Use SpeechRecognition to transcribe
            try:
                with sr.AudioFile(temp_filename) as source:
                    audio_data = self.recognizer.record(source)

                    # Try different recognition engines
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        logger.info(
                            "Successfully transcribed with Google Speech Recognition"
                        )
                    except sr.UnknownValueError:
                        logger.info(
                            "Google Speech Recognition could not understand audio, trying Sphinx"
                        )
                        try:
                            text = self.recognizer.recognize_sphinx(audio_data)
                            logger.info("Successfully transcribed with Sphinx")
                        except Exception as sphinx_error:
                            logger.warning(f"Sphinx recognition failed: {sphinx_error}")
                            text = ""
                    except Exception as google_error:
                        logger.warning(f"Google recognition failed: {google_error}")
                        text = ""
            except Exception as sr_error:
                logger.error(f"SpeechRecognition error: {sr_error}")
                text = ""

            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

            return text

        except Exception as e:
            logger.error(f"Error in fallback transcription: {e}")
            return ""
