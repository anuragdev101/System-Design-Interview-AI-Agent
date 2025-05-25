# modules/audio_capture.py

import concurrent.futures
import logging
import platform
import queue
import threading
import time
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

# Try to import PyAudio for basic audio support
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.info("pyaudio not available. Some fallback options might be limited.")

# Try to import pyaudiowpatch for WASAPI loopback support (Windows)
try:
    import pyaudiowpatch

    PYAUDIOWPATCH_AVAILABLE = True
except ImportError:
    PYAUDIOWPATCH_AVAILABLE = False
    if platform.system().lower() == "windows":
        logging.warning(
            "pyaudiowpatch not available. System audio capture on Windows will be limited to sounddevice Stereo Mix or similar."
        )

# Try to import librosa for resampling
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning(
        "librosa not available. Resampling capability will be limited if needed for system audio capture."
    )

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures system audio from selected devices and provides processed audio chunks."""

    def __init__(
        self, sample_rate=16000, chunk_duration=0.2, channels=1
    ):  # Default VAD-friendly chunk_duration
        self.target_sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # This is for the final VAD/STT chunk size
        self.target_channels = channels

        self.is_capturing = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.visualization_queue = queue.Queue(maxsize=50)
        self.audio_tensor_queue = queue.Queue(maxsize=50)

        self.candidate_stream: Optional[sd.InputStream] = None
        self.interviewer_stream: Optional[Any] = None
        self.candidate_device_index: Optional[int] = None
        self.interviewer_device_index: Optional[int] = None

        self.audio_processor_pool: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None
        )
        self.processing_futures: set[concurrent.futures.Future] = set()

        self.capture_thread: Optional[threading.Thread] = None
        self.processing_futures_reaper_thread: Optional[threading.Thread] = None

        self.os_system = platform.system().lower()
        self.can_capture_system_audio_sd = self._check_system_audio_capabilities_sd()
        self.default_input_device, self.default_output_device = (
            self._get_default_sd_devices()
        )

        logger.info(
            f"AudioCapture Initialized. Target App Rate: {self.target_sample_rate}Hz, "
            f"Target Chunk Duration for VAD/STT: {self.chunk_duration}s, Target App Channels: {self.target_channels}. "
            f"ThreadPoolExecutor will be created on start."
        )

    def _get_default_sd_devices(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            all_devices = sd.query_devices()
            default_input_idx = (
                sd.default.device[0] if sd.default.device[0] != -1 else None
            )
            default_output_idx = (
                sd.default.device[1] if sd.default.device[1] != -1 else None
            )

            if (
                default_input_idx is None and all_devices
            ):  # Check if all_devices is not empty
                for i, device in enumerate(all_devices):
                    if device["max_input_channels"] > 0:
                        default_input_idx = i
                        logger.info(
                            f"No default input device by index, using first available: {device['name']} (Idx {i})"
                        )
                        break
            return default_input_idx, default_output_idx
        except Exception as e:
            logger.error(
                f"Error getting sounddevice default devices: {e}", exc_info=True
            )
            return None, None

    def _check_system_audio_capabilities_sd(self) -> bool:
        try:
            devices = sd.query_devices()
            if not devices:
                return False  # No devices at all
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0 and any(
                    keyword in device["name"].lower()
                    for keyword in [
                        "stereo mix",
                        "loopback",
                        "system audio",
                        "what u hear",
                    ]
                ):
                    try:
                        # Test with a short, non-blocking stream
                        test_channels = 1 if device["max_input_channels"] > 0 else 0
                        if test_channels == 0:
                            continue

                        with sd.InputStream(
                            device=i,
                            channels=test_channels,
                            samplerate=self.target_sample_rate,
                            callback=lambda i, f, t, s: None,
                            blocksize=1024,
                        ):
                            time.sleep(0.05)  # Let it initialize
                        logger.info(
                            f"Successfully tested sounddevice system audio-like device: {device['name']}"
                        )
                        return True
                    except Exception as e_stream:
                        logger.debug(
                            f"Sounddevice could not open system audio-like device {device['name']}: {e_stream}"
                        )
            logger.debug(
                "No sounddevice-compatible system audio capture device passed basic test."
            )
            return False
        except Exception as e:
            logger.error(
                f"Error checking system audio capabilities with sounddevice: {e}",
                exc_info=True,
            )
            return False

    def _sd_callback_wrapper(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
        speaker_role: str,
        stream_sample_rate: int,
    ):
        if status:
            logger.warning(
                f"Sounddevice callback status for {speaker_role} ({status!s})"
            )
        if self.audio_processor_pool and not self.audio_processor_pool._shutdown:
            try:
                future = self.audio_processor_pool.submit(
                    self._process_audio_chunk,
                    indata.copy(),
                    speaker_role,
                    stream_sample_rate,
                )
                self.processing_futures.add(future)
            except (
                RuntimeError
            ) as e_runtime:  # Catch if submit is called after shutdown somehow
                logger.error(
                    f"RuntimeError submitting task for {speaker_role} to audio_processor_pool (already shutdown?): {e_runtime}"
                )
        else:
            logger.error(
                f"Audio processor pool not available or shutdown. Cannot process audio for {speaker_role}."
            )

    def candidate_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ):
        self._sd_callback_wrapper(
            indata,
            frames,
            time_info,
            status,
            speaker_role="candidate",
            stream_sample_rate=self.target_sample_rate,
        )

    def interviewer_sd_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ):
        self._sd_callback_wrapper(
            indata,
            frames,
            time_info,
            status,
            speaker_role="interviewer",
            stream_sample_rate=self.target_sample_rate,
        )

    def wasapi_interviewer_callback(
        self,
        indata_float32: np.ndarray,
        frames: int,
        capture_time_info: Any,
        status_info: Any,
    ):
        # logger.debug(f"AudioCapture: wasapi_interviewer_callback received {frames} frames.") # Can be verbose
        if status_info:
            logger.warning(f"WASAPI interviewer callback status: {status_info}")

        stream_obj = self.interviewer_stream
        if isinstance(stream_obj, self.WasapiLoopbackStream):
            native_rate = stream_obj.stream_open_rate
            if self.audio_processor_pool and not self.audio_processor_pool._shutdown:
                try:
                    future = self.audio_processor_pool.submit(
                        self._process_audio_chunk,
                        indata_float32.copy(),
                        "interviewer",
                        native_rate,
                    )
                    self.processing_futures.add(future)
                except RuntimeError as e_runtime:
                    logger.error(
                        f"RuntimeError submitting WASAPI task to audio_processor_pool (already shutdown?): {e_runtime}"
                    )
            else:
                logger.error(
                    "Audio processor pool not available or shutdown. Cannot process WASAPI audio for interviewer."
                )
        else:
            logger.error(
                "wasapi_interviewer_callback invoked with unexpected interviewer_stream type!"
            )

    def _process_audio_chunk(
        self, indata: np.ndarray, speaker: str, original_sample_rate: int
    ):
        # ... (Implementation from your last complete file, ensure it's robust) ...
        # This method processes the audio (resample, normalize, silence check) and puts it on audio_tensor_queue
        # logger.debug(f"AudioCapture: _process_audio_chunk started for speaker: {speaker}, input shape: {indata.shape}, rate: {original_sample_rate}")
        try:
            if not self.is_capturing:
                return

            if indata is None or indata.size == 0:
                logger.debug(f"Empty audio chunk for {speaker}, skipping.")
                return

            processed_data = indata
            if processed_data.dtype != np.float32:
                if np.issubdtype(processed_data.dtype, np.integer):
                    max_val = np.iinfo(processed_data.dtype).max
                    if max_val == 0:  # Avoid division by zero for unusual int types
                        logger.warning(
                            f"Integer type {processed_data.dtype} has max_val 0. Cannot normalize to float32."
                        )
                        return
                    processed_data = processed_data.astype(np.float32) / max_val
                elif (
                    np.issubdtype(processed_data.dtype, np.floating)
                    and processed_data.dtype != np.float32
                ):
                    processed_data = processed_data.astype(np.float32)
                else:
                    logger.error(
                        f"Cannot convert {speaker} audio (dtype {processed_data.dtype}) to float32. Skipping."
                    )
                    return

            if original_sample_rate != self.target_sample_rate:
                if LIBROSA_AVAILABLE:
                    # logger.debug(f"Resampling {speaker} from {original_sample_rate} to {self.target_sample_rate}")
                    y_to_resample = processed_data
                    if processed_data.ndim > 1 and processed_data.shape[1] > 1:
                        y_to_resample = np.mean(processed_data, axis=1)
                    elif processed_data.ndim > 1 and processed_data.shape[1] == 1:
                        y_to_resample = processed_data[:, 0]

                    resampled_y = librosa.resample(
                        y=y_to_resample,
                        orig_sr=original_sample_rate,
                        target_sr=self.target_sample_rate,
                    )
                    processed_data = (
                        resampled_y.reshape(-1, 1)
                        if self.target_channels == 1
                        else np.tile(
                            resampled_y.reshape(-1, 1), (1, self.target_channels)
                        )
                    )
                else:
                    logger.warning(
                        f"Librosa not available. Cannot resample {speaker} audio from {original_sample_rate}Hz to {self.target_sample_rate}Hz."
                    )

            current_ch = processed_data.shape[1] if processed_data.ndim > 1 else 1
            if current_ch > self.target_channels:
                if self.target_channels == 1:
                    processed_data = np.mean(processed_data, axis=1, keepdims=True)
                else:
                    processed_data = processed_data[:, : self.target_channels]
            elif current_ch < self.target_channels:
                if current_ch == 1:  # Input is mono
                    reshaped_mono = (
                        processed_data.reshape(-1, 1)
                        if processed_data.ndim == 1
                        else processed_data
                    )
                    processed_data = np.tile(reshaped_mono, (1, self.target_channels))
            elif (
                processed_data.ndim == 1 and self.target_channels == 1
            ):  # Input mono, target mono, ensure 2D
                processed_data = processed_data.reshape(-1, 1)

            # Ensure shape is (frames, channels)
            if processed_data.ndim == 1:
                processed_data = processed_data.reshape(-1, 1)

            silence_thresh_speaker = 0.005 if speaker == "candidate" else 0.001
            max_abs_val_in_chunk = np.max(np.abs(processed_data))
            if max_abs_val_in_chunk < silence_thresh_speaker:
                # logger.debug(f"Silent chunk for {speaker} (max_abs: {max_abs_val_in_chunk:.4f} < thresh: {silence_thresh_speaker:.4f}). Skipping.")
                return

            if not self.visualization_queue.full():
                viz_data = processed_data.copy()
                # Max val for viz_data already handled by its own processing or normalization not strictly needed if GUI handles range
                self.visualization_queue.put_nowait(viz_data)

            if not self.audio_tensor_queue.full():
                final_tensor_data = (
                    processed_data  # Should be (frames, target_channels) by now
                )
                self.audio_tensor_queue.put_nowait(
                    {
                        "audio": final_tensor_data,
                        "speaker": speaker,
                        "timestamp": time.time(),
                    }
                )
            else:
                logger.warning(
                    f"Audio tensor queue full for {speaker}. Dropping chunk."
                )
        except queue.Full:
            logger.warning(
                f"A queue was full during _process_audio_chunk for {speaker}"
            )
        except Exception as e:
            logger.error(
                f"Error in _process_audio_chunk for {speaker} (Rate: {original_sample_rate}Hz, Shape: {indata.shape if indata is not None else 'None'}): {e}",
                exc_info=True,
            )

    def get_audio_data(self) -> Optional[np.ndarray]:
        try:
            return self.visualization_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            return None

    def get_audio_tensor(self) -> Optional[Dict[str, Any]]:
        try:
            return self.audio_tensor_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting audio_tensor: {e}")
            return None

    def get_device_list(self) -> List[Dict[str, Any]]:
        devices_info_list = []
        try:
            sd_devices = sd.query_devices()
            logger.debug("Sounddevice Devices (get_device_list):")
            for i, device in enumerate(sd_devices):
                try:
                    hostapi_name = sd.query_hostapis(device["hostapi"])["name"]
                except:
                    hostapi_name = "Unknown HostAPI"
                dev_entry = {
                    "index": i,
                    "name": device["name"],
                    "hostapi_name": hostapi_name,
                    "max_input_channels": device["max_input_channels"],
                    "max_output_channels": device["max_output_channels"],
                    "default_samplerate": device["default_samplerate"],
                    "api_source": "sounddevice",
                }
                devices_info_list.append(dev_entry)
                logger.debug(
                    f"  sd_Idx {i}: {dev_entry['name']} (In {dev_entry['max_input_channels']}, Out {dev_entry['max_output_channels']}, API '{dev_entry['hostapi_name']}')"
                )
            return devices_info_list
        except Exception as e:
            logger.error(f"Error getting device list: {e}", exc_info=True)
            return []

    def start(
        self,
        candidate_device_index: Optional[int] = None,
        interviewer_device_index: Optional[int] = None,
        online_interview: bool = True,
    ):
        if self.is_capturing:
            logger.warning(
                "AudioCapture start() called but already capturing. Stopping existing capture first for a clean restart."
            )
            self.stop()
            time.sleep(0.2)

        logger.info("AudioCapture start() called.")
        self.is_capturing = True

        if self.audio_processor_pool is None or self.audio_processor_pool._shutdown:  # type: ignore
            logger.info("Creating new ThreadPoolExecutor for audio processing.")
            self.audio_processor_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="AudioProc"
            )
            self.processing_futures.clear()
        else:
            logger.debug("Using existing, active ThreadPoolExecutor.")

        try:
            sd_devices = sd.query_devices()
            if not sd_devices:
                raise Exception("No sound devices found by sounddevice library.")

            # Sounddevice streams will use target_sample_rate.
            # AudioCapture's chunk_duration determines how much data before callback.
            sd_chunk_size = int(self.target_sample_rate * self.chunk_duration)
            if (
                sd_chunk_size == 0
            ):  # If chunk_duration is too small, sd might not like blocksize=0
                sd_chunk_size = int(
                    self.target_sample_rate * 0.02
                )  # Default to 20ms if too small
                logger.warning(
                    f"Calculated sd_chunk_size was 0, defaulting to {sd_chunk_size} ({0.02*1000}ms)"
                )

            # --- Candidate Device ---
            # ... (Candidate device selection and stream start as in previous complete file) ...
            _candidate_device_index = candidate_device_index
            if _candidate_device_index is None:
                _candidate_device_index = self.default_input_device
            if (
                _candidate_device_index is None
                or not (0 <= _candidate_device_index < len(sd_devices))
                or sd_devices[_candidate_device_index]["max_input_channels"] == 0
            ):
                cand_idx_found = next(
                    (
                        i
                        for i, dev in enumerate(sd_devices)
                        if dev["max_input_channels"] > 0
                    ),
                    None,
                )
                if cand_idx_found is None:
                    raise ValueError("No suitable candidate input microphone found.")
                _candidate_device_index = cand_idx_found
            self.candidate_device_index = _candidate_device_index
            cand_info = sd_devices[self.candidate_device_index]
            cand_capture_channels = min(
                self.target_channels, cand_info["max_input_channels"]
            )
            logger.info(
                f"Setting up Candidate sounddevice: '{cand_info['name']}' (Idx {self.candidate_device_index}) @{self.target_sample_rate}Hz, {cand_capture_channels}ch."
            )
            self.candidate_stream = sd.InputStream(
                device=self.candidate_device_index,
                channels=cand_capture_channels,
                samplerate=self.target_sample_rate,
                callback=self.candidate_callback,
                blocksize=sd_chunk_size,
            )
            self.candidate_stream.start()
            logger.info("Candidate stream started.")

            # --- Interviewer Device ---
            # ... (Interviewer device selection and stream start logic as in previous complete file) ...
            _interviewer_device_index = (
                interviewer_device_index  # This is for sounddevice fallback
            )
            if online_interview:
                logger.info(
                    "Online interview: Attempting system audio for interviewer."
                )
                if self.os_system == "windows" and PYAUDIOWPATCH_AVAILABLE:
                    self.interviewer_stream = self._try_create_wasapi_loopback_stream()
                    if self.interviewer_stream:
                        self.interviewer_stream.start()
                        logger.info("WASAPI loopback stream started for interviewer.")

                if not self.interviewer_stream:
                    logger.info(
                        "WASAPI N/A or failed. Trying sounddevice loopback for interviewer."
                    )
                    loopback_sd_idx = _interviewer_device_index
                    if (
                        loopback_sd_idx is None
                        or not (0 <= loopback_sd_idx < len(sd_devices))
                        or not any(
                            k in sd_devices[loopback_sd_idx]["name"].lower()
                            for k in ["stereo mix", "loopback", "what u hear", "cable"]
                        )
                    ):
                        logger.debug(
                            f"Provided interviewer index {loopback_sd_idx} not a clear loopback. Auto-detecting sd loopback."
                        )
                        loopback_sd_idx = next(
                            (
                                i
                                for i, dev in enumerate(sd_devices)
                                if dev["max_input_channels"] > 0
                                and any(
                                    k in dev["name"].lower()
                                    for k in [
                                        "stereo mix",
                                        "loopback",
                                        "what u hear",
                                        "cable output",
                                    ]
                                )
                            ),
                            None,
                        )

                    if loopback_sd_idx is not None:
                        self.interviewer_device_index = (
                            loopback_sd_idx  # Store the actually used index
                        )
                        int_sd_info = sd_devices[self.interviewer_device_index]
                        int_sd_capture_ch = int_sd_info["max_input_channels"]
                        logger.info(
                            f"Setting Interviewer (sounddevice loopback): '{int_sd_info['name']}' (Idx {self.interviewer_device_index}) @{self.target_sample_rate}Hz, {int_sd_capture_ch}ch."
                        )
                        self.interviewer_stream = sd.InputStream(
                            device=self.interviewer_device_index,
                            channels=int_sd_capture_ch,
                            samplerate=self.target_sample_rate,
                            callback=self.interviewer_sd_callback,
                            blocksize=sd_chunk_size,
                        )
                        self.interviewer_stream.start()
                        logger.info(
                            "Interviewer stream (sounddevice loopback) started."
                        )
                    else:
                        logger.warning(
                            "No sounddevice loopback for interviewer found or configured for fallback."
                        )
            else:  # In-person
                logger.info("In-person interview: Interviewer is a microphone.")
                if (
                    _interviewer_device_index is None
                    or not (0 <= _interviewer_device_index < len(sd_devices))
                    or sd_devices[_interviewer_device_index]["max_input_channels"] == 0
                ):
                    int_mic_idx = next(
                        (
                            i
                            for i, dev in enumerate(sd_devices)
                            if dev["max_input_channels"] > 0
                            and i != self.candidate_device_index
                        ),
                        None,
                    )
                    if int_mic_idx is None:
                        raise ValueError(
                            "No suitable interviewer microphone for in-person setup (different from candidate)."
                        )
                    _interviewer_device_index = int_mic_idx
                self.interviewer_device_index = _interviewer_device_index
                int_mic_info = sd_devices[self.interviewer_device_index]
                int_mic_capture_ch = min(
                    self.target_channels, int_mic_info["max_input_channels"]
                )
                logger.info(
                    f"Setting Interviewer (mic): '{int_mic_info['name']}' (Idx {self.interviewer_device_index}) @{self.target_sample_rate}Hz, {int_mic_capture_ch}ch."
                )
                self.interviewer_stream = sd.InputStream(
                    device=self.interviewer_device_index,
                    channels=int_mic_capture_ch,
                    samplerate=self.target_sample_rate,
                    callback=self.interviewer_sd_callback,
                    blocksize=sd_chunk_size,
                )
                self.interviewer_stream.start()
                logger.info("Interviewer stream (microphone) started.")

            if not self.candidate_stream:
                raise Exception("Candidate stream failed to initialize properly.")

            # --- Start Monitoring Threads ---
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(
                    target=self._capture_audio_thread_loop,
                    name="AudioCaptureMonitor",
                    daemon=True,
                )
                self.capture_thread.start()
            if (
                self.processing_futures_reaper_thread is None
                or not self.processing_futures_reaper_thread.is_alive()
            ):
                self.processing_futures_reaper_thread = threading.Thread(
                    target=self._reap_processing_futures,
                    name="AudioFutureReaper",
                    daemon=True,
                )
                self.processing_futures_reaper_thread.start()
            logger.info("Audio capture process initiated and threads (re)started.")
        except Exception as e:
            logger.error(f"Fatal error during audio capture start: {e}", exc_info=True)
            self.stop()
            raise

    def _reap_processing_futures(self):
        # ... (Implementation from your last complete file) ...
        logger.debug("Audio processing futures reaper thread started.")
        while self.is_capturing or self.processing_futures:
            if not self.processing_futures:
                if not self.is_capturing:
                    break
                time.sleep(0.05)
                continue

            current_futures = list(self.processing_futures)
            for f_done in current_futures:
                if f_done.done():
                    try:
                        f_done.result(timeout=0.01)
                    except concurrent.futures.TimeoutError:
                        pass
                    except Exception as e_future:
                        logger.error(
                            f"Error in audio_processor_pool task: {e_future}",
                            exc_info=True,
                        )
                    self.processing_futures.discard(f_done)

            if not self.is_capturing and not self.processing_futures:
                break
            time.sleep(0.05)
        logger.debug("Audio processing futures reaper thread finished.")

    def _capture_audio_thread_loop(self):
        # ... (Implementation from your last complete file) ...
        logger.debug("Audio capture monitoring thread started.")
        while self.is_capturing:
            time.sleep(0.2)
        logger.debug(
            "Audio capture monitoring thread finished (is_capturing is False)."
        )

    def stop(self):
        # ... (Implementation from your last complete file, ensure pool is set to None) ...
        if (
            not self.is_capturing
            and self.candidate_stream is None
            and self.interviewer_stream is None
            and (
                self.audio_processor_pool is None or self.audio_processor_pool._shutdown
            )
        ):  # type: ignore
            logger.debug(
                "AudioCapture.stop() called but already fully stopped or not started."
            )
            return

        logger.info("AudioCapture.stop() called. Initiating shutdown sequence...")
        self.is_capturing = False

        if (
            hasattr(self, "capture_thread")
            and self.capture_thread
            and self.capture_thread.is_alive()
        ):
            logger.debug("Joining capture_thread...")
            self.capture_thread.join(timeout=0.5)
            if self.capture_thread.is_alive():
                logger.warning("Capture_thread did not join in time.")
            self.capture_thread = None

        if self.candidate_stream:
            logger.debug("Stopping candidate_stream...")
            try:
                self.candidate_stream.stop()
                self.candidate_stream.close()
                logger.debug("Candidate_stream stopped and closed.")
            except Exception as e:
                logger.error(
                    f"Error stopping/closing candidate stream: {e}", exc_info=True
                )
            finally:
                self.candidate_stream = None

        if self.interviewer_stream:
            logger.debug("Stopping interviewer_stream...")
            try:
                if hasattr(self.interviewer_stream, "stop"):
                    self.interviewer_stream.stop()
                if hasattr(self.interviewer_stream, "close"):
                    self.interviewer_stream.close()
                logger.debug("Interviewer_stream stopped and closed.")
            except Exception as e:
                logger.error(
                    f"Error stopping/closing interviewer_stream: {e}", exc_info=True
                )
            finally:
                self.interviewer_stream = None

        if hasattr(self, "audio_processor_pool") and self.audio_processor_pool and not self.audio_processor_pool._shutdown:  # type: ignore
            logger.debug("Shutting down audio_processor_pool (wait=True)...")
            try:
                self.audio_processor_pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self.audio_processor_pool.shutdown(wait=True)
            logger.debug("Audio_processor_pool shut down.")
        self.audio_processor_pool = None

        # Reaper should be joined after pool shutdown to process any final futures from callbacks during stream stop
        if (
            hasattr(self, "processing_futures_reaper_thread")
            and self.processing_futures_reaper_thread
            and self.processing_futures_reaper_thread.is_alive()
        ):
            logger.debug(
                "Joining processing_futures_reaper_thread (after pool shutdown)..."
            )
            self.processing_futures_reaper_thread.join(timeout=1.0)
            if self.processing_futures_reaper_thread.is_alive():
                logger.warning("Processing_futures_reaper_thread did not join in time.")
            self.processing_futures_reaper_thread = None

        self._clear_queues()
        logger.info("AudioCapture stopped and all resources cleaned up.")

    def _clear_queues(self):
        # ... (Implementation from your last complete file) ...
        logger.debug("Clearing audio queues...")
        queues_to_clear = [
            getattr(self, "audio_queue", None),
            getattr(self, "visualization_queue", None),
            getattr(self, "audio_tensor_queue", None),
        ]
        for q_obj in queues_to_clear:
            if q_obj:
                count = 0
                while not q_obj.empty():
                    try:
                        q_obj.get_nowait()
                        count += 1
                    except queue.Empty:
                        break
                    except Exception as e_q:
                        logger.warning(f"Error clearing item from queue: {e_q}")
                        break
                if count > 0:
                    logger.debug(f"Cleared {count} items from a queue.")
        logger.debug("Audio queues cleared.")

    def _try_create_wasapi_loopback_stream(
        self,
    ) -> Optional["AudioCapture.WasapiLoopbackStream"]:
        # ... (Implementation from your last complete file, ensure target_processing_chunk_frames is passed) ...
        if self.os_system != "windows" or not PYAUDIOWPATCH_AVAILABLE:
            logger.debug(
                "_try_create_wasapi_loopback_stream: Not Windows or PyAudioWPatch unavailable."
            )
            return None
        logger.info("Attempting to create WASAPI loopback stream via PyAudioWPatch...")
        p_wpatch_instance = None
        try:
            p_wpatch_instance = pyaudiowpatch.PyAudio()
            wasapi_api_info = p_wpatch_instance.get_host_api_info_by_type(pyaudiowpatch.paWASAPI)  # type: ignore
            if not wasapi_api_info:
                logger.error("PyAudioWPatch: Could not get WASAPI Host API info.")
                if p_wpatch_instance:
                    p_wpatch_instance.terminate()
                return None

            default_output_pyaudio_idx = wasapi_api_info.get("defaultOutputDevice", -1)
            if default_output_pyaudio_idx == -1:
                logger.warning(
                    "PyAudioWPatch: No default WASAPI output device found by index."
                )
                if p_wpatch_instance:
                    p_wpatch_instance.terminate()
                return None

            default_output_info = p_wpatch_instance.get_device_info_by_index(
                default_output_pyaudio_idx
            )
            logger.info(
                f"PyAudioWPatch: Default output is '{default_output_info['name']}' (paIdx {default_output_pyaudio_idx})"
            )

            loopback_device_info = None
            for dev_info_gen in p_wpatch_instance.get_loopback_device_info_generator():
                if default_output_info["name"].lower() in dev_info_gen[
                    "name"
                ].lower() and dev_info_gen.get("isLoopbackDevice"):
                    loopback_device_info = dev_info_gen
                    break

            if not loopback_device_info:
                try:
                    loopback_device_info = next(
                        p_wpatch_instance.get_loopback_device_info_generator()
                    )
                except StopIteration:
                    logger.error(
                        "PyAudioWPatch: No WASAPI loopback devices found at all."
                    )
                    if p_wpatch_instance:
                        p_wpatch_instance.terminate()
                    return None
                logger.warning(
                    f"Using first available WASAPI loopback as fallback: '{loopback_device_info['name']}'"
                )

            if (
                not loopback_device_info
            ):  # Should be caught by StopIteration above, but as a safeguard
                logger.error(
                    "PyAudioWPatch: Critical - loopback_device_info is None after checks."
                )
                if p_wpatch_instance:
                    p_wpatch_instance.terminate()
                return None

            logger.info(
                f"Selected loopback device for WASAPI: '{loopback_device_info['name']}' (paIdx {loopback_device_info['index']})"
            )

            wasapi_capture_channels = int(loopback_device_info["maxInputChannels"])
            if not (0 < wasapi_capture_channels <= 8):
                wasapi_capture_channels = 2

            wasapi_device_native_rate = int(loopback_device_info["defaultSampleRate"])
            if wasapi_device_native_rate <= 0:
                wasapi_device_native_rate = 48000
            logger.info(
                f"WASAPI loopback: Opening at device native rate {wasapi_device_native_rate}Hz, {wasapi_capture_channels}ch."
            )

            # This is the target size of the chunks AudioCapture should provide to main_gui
            target_processing_chunk_frames = int(
                self.chunk_duration * wasapi_device_native_rate
            )

            custom_stream = self.WasapiLoopbackStream(
                pyaudio_instance=p_wpatch_instance,
                loopback_device_info=loopback_device_info,
                callback_to_audiocapture=self.wasapi_interviewer_callback,
                device_native_rate=wasapi_device_native_rate,
                capture_channels=wasapi_capture_channels,
                target_processing_chunk_frames=target_processing_chunk_frames,
            )
            return custom_stream

        except OSError as e_os:
            logger.warning(
                f"PyAudioWPatch OS Error (WASAPI Host API not available or error): {e_os}"
            )
            if p_wpatch_instance:
                p_wpatch_instance.terminate()
            return None
        except Exception as e:
            logger.error(
                f"General failure in _try_create_wasapi_loopback_stream: {e}",
                exc_info=True,
            )
            if p_wpatch_instance:
                p_wpatch_instance.terminate()
            return None

    class WasapiLoopbackStream:
        def __init__(
            self,
            pyaudio_instance: pyaudio.PyAudio,
            loopback_device_info: dict,  # type: ignore
            callback_to_audiocapture,
            device_native_rate: int,
            capture_channels: int,
            target_processing_chunk_frames: int,
        ):

            self.pyaudio_instance = pyaudio_instance
            self.loopback_device_info = loopback_device_info
            self.audiocapture_callback_fn = callback_to_audiocapture

            self.stream_open_rate = device_native_rate
            self.stream_capture_channels = capture_channels
            self.internal_read_chunk_frames = 1024  # Smaller chunk for PyAudio read
            self.target_callback_chunk_frames = target_processing_chunk_frames

            self.active = False
            self.stream_pa: Optional[pyaudio.Stream] = None  # type: ignore
            self.stop_event = threading.Event()
            self.read_thread: Optional[threading.Thread] = None
            self.stream_format = pyaudio.paInt16  # type: ignore

            logger.info(
                f"WasapiLoopbackStream Init: Dev='{self.loopback_device_info['name']}', "
                f"OpenRate={self.stream_open_rate}, CaptureCh={self.stream_capture_channels}, "
                f"InternalReadFrames={self.internal_read_chunk_frames}, TargetCallbackFrames={self.target_callback_chunk_frames}"
            )

        def start(self):  # -> 'AudioCapture.WasapiLoopbackStream':
            if self.active:
                logger.warning(
                    f"WasapiLoopbackStream.start() called but already active for paIdx {self.loopback_device_info['index']}."
                )
                return self
            try:
                dev_idx = self.loopback_device_info["index"]
                logger.info(
                    f"WasapiLoopbackStream: Starting capture on paIdx {dev_idx}..."
                )
                self.stream_pa = self.pyaudio_instance.open(
                    format=self.stream_format,
                    channels=self.stream_capture_channels,
                    rate=self.stream_open_rate,
                    input=True,
                    frames_per_buffer=self.internal_read_chunk_frames,
                    input_device_index=dev_idx,
                    stream_callback=None,
                )
                self.active = True
                self.stop_event.clear()
                self.read_thread = threading.Thread(
                    target=self._threaded_read_loop,
                    name=f"WASAPIReadLoop-Idx{dev_idx}",
                    daemon=True,
                )
                self.read_thread.start()
                logger.info(
                    f"WasapiLoopbackStream started for paIdx {dev_idx}, read thread launched."
                )
                return self
            except Exception as e:
                logger.error(
                    f"WasapiLoopbackStream: Error starting PyAudio stream for paIdx {self.loopback_device_info['index']}: {e}",
                    exc_info=True,
                )
                self.active = False
                if self.stream_pa:
                    try:
                        self.stream_pa.close()
                    except:
                        pass
                self.stream_pa = None
                raise

        def _threaded_read_loop(self):
            # ... (Implementation from the previous answer with detailed logging and accumulation) ...
            dev_idx = self.loopback_device_info["index"]
            logger.info(
                f"WASAPI _threaded_read_loop starting for paIdx {dev_idx} | Thread ID: {threading.get_ident()}."
            )

            accumulated_audio_data: List[np.ndarray] = []
            frames_accumulated = 0

            consecutive_errors = 0
            consecutive_no_data_available = 0
            max_consecutive_no_data_available = (
                200  # e.g., 200 * 10ms sleep = 2 seconds
            )

            while self.active and not self.stop_event.is_set():
                try:
                    if not self.stream_pa:
                        logger.warning(
                            f"WASAPI Loop (Idx {dev_idx}): self.stream_pa is None. Exiting loop."
                        )
                        self.active = False
                        break
                    if not self.stream_pa.is_active():
                        if not self.active or self.stop_event.is_set():
                            break
                        logger.warning(
                            f"WASAPI Loop (Idx {dev_idx}): Stream inactive. Pausing."
                        )
                        time.sleep(0.1)
                        continue

                    available_frames = self.stream_pa.get_read_available()

                    if available_frames < self.internal_read_chunk_frames:
                        if not self.active or self.stop_event.is_set():
                            break
                        time.sleep(0.010)
                        consecutive_no_data_available += 1
                        if (
                            consecutive_no_data_available
                            > max_consecutive_no_data_available
                        ):
                            logger.warning(
                                f"WASAPI Loop (Idx {dev_idx}): No new data available for internal read for {max_consecutive_no_data_available * 10}ms. Loop continues."
                            )
                            consecutive_no_data_available = 0
                        continue

                    consecutive_no_data_available = 0

                    raw_bytes = self.stream_pa.read(
                        self.internal_read_chunk_frames, exception_on_overflow=False
                    )

                    if not self.active or self.stop_event.is_set():
                        break
                    if not raw_bytes:
                        time.sleep(0.005)
                        continue

                    consecutive_errors = 0

                    audio_np_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    audio_np_float32_chunk = audio_np_int16.astype(np.float32) / 32767.0

                    num_samples_in_chunk = (
                        len(audio_np_float32_chunk) // self.stream_capture_channels
                    )
                    if num_samples_in_chunk * self.stream_capture_channels != len(
                        audio_np_float32_chunk
                    ):
                        audio_np_float32_chunk = audio_np_float32_chunk[
                            : num_samples_in_chunk * self.stream_capture_channels
                        ]

                    if num_samples_in_chunk == 0:
                        continue

                    audio_reshaped_chunk = audio_np_float32_chunk.reshape(
                        num_samples_in_chunk, self.stream_capture_channels
                    )

                    accumulated_audio_data.append(audio_reshaped_chunk)
                    frames_accumulated += num_samples_in_chunk

                    if frames_accumulated >= self.target_callback_chunk_frames:
                        final_segment_np = np.concatenate(
                            accumulated_audio_data, axis=0
                        )
                        # logger.debug(f"WASAPI Loop (Idx {dev_idx}): Accumulated {frames_accumulated} frames. Calling audiocapture_callback_fn.")
                        self.audiocapture_callback_fn(
                            final_segment_np, frames_accumulated, time.time(), None
                        )
                        accumulated_audio_data = []
                        frames_accumulated = 0

                except IOError as e:
                    if not self.active or self.stop_event.is_set():
                        logger.info(
                            f"WASAPI _threaded_read_loop (Idx {dev_idx}): Expected IOError during stop/close. Exiting. Error: {e}"
                        )
                        break
                    logger.warning(
                        f"IOError in WASAPI read loop (Idx {dev_idx}) while active: {e}"
                    )
                    consecutive_errors += 1
                    time.sleep(0.1 * consecutive_errors)
                    if consecutive_errors > 10:
                        logger.error(
                            f"Too many IOErrors in WASAPI (Idx {dev_idx}). Terminating."
                        )
                        self.active = False
                        break
                except Exception as e:
                    logger.error(
                        f"Generic error in WASAPI read loop (Idx {dev_idx}): {e}",
                        exc_info=True,
                    )
                    consecutive_errors += 1
                    time.sleep(0.5)
                    if consecutive_errors > 5:
                        logger.error(
                            f"Too many generic errors in WASAPI (Idx {dev_idx}). Terminating."
                        )
                        self.active = False
                        break

            if (
                accumulated_audio_data and frames_accumulated > 0 and self.active
            ):  # Process remaining if still active (e.g. stop called mid-accumulation)
                logger.info(
                    f"WASAPI Loop (Idx {dev_idx}): Processing remaining {frames_accumulated} accumulated frames on exit/stop."
                )
                final_segment_np = np.concatenate(accumulated_audio_data, axis=0)
                self.audiocapture_callback_fn(
                    final_segment_np, frames_accumulated, time.time(), "partial_on_exit"
                )

            logger.info(
                f"WASAPI _threaded_read_loop finished for paIdx {dev_idx} | Thread ID: {threading.get_ident()}. Active: {self.active}, StopSet: {self.stop_event.is_set()}"
            )

        def stop(self):
            # ... (Implementation from your last complete file, ensure logging includes dev_idx) ...
            dev_idx = self.loopback_device_info["index"]
            logger.debug(f"WasapiLoopbackStream.stop() called for paIdx {dev_idx}.")
            self.active = False
            if hasattr(self, "stop_event"):
                self.stop_event.set()

            if hasattr(self, "stream_pa") and self.stream_pa:
                try:
                    if self.stream_pa.is_active():
                        logger.debug(
                            f"Stopping PyAudio stream_pa in stop() for paIdx {dev_idx}."
                        )
                        self.stream_pa.stop_stream()
                    logger.debug(
                        f"Closing PyAudio stream_pa in stop() for paIdx {dev_idx}."
                    )
                    self.stream_pa.close()
                except Exception as e:
                    logger.error(
                        f"Error stopping/closing stream_pa in WasapiLoopbackStream.stop() for paIdx {dev_idx}: {e}",
                        exc_info=True,
                    )
                finally:
                    self.stream_pa = None

            if (
                hasattr(self, "read_thread")
                and self.read_thread
                and self.read_thread.is_alive()
            ):
                logger.debug(f"Joining read_thread for paIdx {dev_idx}...")
                self.read_thread.join(timeout=1.5)
                if self.read_thread.is_alive():
                    logger.warning(
                        f"WasapiLoopbackStream read_thread (paIdx {dev_idx}) did not join in time during stop()."
                    )
            else:
                logger.debug(
                    f"read_thread for paIdx {dev_idx} already stopped or not initialized."
                )
            logger.debug(f"WasapiLoopbackStream.stop() completed for paIdx {dev_idx}.")

        def close(self):
            # ... (Implementation from your last complete file, ensure logging includes dev_idx) ...
            dev_idx = self.loopback_device_info["index"]
            logger.debug(
                f"WasapiLoopbackStream.close() called for paIdx {dev_idx}, invoking self.stop()."
            )
            self.stop()

            if hasattr(self, "pyaudio_instance") and self.pyaudio_instance:
                try:
                    logger.debug(
                        f"Terminating PyAudio instance in WasapiLoopbackStream.close() for paIdx {dev_idx}."
                    )
                    self.pyaudio_instance.terminate()
                except Exception as e_pa_term:
                    logger.error(
                        f"Error terminating PyAudio instance in WasapiLoopbackStream.close() for paIdx {dev_idx}: {e_pa_term}",
                        exc_info=True,
                    )
                finally:
                    self.pyaudio_instance = None
            logger.debug(f"WasapiLoopbackStream.close() completed for paIdx {dev_idx}.")

        def is_active(self) -> bool:
            return (
                self.active
                and self.stream_pa is not None
                and self.stream_pa.is_active()
            )

    # --- End of WasapiLoopbackStream Class ---

    def _numpy_to_bytes(
        self, audio_data_int16: np.ndarray, channels: int
    ) -> Optional[bytes]:
        # ... (Implementation from your last complete file) ...
        if audio_data_int16 is None or audio_data_int16.size == 0:
            return None
        if audio_data_int16.dtype != np.int16:
            logger.error(
                f"_numpy_to_bytes expected int16, got {audio_data_int16.dtype}."
            )
            return None

        buffer = BytesIO()
        try:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(self.target_sample_rate)
                wf.writeframes(audio_data_int16.tobytes())
            wav_bytes = buffer.getvalue()
            return wav_bytes if len(wav_bytes) > 44 else None
        except Exception as e:
            logger.error(f"Error in _numpy_to_bytes: {e}", exc_info=True)
            return None
        finally:
            buffer.close()


if __name__ == "__main__":

    logger.info("Starting AudioCapture module test from __main__...")
    candidate_mic_idx = None
    ac_instance = None
    try:
        ac_instance = AudioCapture(
            sample_rate=16000, chunk_duration=0.2, channels=1
        )  # Using VAD friendly chunk

        logger.info("Full device list from AudioCapture instance:")
        all_devices_info = ac_instance.get_device_list()
        if not all_devices_info:
            logger.error("No audio devices found. Cannot proceed with test.")
            exit()

        candidate_mic_idx = next(
            (
                dev["index"]
                for dev in all_devices_info
                if dev["max_input_channels"] > 0
                and (
                    "microphone" in dev["name"].lower() or "mic" in dev["name"].lower()
                )
            ),
            None,
        )
        if candidate_mic_idx is None:
            candidate_mic_idx = next(
                (
                    dev["index"]
                    for dev in all_devices_info
                    if dev["max_input_channels"] > 0
                ),
                None,
            )

        if candidate_mic_idx is None:
            logger.error(
                "No input devices found for candidate microphone. Test cannot proceed effectively."
            )
            exit()
        else:
            cand_dev_name = next(
                dev["name"]
                for dev in all_devices_info
                if dev["index"] == candidate_mic_idx
            )
            logger.info(
                f"Auto-selected candidate microphone: '{cand_dev_name}' (sd_Idx {candidate_mic_idx})"
            )

        interviewer_loopback_idx = None
        if ac_instance.os_system == "windows":
            interviewer_loopback_idx = next(
                (
                    dev["index"]
                    for dev in all_devices_info
                    if dev["max_input_channels"] > 0
                    and any(
                        k in dev["name"].lower()
                        for k in ["stereo mix", "loopback", "what u hear"]
                    )
                ),
                None,
            )
            if interviewer_loopback_idx:
                int_dev_name = next(
                    dev["name"]
                    for dev in all_devices_info
                    if dev["index"] == interviewer_loopback_idx
                )
                logger.info(
                    f"Auto-selected sounddevice loopback for interviewer: '{int_dev_name}' (sd_Idx {interviewer_loopback_idx})"
                )
            else:
                logger.info(
                    "No obvious sounddevice loopback found, interviewer capture might rely on PyAudioWPatch or fail if not configured."
                )

        logger.info(
            "Starting audio capture for testing. Ensure system audio is playing for interviewer capture if loopback is used."
        )
        ac_instance.start(
            candidate_device_index=candidate_mic_idx,
            interviewer_device_index=interviewer_loopback_idx,
            online_interview=True,
        )

        logger.info("Capture started. Listening for 10 seconds...")
        test_duration = 10
        loop_start_time = time.time()
        tensors_received_count = {"candidate": 0, "interviewer": 0, "unknown": 0}

        while time.time() - loop_start_time < test_duration:
            tensor_data_dict = ac_instance.get_audio_tensor()
            if tensor_data_dict:
                spk = tensor_data_dict.get("speaker", "unknown")
                audio_array = tensor_data_dict.get("audio")
                tensors_received_count[spk] = tensors_received_count.get(spk, 0) + 1

                if tensors_received_count[spk] % 2 == 1 and audio_array is not None:
                    logger.info(
                        f"Tensor: Speaker='{spk}', Shape={audio_array.shape}, "
                        f"Dtype={audio_array.dtype}, TS={tensor_data_dict.get('timestamp', 0):.2f}, "
                        f"MaxVal={np.max(np.abs(audio_array)):.4f} (Rcvd: C:{tensors_received_count['candidate']}, I:{tensors_received_count['interviewer']})"
                    )
            time.sleep(0.02)

        logger.info(f"Test loop finished. Tensors received: {tensors_received_count}")

    except Exception as e:
        logger.error(f"Error during AudioCapture __main__ test: {e}", exc_info=True)
    finally:
        if ac_instance and ac_instance.is_capturing:
            logger.info("Stopping audio capture from __main__ finally block...")
            ac_instance.stop()
        logger.info("AudioCapture module __main__ test finished.")
