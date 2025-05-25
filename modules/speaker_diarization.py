# In speaker_diarization.py

import logging
import os

import torch

# from pydub import AudioSegment # Not directly used in this method
from pyannote.audio import Pipeline

# Assuming torchaudio is available for resampling, if not, a warning or alternative is needed
try:
    import torchaudio.functional as F

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logging.warning(
        "torchaudio not available. Resampling in SpeakerDiarization might be limited or unavailable if input sample rate is not 16kHz."
    )

logger = logging.getLogger(__name__)


class SpeakerDiarization:
    """
    Identifies and distinguishes between different speakers in audio.
    Uses pyannote.audio for speaker diarization.
    """

    def __init__(
        self, model_name="pyannote/speaker-diarization-3.1", target_sample_rate=16000
    ):
        """
        Initialize the speaker diarization module.

        Args:
            model_name: Name of the pretrained diarization model
            target_sample_rate: The sample rate the pipeline expects (usually 16000 for pyannote models)
        """
        logger.info(f"Initializing speaker diarization with model: {model_name}")

        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            # Try to load from Hugging Face CLI login if available
            try:
                from huggingface_hub import HfFolder

                self.hf_token = HfFolder.get_token()
                if not self.hf_token:
                    raise ValueError(
                        "HF_TOKEN environment variable not set, and no token found via huggingface_hub. Please set your Hugging Face token."
                    )
                logger.info("Using Hugging Face token from HfFolder.")
            except ImportError:
                raise ValueError(
                    "HF_TOKEN environment variable not set. Please set your Hugging Face token or ensure huggingface_hub is installed and you are logged in."
                )
            except Exception as e:
                raise ValueError(f"HF_TOKEN error: {e}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for SpeakerDiarization: {self.device}")

        self.pipeline_sample_rate = (
            target_sample_rate  # The rate the pyannote pipeline expects
        )

        try:
            self.pipeline = Pipeline.from_pretrained(
                model_name, use_auth_token=self.hf_token
            )
            self.pipeline = self.pipeline.to(self.device)

            self.speaker_map = (
                {}
            )  # e.g., {'SPEAKER_00': 'interviewer', 'SPEAKER_01': 'candidate'}
            self.calibration_done = False
            self.calibration_samples = []  # Store segments for calibration
            self.min_calibration_utterances = (
                5  # Number of distinct utterances before attempting calibration
            )

            logger.info("Speaker diarization initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize speaker diarization: {e}", exc_info=True
            )
            raise

    def identify_speaker(
        self, audio_waveform_tensor: torch.Tensor, input_sample_rate: int
    ):
        """
        Identify the speaker in an audio chunk.

        Args:
            audio_waveform_tensor: Audio data as a torch tensor in (channels, time) format.
                                   Expected to be float32.
            input_sample_rate: Sample rate of the input audio_waveform_tensor.

        Returns:
            Dictionary with speaker information:
            {'speaker': 'interviewer'/'candidate'/raw_id/'unknown'/'error',
             'raw_speaker_id': raw_id or None,
             'confidence': float_confidence_score}
        """
        if not isinstance(audio_waveform_tensor, torch.Tensor):
            logger.error(
                f"identify_speaker expects a torch.Tensor, got {type(audio_waveform_tensor)}"
            )
            return {"speaker": "error", "raw_speaker_id": None, "confidence": 0.0}

        if (
            audio_waveform_tensor.ndim != 2 or audio_waveform_tensor.shape[0] > 2
        ):  # Expect (channels, time)
            logger.error(
                f"Input audio_waveform_tensor has unexpected shape: {audio_waveform_tensor.shape}. Expected (channels, time)."
            )
            return {"speaker": "error", "raw_speaker_id": None, "confidence": 0.0}

        # Ensure the tensor is on the correct device
        audio_waveform_tensor = audio_waveform_tensor.to(self.device)

        try:
            # Ensure audio is mono. If stereo, take the average.
            if audio_waveform_tensor.shape[0] == 2:  # Stereo
                logger.debug(
                    "Stereo audio detected, converting to mono by averaging channels."
                )
                audio_waveform_tensor = torch.mean(
                    audio_waveform_tensor, dim=0, keepdim=True
                )
            elif audio_waveform_tensor.shape[0] != 1:  # Not mono or stereo
                logger.error(
                    f"Unsupported number of channels: {audio_waveform_tensor.shape[0]}"
                )
                return {"speaker": "error", "raw_speaker_id": None, "confidence": 0.0}

            # Resample if necessary
            if input_sample_rate != self.pipeline_sample_rate:
                if TORCHAUDIO_AVAILABLE:
                    logger.debug(
                        f"Resampling audio from {input_sample_rate}Hz to {self.pipeline_sample_rate}Hz"
                    )
                    audio_waveform_tensor = F.resample(
                        audio_waveform_tensor,
                        orig_freq=input_sample_rate,
                        new_freq=self.pipeline_sample_rate,
                    )
                else:
                    logger.warning(
                        f"Cannot resample: torchaudio not available and input sample rate {input_sample_rate}Hz != pipeline rate {self.pipeline_sample_rate}Hz. Results may be poor."
                    )

            # Prepare payload for the pipeline
            # The pipeline expects a dictionary with 'waveform' (the tensor) and 'sample_rate'.
            diarization_input = {
                "waveform": audio_waveform_tensor,  # Should be (channels=1, time)
                "sample_rate": self.pipeline_sample_rate,
            }

            # Perform diarization
            with torch.no_grad():
                diarization = self.pipeline(diarization_input)

            # Process diarization results to find the most dominant speaker in this chunk
            speaker_durations = {}  # Stores duration for each speaker_id in this chunk
            total_speech_duration_in_chunk = 0.0
            for turn, _, speaker_id_raw in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_durations[speaker_id_raw] = (
                    speaker_durations.get(speaker_id_raw, 0.0) + duration
                )
                total_speech_duration_in_chunk += duration

            if not speaker_durations:
                return {"speaker": "unknown", "raw_speaker_id": None, "confidence": 0.0}

            # Determine the most dominant speaker raw ID for this chunk
            dominant_raw_id = max(speaker_durations, key=speaker_durations.get)
            confidence = (
                (speaker_durations[dominant_raw_id] / total_speech_duration_in_chunk)
                if total_speech_duration_in_chunk > 0
                else 0.0
            )

            # Add to calibration samples if calibration is not yet done
            if not self.calibration_done:
                # Store enough info for calibration (e.g., the raw ID and maybe a feature vector if doing more advanced calibration)
                self.calibration_samples.append(
                    {
                        "raw_speaker_id": dominant_raw_id,
                        "duration": speaker_durations[dominant_raw_id],
                    }
                )
                # Simple calibration: try to calibrate if enough distinct speaker turns have been observed
                # This is a placeholder for a more robust calibration trigger.
                # For a real system, you might need a UI button or explicit "I am interviewer", "I am candidate" steps.
                if (
                    len(set(s["raw_speaker_id"] for s in self.calibration_samples)) >= 2
                    and len(self.calibration_samples) >= self.min_calibration_utterances
                ):
                    self._calibrate_speakers()  # Attempt calibration

            # Map raw speaker ID to application role (interviewer/candidate) if calibration is done
            final_speaker_label = dominant_raw_id  # Default to raw ID
            if self.calibration_done and dominant_raw_id in self.speaker_map:
                final_speaker_label = self.speaker_map[dominant_raw_id]
            elif (
                self.calibration_done
            ):  # Calibrated, but this speaker ID is new or unmapped
                logger.warning(
                    f"Calibration done, but raw_id '{dominant_raw_id}' not in speaker_map: {self.speaker_map}. Treating as new speaker."
                )
                # Optionally, you could assign a new generic label like 'other_speaker' or try to dynamically add to map

            return {
                "speaker": final_speaker_label,
                "raw_speaker_id": dominant_raw_id,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(
                f"Error during speaker diarization processing: {e}", exc_info=True
            )
            return {"speaker": "error", "raw_speaker_id": None, "confidence": 0.0}

    def _calibrate_speakers(self):
        """
        Calibrate speakers based on collected samples.
        This is a simplified example. A robust system might need manual assignment
        or use characteristics of the first few interactions (e.g., interviewer speaks first).
        """
        logger.info(
            f"Attempting speaker calibration with {len(self.calibration_samples)} samples."
        )
        if not self.calibration_samples:
            logger.warning("No calibration samples to process.")
            return

        # Aggregate durations for each raw speaker ID
        aggregated_durations = {}
        for sample in self.calibration_samples:
            raw_id = sample["raw_speaker_id"]
            aggregated_durations[raw_id] = aggregated_durations.get(
                raw_id, 0.0
            ) + sample.get(
                "duration", 0.1
            )  # Add small default if no duration

        if len(aggregated_durations) < 2:
            logger.warning(
                f"Not enough distinct speakers ({len(aggregated_durations)}) detected for calibration. Need at least 2."
            )
            return

        # Sort speakers by total duration (simplistic assumption: longest speaker is interviewer, next is candidate)
        # This assumption is weak and likely needs improvement for real-world use.
        sorted_speakers_by_duration = sorted(
            aggregated_durations.items(), key=lambda item: item[1], reverse=True
        )

        # Assign roles
        # IMPORTANT: This is a very basic assumption.
        # In a real scenario, you might want the user to confirm or provide initial labels.
        # For example, assume the first distinct speaker to appear for a significant duration is the interviewer.

        interviewer_id = sorted_speakers_by_duration[0][0]
        candidate_id = sorted_speakers_by_duration[1][0]

        # Ensure they are not the same, though the distinct speaker check above should handle this.
        if (
            interviewer_id == candidate_id and len(sorted_speakers_by_duration) > 1
        ):  # Should not happen if len(aggregated_durations) >=2
            logger.error(
                "Calibration logic error: interviewer and candidate IDs are the same."
            )
            return

        self.speaker_map = {interviewer_id: "interviewer", candidate_id: "candidate"}
        # Map any other speakers to generic labels if needed
        for i, (other_id, _) in enumerate(sorted_speakers_by_duration[2:]):
            self.speaker_map[other_id] = f"other_speaker_{i}"

        self.calibration_done = True
        logger.info(f"Speaker calibration completed. Map: {self.speaker_map}")
        # self.calibration_samples.clear() # Optionally clear samples after calibration

    def reset_calibration(self):
        """Reset the speaker calibration state."""
        self.speaker_map = {}
        self.calibration_done = False
        self.calibration_samples = []
        logger.info("Speaker calibration reset.")
