
# VoxSense - Evaluation Notes

## 1. Gnani Prisma v3 Hindi STT Evaluation

**Dataset:** Mozilla Common Voice Scripted Speech 27.0
**Language:** Hindi (`hi-IN`)
**Recordings:** 10 clips from 10 unique contributors
**Model:** Gnani Prisma v3.0
**Evaluation date:** 8 November 2026

### Results

- Successful transcriptions: 10/10
- Failed transcriptions: 0
- Total reference words: 80
- Substitutions: 7
- Deletions: 1
- Insertions: 3
- Overall Word Error Rate (WER): 13.75%
- Average API processing time: 0.2906 seconds

WER was calculated after Unicode normalization, lowercasing,
punctuation removal, and whitespace normalization.

### Observations

- All 10 MP3 recordings were accepted by the Gnani STT API.
- All recordings were detected as Hindi (`hi-IN`).
- Some clips had exact normalized word matches.
- The highest individual WER was 50%.
- Most observed errors were substitutions.

This is a small preliminary evaluation and should not be
treated as a general Hindi transcription accuracy benchmark.

## 2. Clean vs Degraded Audio Test

**Test recording:** `data/sample.wav`

Three versions were tested:

| Version | Detected language | Transcript match |
|---|---|---|
| Original 16 kHz | hi-IN | Matched |
| Resampled 8 kHz | hi-IN | Matched |
| Noisy audio (10 dB SNR) | hi-IN | Matched |

All three versions produced the same transcript:

`ओ हैलो, हैलो, माइक टेस्ट वन टू थ्री।`

### Observations

- Gnani successfully processed the original recording.
- The 8 kHz resampled recording was also transcribed successfully.
- The 10 dB SNR noisy recording produced the same transcript.
- This is only a single-recording robustness check and does not establish general noise robustness.

### 2.1 Common Voice Batch Degradation Evaluation

**Dataset:** Mozilla Common Voice Scripted Speech 27.0 (Hindi)
**Recordings:** 10 clips from 10 unique contributors
**Model:** Gnani Prisma v3.0
**Evaluation date:** 8 November 2026

Each original recording was evaluated in three conditions:
- Original audio (clean)
- Resampled audio at 8 kHz
- Artificially degraded audio with white Gaussian noise at 10 dB SNR

| Condition | Successful | Failed | Overall WER |
|---|---:|---:|---:|
| Clean | 10/10 | 0 | 13.75% |
| 8 kHz | 10/10 | 0 | 16.25% |
| Noisy (10 dB SNR) | 10/10 | 0 | 23.75% |

**8 kHz results:**
- Reference words: 80
- Substitutions: 10
- Deletions: 1
- Insertions: 2
- Average API processing time: 0.1202 seconds

**Noisy audio results (10 dB SNR):**
- Reference words: 80
- Substitutions: 14
- Deletions: 2
- Insertions: 3
- Average API processing time: 0.1160 seconds

### Observations

- All 20 degraded audio files were successfully transcribed.
- 8 kHz audio had a WER increase of 2.5 percentage points compared with clean audio.
- Audio with 10 dB SNR noise had a WER increase of 10 percentage points.
- The results suggest that added noise affected transcription more than resampling to 8 kHz in this sample.
- These findings are preliminary and are not a general Hindi ASR robustness benchmark.

## 3. Timestamp Availability

The inspected Gnani REST response included the transcript,
detected language, model, request ID, processing time,
and end-to-end latency.

It did not include word-level timestamps.

The pipeline therefore records timestamp availability as
`not_available_from_rest` instead of generating artificial timestamps.

## 4. Limitations and Next Steps

- Expand evaluation beyond 10 public recordings.
- Test more noise levels and recording conditions.
- Collect additional consent-based recordings across multiple sessions.
- Improve pause detection using a more robust voice activity detector.
- Evaluate acoustic feature stability across recording sessions.


## 5. Data Collection Status

### Public Dataset

- Source: Mozilla Common Voice Scripted Speech 27.0
- Language: Hindi
- Recordings selected: 10
- Unique contributors: 10
- Purpose: STT evaluation and benchmarking

### Self-Recorded Dataset

- Audio files collected: 5
- Unique audio recordings: 4
- Duplicate files identified: 1
- Processing: 5/5 files successfully processed
- Format: 16 kHz, mono WAV
- STT model: Gnani Prisma v3.0
- Purpose: Initial speaker-specific acoustic feature exploration

The duplicate recording was excluded from the initial baseline.
Original recordings and generated WAV files are kept locally
and are not included in the public repository.

The initial descriptive baseline is saved locally at
`data/processed/baseline.json`.

The baseline is preliminary and requires additional recordings
across multiple days before meaningful longitudinal comparisons.

### Next Steps

- Collect additional consent-based recordings across multiple sessions.
- Improve pause detection using a more robust voice activity detector.
- Evaluate feature stability across recording sessions.
- Expand the public STT evaluation dataset.
