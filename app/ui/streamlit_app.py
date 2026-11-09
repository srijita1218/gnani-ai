"""VoxSense UI (Streamlit).

Run from the repo root:
    streamlit run app/ui/streamlit_app.py

Pages: 1. Consent + language  2. Record / upload  3. Patient (sessions, trend, risk badge)
Data source: Demo (built-in mock patients) or Live (the FastAPI backend).
"""
import sys
from pathlib import Path

# make `import app...` work when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from app.ui import api_client
from app.ui.api_client import ApiError
from app.ui.logic import (
    BASELINE_SESSIONS,
    DEFAULT_LANGUAGE,
    FEATURE_KEYS,
    FEATURE_LABELS,
    LANGUAGES,
    RISK_MESSAGES,
    RISK_STYLES,
    badge_html,
    compute_baseline,
    latest_risk_score,
    risk_level,
    sessions_to_frame,
)
from app.ui.mock_data import MOCK_PATIENTS

DEMO = "Demo (mock data)"
LIVE = "Live backend"
PAGES = ["1. Consent", "2. Record", "3. Patient"]

CONSENT_TEXT = """
**Please read before you start**

- This is a **research prototype**. It looks at how your speech changes over time.
  It is **not a medical test** and cannot diagnose any condition.
- You will record or upload short voice samples. The app measures things like
  speaking speed, pauses, pitch and voice quality, and compares them with **your own**
  earlier recordings.
- Your audio is sent to **Gnani's Prisma speech-to-text service** to get a transcript.
  The app does not keep the audio file; it keeps the measurements and the transcript
  in a local database.
- Use a **participant code** (for example P001), not your real name.
- If the app suggests a follow-up, that is a prompt to talk to a healthcare
  professional, not a diagnosis.
"""

PROMPTS = {
    "en-IN": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
    "hi-IN": "आज मौसम बहुत अच्छा है और मैं अपनी आवाज़ रिकॉर्ड कर रहा हूँ।",
}
DEFAULT_PROMPT = PROMPTS["en-IN"]


# ------------------------------------------------------------------ state --

def init_state():
    st.session_state.setdefault("consented", False)
    st.session_state.setdefault("participant_id", "P001")
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)


def language_label(code: str) -> str:
    return next((label for label, c in LANGUAGES.items() if c == code), code)


# ---------------------------------------------------------------- sidebar --

def sidebar():
    st.sidebar.title("🎙️ VoxSense")
    st.sidebar.caption("Research prototype for speech-change monitoring. Not a diagnostic tool.")

    mode = st.sidebar.radio("Data source", [DEMO, LIVE])
    api_url = api_client.DEFAULT_API_URL
    if mode == LIVE:
        api_url = st.sidebar.text_input("Backend URL", value=api_client.DEFAULT_API_URL).rstrip("/")
        if st.sidebar.button("Check backend"):
            try:
                api_client.health(api_url)
                st.sidebar.success("Backend is running")
            except ApiError as err:
                st.sidebar.error(err.message)

    page = st.sidebar.radio("Page", PAGES)

    st.sidebar.divider()
    if st.session_state.consented:
        st.sidebar.write(f"Participant: **{st.session_state.participant_id}**")
        st.sidebar.write(f"Language: **{language_label(st.session_state.language)}**")
    else:
        st.sidebar.write("Consent: not given yet")
    return mode, api_url, page


# ----------------------------------------------------------- 1. consent --

def consent_page(mode, api_url):
    st.header("1. Consent and language")
    st.markdown(CONSENT_TEXT)

    participant_id = st.text_input(
        "Participant ID (a code, not your real name)", value=st.session_state.participant_id
    ).strip()

    codes = list(LANGUAGES.values())
    current = codes.index(st.session_state.language) if st.session_state.language in codes else 0
    label = st.selectbox("Language you will speak", list(LANGUAGES), index=current)

    agree = st.checkbox("I have read the above and I agree to take part.")

    if st.button("Continue", type="primary", disabled=not agree):
        if not participant_id:
            st.error("Please enter a participant ID.")
            return
        if mode == LIVE:
            try:
                api_client.create_participant(api_url, participant_id)
            except ApiError as err:
                if err.status == 409:
                    st.info(f"{participant_id} is already registered. Continuing.")
                else:
                    st.error(err.message)
                    return
        st.session_state.consented = True
        st.session_state.participant_id = participant_id
        st.session_state.language = LANGUAGES[label]
        st.success("Consent recorded. Open **2. Record** in the sidebar.")


# ------------------------------------------------------------- 2. record --

def record_page(mode, api_url):
    st.header("2. Record or upload")
    if not st.session_state.consented:
        st.warning("Please complete **1. Consent** first.")
        return

    language = st.session_state.language
    st.write(f"Participant **{st.session_state.participant_id}** · language **{language_label(language)}**")
    st.info("Please read aloud, in your normal voice:\n\n> " + PROMPTS.get(language, DEFAULT_PROMPT))

    record_tab, upload_tab = st.tabs(["🎤 Record", "📁 Upload a file"])
    audio_bytes, filename = None, None

    with record_tab:
        recording = st.audio_input("Press the microphone, read the sentence, then stop")
        if recording is not None:
            audio_bytes, filename = recording.getvalue(), "recording.wav"

    with upload_tab:
        uploaded = st.file_uploader("WAV, MP3 or FLAC, up to 25 MB", type=["wav", "mp3", "flac"])
        if uploaded is not None:
            audio_bytes, filename = uploaded.getvalue(), uploaded.name
            st.audio(audio_bytes)

    if audio_bytes is None:
        return

    if mode == DEMO:
        st.info("Demo mode: audio is not sent anywhere. Switch **Data source** to *Live backend* to submit.")
        return

    if st.button("Submit recording", type="primary"):
        with st.spinner("Uploading and analysing…"):
            try:
                result = api_client.post_audio(
                    api_url, st.session_state.participant_id, language, filename, audio_bytes
                )
            except ApiError as err:
                st.error(f"Could not save the recording: {err.message}")
                return
        show_result(result)


def show_result(result: dict):
    st.success(f"Saved session {result['session_id']}")
    transcription = result.get("transcription") or {}
    if transcription.get("text"):
        st.write("**Transcript:**", transcription["text"])
    else:
        st.info("No transcript (speech-to-text unavailable). The voice measurements were still saved.")
    features = result.get("features") or {}
    table = pd.DataFrame(
        {"Feature": [FEATURE_LABELS[k] for k in FEATURE_KEYS],
         "Value": [features.get(k) for k in FEATURE_KEYS]}
    )
    st.dataframe(table, hide_index=True)
    st.caption("Open **3. Patient** to see this session in the trend.")


# ------------------------------------------------------------ 3. patient --

def trend_chart(df: pd.DataFrame, feature: str, baseline):
    label = FEATURE_LABELS[feature]
    data = df[["timestamp", "date", feature]].dropna(subset=[feature])

    line = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Session date"),
            y=alt.Y(f"{feature}:Q", title=label, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip(f"{feature}:Q", title=label)],
        )
    )
    layers = [line]

    if baseline is not None:
        band = pd.DataFrame({
            "start": [data["timestamp"].min()],
            "end": [data["timestamp"].max()],
            "low": [baseline["mean"] - baseline["std"]],
            "high": [baseline["mean"] + baseline["std"]],
            "mean": [baseline["mean"]],
        })
        layers.insert(0, alt.Chart(band).mark_rect(opacity=0.18, color="#1f77b4").encode(
            x="start:T", x2="end:T", y="low:Q", y2="high:Q"))
        layers.insert(1, alt.Chart(band).mark_rule(strokeDash=[6, 4], color="#1f77b4").encode(y="mean:Q"))

    return alt.layer(*layers).properties(height=340)


def patient_page(mode, api_url):
    st.header("3. Patient")

    if mode == DEMO:
        pid = st.selectbox("Patient", list(MOCK_PATIENTS), format_func=lambda k: MOCK_PATIENTS[k]["label"])
        sessions = MOCK_PATIENTS[pid]["sessions"]
        st.caption("Showing built-in mock data.")
    else:
        pid = st.text_input("Participant ID", value=st.session_state.participant_id).strip()
        try:
            sessions = api_client.get_sessions(api_url, pid)
        except ApiError as err:
            st.error(err.message)
            return
        if not sessions:
            st.info("No sessions for this participant yet. Record one on page 2.")
            return

    df = sessions_to_frame(sessions)

    # risk badge
    score = latest_risk_score(df)
    level = risk_level(score)
    st.subheader("Current status")
    st.markdown(badge_html(level, score), unsafe_allow_html=True)
    st.write("")
    st.write(RISK_MESSAGES[level])
    st.caption("Screening indicator only. It is not a diagnosis.")

    # trend chart
    st.subheader("Trend against personal baseline")
    available = [k for k in FEATURE_KEYS if df[k].notna().any()]
    if not available:
        st.info("No feature values to chart yet.")
    else:
        default = available.index("speaking_rate") if "speaking_rate" in available else 0
        feature = st.selectbox("Feature", available, index=default, format_func=lambda k: FEATURE_LABELS[k])
        baseline = compute_baseline(df, feature)
        st.altair_chart(trend_chart(df, feature, baseline))
        if baseline is None:
            st.caption(f"Baseline appears after {BASELINE_SESSIONS} sessions with this feature.")
        else:
            st.caption(
                f"Shaded band: baseline mean ± 1 SD of the first {baseline['n']} sessions "
                f"(mean {baseline['mean']:.3g})."
            )

    # session list
    st.subheader("Sessions")
    table = df[["date", "risk_score", "monitoring_status", "speaking_rate", "pause_ratio", "pitch_mean", "hnr_db"]]
    st.dataframe(
        table.rename(columns={
            "date": "Date", "risk_score": "Risk score", "monitoring_status": "Status",
            "speaking_rate": "Speaking rate", "pause_ratio": "Pause ratio",
            "pitch_mean": "Pitch (Hz)", "hnr_db": "HNR (dB)",
        }),
        hide_index=True,
    )

    with st.expander("Badge colours"):
        for key in ("low", "moderate", "high"):
            st.markdown(badge_html(key), unsafe_allow_html=True)
            st.caption(RISK_MESSAGES[key])


# ------------------------------------------------------------------- main --

def main():
    st.set_page_config(page_title="VoxSense", page_icon="🎙️", layout="wide")
    init_state()
    mode, api_url, page = sidebar()
    if page == PAGES[0]:
        consent_page(mode, api_url)
    elif page == PAGES[1]:
        record_page(mode, api_url)
    else:
        patient_page(mode, api_url)


main()
