"""Template-based clinician and patient notes for VoxSense referrals."""

def generate_notes(
    patient_id: str,
    score: float,
    domain: str = "Speech monitoring",
    patient_language: str = "en-IN",
) -> dict:
    """Generate non-diagnostic notes. LLM integration can replace this later."""

    clinician_note = (
        f"Participant: {patient_id}\n"
        f"Monitoring domain: {domain}\n"
        f"Risk score: {score:.2f} (0–1 scale)\n\n"
        "The latest voice-monitoring session crossed the configured "
        "referral threshold. Please review the participant's longitudinal "
        "speech measurements and recording history, and determine whether "
        "a clinical follow-up is appropriate. This automated score is a "
        "screening indicator only and is not a diagnosis."
    )

    messages = {
        "en-IN": (
            "Your recent voice-monitoring result suggests that it may be "
            "helpful to discuss your results with a healthcare professional. "
            "This is not a diagnosis. Please contact your healthcare provider "
            "if you have concerns."
        ),
        "hi-IN": (
            "आपकी हाल की आवाज़ की जाँच के परिणामों के बारे में किसी "
            "स्वास्थ्य पेशेवर से बात करना उपयोगी हो सकता है। यह कोई "
            "निदान नहीं है। यदि आपको चिंता है, तो कृपया अपने स्वास्थ्य "
            "सेवा प्रदाता से संपर्क करें।"
        ),
    }

    patient_message = messages.get(
        patient_language,
        messages["en-IN"],
    )

    return {
        "clinician_note": clinician_note,
        "patient_message": patient_message,
        "patient_language": patient_language,
    }
