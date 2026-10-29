
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image as PILImage, UnidentifiedImageError

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.groq import Groq


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Image Analysis Tool",
    page_icon="🩺",
    layout="centered",
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "gif"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# 2. AI AGENT
# --------------------------------------------------

@st.cache_resource
def get_medical_agent():
    """Create and reuse the Groq vision agent."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return Agent(
        model=Groq(
            id="qwen/qwen3.8-27b",
            api_key=GROQ_API_KEY,
        ),
        markdown=True,
    )


# --------------------------------------------------
# 3. MEDICAL ANALYSIS INSTRUCTIONS
# --------------------------------------------------

QUERY = """
You are an educational medical-imaging assistant.

Analyze the supplied image cautiously. Do not claim to be a
radiologist or provide a definitive medical diagnosis.

Use these sections:

1. Image Type & Region
   - Identify the likely modality and anatomy when possible.
   - Describe positioning and image quality.
   - State uncertainty if the modality or region is unclear.

2. Key Findings
   - Describe visible features systematically.
   - Separate direct observations from interpretations.
   - Do not invent measurements or findings.

3. Diagnostic Assessment
   - Give cautious possible interpretations and differentials.
   - Explain confidence qualitatively, not as fabricated percentages.
   - State when the image is insufficient for assessment.
   - Highlight possible urgent findings that need prompt evaluation.

4. Patient-Friendly Explanation
   - Explain the observations in plain language.
   - Do not imply that an AI-generated interpretation is confirmed.

5. Educational Context
   - Explain relevant general clinical concepts.
   - Do not invent literature searches, citations, or treatment advice.

This is an educational prototype, not a diagnostic device.
A qualified medical professional must review all findings.
For severe symptoms or a possible emergency, seek urgent medical care.
"""


# --------------------------------------------------
# 4. IMAGE ANALYSIS FUNCTION
# --------------------------------------------------

def analyze_medical_image(image_file) -> str:
    """Validate, resize, and submit an image to the vision model."""

    temp_path = None

    try:
        if not GROQ_API_KEY:
            return (
                "Configuration error: GROQ_API_KEY is missing. "
                "Check your .env file."
            )

        image_file.seek(0)

        if image_file.size > MAX_FILE_SIZE:
            return "File too large. Please upload an image under 10 MB."

        # Validate the uploaded image.
        try:
            with PILImage.open(image_file) as source:
                source.verify()

            image_file.seek(0)

            with PILImage.open(image_file) as source:
                image = source.convert("RGB")
                image.thumbnail((1200, 1200))

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name

                image.save(
                    temp_path,
                    format="JPEG",
                    quality=90,
                )

        except (UnidentifiedImageError, OSError):
            return (
                "Invalid image. Please upload a valid JPG, PNG, "
                "BMP, or GIF file."
            )

        # Send the image to the AI agent.
        agent = get_medical_agent()

        response = agent.run(
            QUERY,
            images=[AgnoImage(filepath=temp_path)],
        )

        if not response or not response.content:
            return "The AI model returned an empty response."

        return str(response.content)

    except Exception as exc:
        # Show a useful error without exposing environment secrets.
        error_text = str(exc)
        return (
            "Analysis failed. Check your internet connection, "
            "Groq API key, model availability, account limits, "
            "and installed package versions.\n\n"
            f"Technical details: {error_text[:500]}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# --------------------------------------------------
# 5. STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🩺 Medical Image Analysis Tool")

st.caption(
    "AI-assisted educational analysis of medical images"
)

st.warning(
    "Educational use only. This tool is not a substitute "
    "for a radiologist or qualified medical professional."
)

st.info(
    "Supported formats: JPG, JPEG, PNG, BMP, and GIF. "
    "DICOM files are not directly supported in this version."
)

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=SUPPORTED_TYPES,
    help="Maximum file size: 10 MB.",
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded medical image",
        use_container_width=True,
    )

    if st.button("Analyze Image", type="primary"):

        with st.spinner("Analyzing image. Please wait..."):
            report = analyze_medical_image(uploaded_file)

        st.subheader("Analysis Report")
        st.markdown(report)

        st.caption(
            "This AI-generated report may contain errors. "
            "It must not be used alone to make clinical decisions."
        )

else:
    st.info("Upload an image to begin.")
