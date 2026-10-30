
import streamlit as st
import joblib
from pathlib import Path
from datetime import datetime
from streamlit_mic_recorder import speech_to_text

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="VoxMail AI",
    page_icon="📧",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "voxmail_model"

@st.cache_resource
def load_model():
    classifier = joblib.load(
        MODEL_DIR / "voxmail_classifier.pkl"
    )
    tfidf = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )
    return classifier, tfidf

model, vectorizer = load_model()

# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #10131c;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}
.subtitle {
    color: #aab3c5;
    font-size: 17px;
}
.result-card {
    background: #20283a;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #394762;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">📧 VoxMail AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Voice-Command Email Intent Classifier</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- MAIN LAYOUT ----------------

left, right = st.columns([2, 1])

with left:

    st.subheader("🎯 Command Recognition")

    command = st.text_area(
        "Enter your email command",
        placeholder="Example: Open my inbox",
        height=130
    )

    st.write("### 🎙️ Or speak your command")

    spoken_text = speech_to_text(
        language="en",
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="voxmail_microphone"
    )

    if spoken_text:
        st.success("Speech recognised!")
        st.write("Recognised command:", spoken_text)

    predict_button = st.button(
        "🚀 Predict Intent",
        use_container_width=True
    )

    command_to_predict = spoken_text if spoken_text else command

    if predict_button or spoken_text:

        if command_to_predict.strip():

            features = vectorizer.transform([command_to_predict])

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            confidence = max(probabilities) * 100
      
            if confidence < 30:
                prediction = "UNKNOWN"

            st.session_state.history.insert(0, {
                "command": command_to_predict,
                "intent": prediction,
                "confidence": round(confidence, 2),
                "time": datetime.now().strftime("%H:%M:%S")
            })

            st.markdown("### Prediction Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <p>PREDICTED INTENT</p>
                    <h2>{prediction}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if confidence < 40:
                st.warning("Low confidence. Review this prediction.")
            elif confidence < 65:
                st.info("Moderate confidence. Check the prediction.")
            else:
                st.success("Relatively confident prediction.")

        else:
            st.warning("Please enter or speak an email command.")
with right:

    st.subheader("📋 Supported Intents")

    intent_descriptions = {
        "OPEN_INBOX": "Open or list the inbox",
        "READ_EMAIL": "Read an email",
        "COMPOSE_EMAIL": "Create a new email",
        "SEND_EMAIL": "Send a prepared email",
        "REPLY_EMAIL": "Prepare a reply draft",
        "DELETE_EMAIL": "Delete an email",
        "SEARCH_EMAIL": "Find specific emails",
        "CANCEL": "Cancel an action",
        "UNKNOWN": "Unsupported or unrelated command"
    }

    for intent, description in intent_descriptions.items():
        st.markdown(f"**{intent}**")
        st.caption(description)

# ---------------- COMMAND HISTORY ----------------

st.divider()

st.subheader("🕘 Command History")

if st.session_state.history:

    for item in st.session_state.history:

        with st.expander(
            f"{item['time']} | {item['intent']}"
        ):
            st.write("**Command:**", item["command"])
            st.write("**Intent:**", item["intent"])
            st.write(
                "**Confidence:**",
                f"{item['confidence']:.2f}%"
            )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("Your predictions will appear here.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "VoxMail AI | MCA Minor Project | "
    "TF-IDF + Logistic Regression"
)
