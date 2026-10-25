
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# TRUTHLENS AI — Evidence-Based AI Verification Platform
# =====================================================

MODEL_NAME = "FacebookAI/roberta-large-mnli"

st.set_page_config(
    page_title="TRUTHLENS AI | Claim Verification",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# =====================================================
# LABEL MAPPING
# =====================================================

def normalize_label(label, index):
    label = str(label).lower()

    if "contradiction" in label:
        return "Contradiction"

    if "entailment" in label:
        return "Entailment"

    if "neutral" in label:
        return "Neutral"

    # Fallback for standard MNLI label ordering.
    fallback = {
        0: "Contradiction",
        1: "Neutral",
        2: "Entailment"
    }

    if label.startswith("label_"):
        return fallback.get(index, f"Unknown {label}")

    return f"Unknown {label}"


# =====================================================
# CLAIM VERIFICATION
# =====================================================

def verify_claim(evidence, claim, tokenizer, model):
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    raw_labels = model.config.id2label

    scores = {
        normalize_label(raw_labels.get(i, f"LABEL_{i}"), i):
        probabilities[i].item()
        for i in range(len(probabilities))
    }

    prediction = max(scores, key=scores.get)

    return prediction, scores


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🔎 TRUTHLENS AI")
    st.caption("Evidence-based AI verification")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        """
        **1.** Provide reference evidence.

        **2.** Enter a claim.

        **3.** Compare the claim against the evidence.

        **4.** Review the model's prediction and scores.
        """
    )

    st.divider()

    st.markdown("### Technology")
    st.markdown(
        """
        - Python
        - Streamlit
        - PyTorch
        - Hugging Face Transformers
        - RoBERTa Large MNLI
        """
    )

    st.divider()

    st.caption(
        "Research prototype. Results are predictions, not guarantees."
    )


# =====================================================
# MAIN PAGE
# =====================================================

st.title("🔎 TRUTHLENS AI")
st.subheader("Evidence-Based AI Evaluation & Verification Platform")

st.markdown("### Every claim deserves evidence.")

st.write(
    "Evaluate whether a claim is supported by, contradicted by, "
    "or not established by a reference passage."
)

st.info(
    "TRUTHLENS evaluates the relationship between supplied evidence "
    "and a claim. It does not independently establish whether the "
    "source itself is accurate or trustworthy."
)


# =====================================================
# EXAMPLE INPUTS
# =====================================================

st.markdown("### Try an example")

example = st.selectbox(
    "Choose a sample test",
    [
        "Custom input",
        "Supported claim",
        "Contradicted claim",
        "Insufficient evidence"
    ]
)

examples = {
    "Supported claim": (
        "Mercury is the closest planet to the Sun.",
        "Mercury is the nearest planet to the Sun."
    ),
    "Contradicted claim": (
        "Mercury is the closest planet to the Sun.",
        "Venus is the closest planet to the Sun."
    ),
    "Insufficient evidence": (
        "Mercury is the closest planet to the Sun.",
        "Mercury has two moons."
    )
}

default_evidence, default_claim = examples.get(
    example,
    ("", "")
)


# =====================================================
# INPUT FORM
# =====================================================

with st.form("verification_form"):
    st.markdown("### Enter your information")

    evidence = st.text_area(
        "Reference evidence",
        value=default_evidence,
        placeholder="Enter the reference passage...",
        height=140
    )

    claim = st.text_area(
        "Claim to verify",
        value=default_claim,
        placeholder="Enter the claim you want to check...",
        height=110
    )

    submitted = st.form_submit_button(
        "🔍 Verify claim",
        type="primary",
        use_container_width=True
    )


# =====================================================
# VERIFICATION REPORT
# =====================================================

if submitted:
    evidence = evidence.strip()
    claim = claim.strip()

    if not evidence or not claim:
        st.warning(
            "Please enter both reference evidence and a claim."
        )

    else:
        try:
            with st.spinner(
                "Loading the model and evaluating your claim..."
            ):
                tokenizer, model = load_model()

                prediction, scores = verify_claim(
                    evidence,
                    claim,
                    tokenizer,
                    model
                )

            st.divider()
            st.header("Verification report")

            if prediction == "Entailment":
                st.success("✅ SUPPORTED BY EVIDENCE")
                explanation = (
                    "The model predicts that the evidence supports the claim."
                )

            elif prediction == "Contradiction":
                st.error("❌ CONTRADICTED BY EVIDENCE")
                explanation = (
                    "The model predicts that the evidence conflicts with the claim."
                )

            elif prediction == "Neutral":
                st.warning("⚠️ INSUFFICIENT EVIDENCE")
                explanation = (
                    "The model predicts that the evidence does not establish "
                    "the claim."
                )

            else:
                st.warning(f"Unrecognized prediction: {prediction}")
                explanation = "The model returned an unknown category."

            st.write(explanation)

            st.caption(
                "A classification score is not the probability that a claim "
                "is true. Model predictions may be incorrect."
            )

            # ---------------- SUMMARY ----------------

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted category",
                    prediction
                )

            with col2:
                st.metric(
                    "Top classification score",
                    f"{scores[prediction]:.2%}"
                )

            # ---------------- SCORES ----------------

            st.subheader("Model prediction scores")

            score_df = pd.DataFrame({
                "Category": list(scores.keys()),
                "Score": list(scores.values())
            })

            # Simple, compatible chart format
            st.bar_chart(
                score_df.set_index("Category")["Score"]              
            )

            # Exact scores
            st.markdown("### Score breakdown")

            score_columns = st.columns(len(scores))

            for column, (label, score) in zip(
                score_columns,
                scores.items()
            ):
                with column:
                    st.metric(
                        label,
                        f"{score:.2%}"
                    )
                    st.progress(float(score))

            # ---------------- EVIDENCE DETAILS ----------------

            with st.expander("📄 Review evidence and claim"):
                st.markdown("**Reference evidence**")
                st.write(evidence)

                st.markdown("**Claim evaluated**")
                st.write(claim)

            # ---------------- METHODOLOGY ----------------

            with st.expander("ℹ️ What do the categories mean?"):
                st.markdown(
                    """
                    **Supported (entailment)**  
                    The model predicts that the reference supports the claim.

                    **Contradicted**  
                    The model predicts that the reference conflicts with
                    the claim.

                    **Insufficient evidence (neutral)**  
                    The model predicts that the reference neither establishes
                    nor directly contradicts the claim.

                    These are natural-language inference predictions.
                    They are not guarantees of factual correctness.
                    """
                )

        except Exception as error:
            st.error("Verification failed.")
            st.code(
                f"{type(error).__name__}: {error}",
                language="text"
            )
            st.info(
                "Check your terminal for additional diagnostic details."
            )


# =====================================================
# PROJECT INFORMATION
# =====================================================

st.divider()

with st.expander("About TRUTHLENS AI"):
    st.markdown(
        """
        ### Project mission

        Explore techniques for making AI-generated answers more
        transparent through evidence-based verification.

        ### Current capabilities

        - Claim-versus-evidence comparison
        - Three-category NLI classification
        - Model classification score visualization
        - Interactive verification interface
        - Sample test cases

        ### Current limitations

        - The user supplies the reference evidence.
        - Source credibility is not independently verified.
        - Claims may be misclassified.
        - Performance must be evaluated on a larger, diverse dataset.

        ### Technology stack

        Python · Streamlit · PyTorch · Hugging Face Transformers
        """
    )

st.caption(
    "TRUTHLENS AI | Open-source research prototype | "
    "Always independently verify important claims."
)
