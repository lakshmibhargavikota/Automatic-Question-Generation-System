import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import PyPDF2
from fpdf import FPDF
import wikipedia

st.set_page_config(page_title="Automatic Question Generation System")
st.title("📚 Automatic Question Generation System")

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    model_name = "iarfmoose/t5-base-question-generator"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    qg_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return qg_pipeline


qg_pipeline = load_model()

# ---------------- SESSION ----------------

if "questions" not in st.session_state:
    st.session_state.questions = []

# ---------------- PDF TEXT ----------------

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text

# ---------------- TOPIC TEXT ----------------

def get_topic_text(topic):
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(topic)
        return page.content[:4000]
    except:
        return ""

# ---------------- INPUT UI ----------------

topic_input = st.text_input("Enter Topic")

text_input = st.text_area("Enter Text (optional)")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

question_type = st.selectbox(
    "Question Type",
    ["Short Answer", "Long Answer"]
)

difficulty = st.selectbox(
    "Difficulty Level",
    ["Easy", "Medium", "Hard"]
)

num_questions = st.slider(
    "Number of Questions",
    1,
    20,
    10
)

# ---------------- SOURCE SELECTION ----------------

if pdf_file is not None:
    text_input = extract_text_from_pdf(pdf_file)

elif topic_input.strip() != "":
    topic_text = get_topic_text(topic_input)

    if topic_text == "":
        st.warning("Topic not found. Try another topic.")
    else:
        text_input = topic_text

# ---------------- GENERATE QUESTIONS ----------------

if st.button("Generate Questions"):

    if text_input.strip() == "":
        st.error("Please enter text, topic, or upload a PDF.")

    else:

        sentences = [
            s.strip()
            for s in text_input.split(".")
            if len(s.split()) > 6
        ]

        questions = []

        for sentence in sentences:

            if len(questions) >= num_questions:
                break

            if difficulty == "Easy":
                prompt = f"generate question: {sentence}"
            elif difficulty == "Medium":
                prompt = f"generate question: {sentence}"
            else:
                prompt = f"generate difficult question: {sentence}"

            try:
                result = qg_pipeline(
                    prompt,
                    max_new_tokens=64,
                    do_sample=False
                )

                question = result[0]["generated_text"].strip()

                if (
                    question
                    and question not in questions
                ):
                    questions.append(question)

            except Exception:
                continue

        st.session_state.questions = questions

        st.subheader("Generated Questions")

        if questions:
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")
        else:
            st.warning("No questions could be generated.")

# ---------------- DOWNLOAD PDF ----------------

if st.button("Download Questions PDF"):

    if not st.session_state.questions:
        st.warning("Generate questions first.")

    else:

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "Generated Questions", ln=True)

        for i, q in enumerate(st.session_state.questions, 1):
            safe_q = (
                f"{i}. {q}"
                .encode("latin-1", "ignore")
                .decode("latin-1")
            )

            pdf.multi_cell(0, 10, safe_q)
            pdf.ln(2)

        pdf.output("questions.pdf")

        with open("questions.pdf", "rb") as f:
            st.download_button(
                "⬇ Download PDF",
                data=f,
                file_name="questions.pdf",
                mime="application/pdf"
            )