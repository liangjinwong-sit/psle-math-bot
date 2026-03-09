import streamlit as st
from src.generation import answer_question
from src.practice import get_random_question, get_final_answer


def render_qa_tab():
    """Render the 'Ask a Math Question' tab."""
    st.header("Ask a Math Question")
    question = st.text_area(
        "Type your PSLE Math question below:",
        height=120,
        placeholder="e.g. Ali has 24 apples. He gives 1/3 of them to Ben. How many apples does Ali have left?",
    )

    if st.button("Get Answer", key="qa_submit"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Thinking..."):
            result = answer_question(question.strip())

        st.subheader("Answer")
        st.markdown(result["answer"])

        unique_sources = list(dict.fromkeys(result["sources"]))
        if unique_sources:
            st.subheader("Sources Used")
            for src in unique_sources:
                st.write(f"- {src}")


def render_practice_tab():
    """Render the 'Practice Mode' tab."""
    st.header("Practice Mode")

    if st.button("Give me a question", key="practice_new"):
        q = get_random_question()
        st.session_state["practice_question"] = q["question"]
        st.session_state["practice_answer"] = q["answer"]
        st.session_state["show_answer"] = False
        st.session_state["show_working"] = False

    if "practice_question" not in st.session_state:
        st.info("Click the button above to get a practice question.")
        return

    st.subheader("Question")
    st.write(st.session_state["practice_question"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Show Answer", key="practice_answer_btn"):
            st.session_state["show_answer"] = True

    with col2:
        if st.button("Show Full Working", key="practice_working_btn"):
            st.session_state["show_working"] = True

    if st.session_state.get("show_answer", False):
        final = get_final_answer(st.session_state["practice_answer"])
        st.success(f"**Answer:** {final}")

    if st.session_state.get("show_working", False):
        st.subheader("Full Working")
        st.markdown(st.session_state["practice_answer"])
