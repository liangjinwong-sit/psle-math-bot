import streamlit as st
from src.generation import answer_question
from src.practice import get_random_question, get_final_answer


def render_qa_tab():
    st.header("Ask a Math Question")

    question = st.text_area(
        "Type your PSLE Math question below:",
        height=120,
        placeholder="e.g. 5 notebooks cost $15. How much does 1 notebook cost?",
    )

    if st.button("Get Answer", key="qa_submit"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Thinking..."):
            result = answer_question(question.strip())

        st.subheader("Answer")

        if result.get("supported", False):
            route = result.get("route", {})
            answer_text = result.get("answer", "")

            final_answer = _extract_section(answer_text, "Final answer:")
            working = _extract_section(answer_text, "Working:")
            why = _extract_section(answer_text, "Why this works:")

            st.success("A supported method was found.")

            topic = route.get("topic")
            method = route.get("method")
            reason = route.get("reason")

            if topic or method:
                st.markdown("### Method chosen")
                if topic:
                    st.write(f"Topic: {topic}")
                if method:
                    st.write(f"Method: {method}")
                if reason:
                    st.write(f"Reason: {reason}")

            if final_answer:
                st.markdown("### Final answer")
                st.success(final_answer)

            if working:
                st.markdown("### Working")
                st.markdown(working)

            if why:
                st.markdown("### Why this works")
                st.info(why)

            if not final_answer and not working and answer_text:
                st.markdown(answer_text)

        else:
            st.warning(result.get("answer", "No supported answer found."))

            route = result.get("route", {})
            if route:
                with st.expander("Routing details", expanded=False):
                    st.write(f"Topic: {route.get('topic')}")
                    st.write(f"Method: {route.get('method')}")
                    st.write(f"Confidence: {route.get('confidence')}")
                    st.write(f"Reason: {route.get('reason')}")

        unique_sources = list(dict.fromkeys(result.get("sources", [])))
        if unique_sources:
            with st.expander("Sources", expanded=False):
                for src in unique_sources:
                    st.write(f"- {src}")


def _extract_section(text: str, header: str):
    if not text or header not in text:
        return None

    start = text.find(header) + len(header)
    remaining = text[start:].strip()

    next_headers = [
        "Final answer:",
        "Working:",
        "Why this works:",
    ]
    next_positions = []

    for h in next_headers:
        if h == header:
            continue
        pos = remaining.find(h)
        if pos != -1:
            next_positions.append(pos)

    if next_positions:
        end = min(next_positions)
        return remaining[:end].strip()

    return remaining.strip()


def render_practice_tab():
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
        st.success(f"Answer: {final}")

    if st.session_state.get("show_working", False):
        st.subheader("Full Working")
        st.markdown(st.session_state["practice_answer"])
