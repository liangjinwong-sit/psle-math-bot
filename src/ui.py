"""
Streamlit UI components for the PSLE Math Study Bot.

Provides two main tabs:
1. Ask a Question - RAG-powered Q&A with topic filtering and citations
2. Practice Mode - Random questions + AI-generated practice questions
"""

import re
import streamlit as st
from src.generation import (
    answer_question, auto_mark_answer, explain_mistake,
    generate_hints, generate_mcq_options, generate_similar_question,
)
from src.practice import get_random_question, get_final_answer, generate_practice_question, estimate_difficulty
from src.topic_classifier import get_all_topics


def _escape_dollars(text: str) -> str:
    """Escape dollar signs so Streamlit does not render them as LaTeX."""
    return text.replace("$", r"\$")


def _format_working(raw_answer: str) -> list:
    """
    Clean and split GSM8K solution text into readable step lines.

    Strips <<calc>> annotations and the final #### answer line,
    returning a list of non-empty step strings.
    """
    cleaned = re.sub(r"<<[^>]*>>", "", raw_answer)
    cleaned = cleaned.split("####")[0]
    steps = [line.strip() for line in cleaned.split("\n") if line.strip()]
    return steps


def render_qa_tab():
    """Render the 'Ask a Math Question' tab with topic selection and citations."""
    st.markdown("#### Need help with a math question? Type it below!")

    topics = get_all_topics()

    topic_options = ["All Topics"] + [t["name"] for t in topics]
    topic_keys = [None] + [t["key"] for t in topics]

    selected_idx = st.selectbox(
        "Pick a topic (or leave as All Topics):",
        range(len(topic_options)),
        format_func=lambda i: topic_options[i],
        key="topic_selector",
    )
    topic_key = topic_keys[selected_idx]

    placeholders = {
        "percentage": "e.g. A shirt costs $60 and is sold at a 20% discount. What is the sale price?",
        "fractions_decimals": "e.g. Ali has 24 apples. He gives 1/3 of them to Ben. How many apples does Ali have left?",
        "ratio_proportion": "e.g. The ratio of boys to girls in a class is 3:2. If there are 15 boys, how many girls are there?",
        "rate": "e.g. 5 notebooks cost $15. How much does 1 notebook cost?",
        "measurement": "e.g. Find the area of a rectangle with length 8 cm and width 5 cm.",
        "data_handling": "e.g. The scores of 5 students are 70, 80, 85, 90, 95. What is the average score?",
    }
    placeholder = placeholders.get(
        topic_key,
        "e.g. Janet's ducks lay 16 eggs per day. She eats three for breakfast. How many are left?",
    )

    question = st.text_area(
        "Type your math question here:",
        height=100,
        placeholder=placeholder,
    )

    if st.button("Get Answer", key="qa_submit", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        with st.spinner("Finding examples and solving..."):
            try:
                result = answer_question(question.strip(), topic=topic_key)
            except Exception:
                st.error(
                    "Oops! Something went wrong while solving your question. "
                    "This could be a temporary issue -- please try again in a moment."
                )
                return

        st.markdown("---")

        if result["num_docs_retrieved"] > 0:
            confidence_pct = round(result["confidence"] * 100)
            if result["low_confidence"]:
                st.warning(
                    f"I found some examples but they may not be very similar "
                    f"(match: {confidence_pct}%). I'll still try my best!"
                )
            else:
                st.success(
                    f"Great! I found {result['num_docs_retrieved']} similar problems to help "
                    f"(match: {confidence_pct}%)"
                )

        st.markdown("#### Here's the solution:")
        st.markdown(result["answer"])

        if result.get("used_tools"):
            tools_text = ", ".join(result["used_tools"])
            st.caption(f"Used tools: {tools_text}")

        if result.get("citations"):
            with st.expander("Where did I learn this? (Sources)", expanded=False):
                st.caption(
                    f"I looked at {len(result['citations'])} similar problems from a math dataset:"
                )
                for i, cite in enumerate(result["citations"], 1):
                    similarity_pct = round(cite["score"] * 100)
                    st.markdown(
                        f"**{i}.** {cite['source']} - Topic: {cite['topic']} "
                        f"- Match: {similarity_pct}%"
                    )
                    st.write(_escape_dollars(cite["question"]))
                    if cite["method"]:
                        st.caption(f"Method: {_escape_dollars(cite['method'])}")
                    st.markdown("")


def _update_weak_topics(topic_display: str, is_correct: bool):
    """Update the session weak topic tracker."""
    if "weak_topics" not in st.session_state:
        st.session_state["weak_topics"] = {}

    tracker = st.session_state["weak_topics"]
    if topic_display not in tracker:
        tracker[topic_display] = {"correct": 0, "total": 0}

    tracker[topic_display]["total"] += 1
    if is_correct:
        tracker[topic_display]["correct"] += 1


def _render_weak_topics():
    """Display weak topic suggestions based on session performance."""
    if "weak_topics" not in st.session_state or not st.session_state["weak_topics"]:
        return

    tracker = st.session_state["weak_topics"]

    # ── Score Summary Panel ──
    total_correct = sum(d["correct"] for d in tracker.values())
    total_attempted = sum(d["total"] for d in tracker.values())
    if total_attempted > 0:
        with st.expander(
            f"Your Score: {total_correct}/{total_attempted} correct", expanded=False
        ):
            for topic, data in sorted(tracker.items()):
                pct = round(data["correct"] / data["total"] * 100) if data["total"] else 0
                st.markdown(
                    f"**{topic}:** {data['correct']}/{data['total']} ({pct}%)"
                )

    # ── Weak Topic Suggestions ──
    weak = []
    for topic, data in tracker.items():
        if data["total"] >= 2 and (data["correct"] / data["total"]) < 0.5:
            weak.append((topic, data))

    if weak:
        st.info(
            "**Keep practising these topics:** "
            + ", ".join(
                f"**{t}** ({d['correct']}/{d['total']})"
                for t, d in weak
            )
            + " -- you've got this!"
        )


def render_practice_tab():
    """Render the 'Practice Mode' tab with random questions and question generation."""
    st.markdown("#### Let's practice! Pick a mode and start solving.")

    _render_weak_topics()

    topics = get_all_topics()

    practice_mode = st.radio(
        "What kind of question do you want?",
        ["Practice from Real Exam-Style Questions", "Create a New Question for Me"],
        key="practice_mode_select",
        horizontal=True,
    )

    topic_options = ["Any Topic"] + [t["name"] for t in topics]
    topic_keys = [None] + [t["key"] for t in topics]

    selected_topic_idx = st.selectbox(
        "Pick a topic:",
        range(len(topic_options)),
        format_func=lambda i: topic_options[i],
        key="practice_topic_filter",
    )
    selected_topic_key = topic_keys[selected_topic_idx]

    st.markdown("---")

    if practice_mode == "Practice from Real Exam-Style Questions":
        _render_random_practice(selected_topic_key)
    else:
        _render_generated_practice(selected_topic_key, topics)


def _render_random_practice(topic_key):
    """Render the random question practice sub-section."""

    difficulty = st.select_slider(
        "How hard?",
        options=["easy", "medium", "hard"],
        value="medium",
        key="random_difficulty",
    )

    answer_mode = st.radio(
        "How do you want to answer?",
        ["Type My Answer", "Multiple Choice"],
        key="random_answer_mode",
        horizontal=True,
    )

    if st.button("Give Me a Question!", key="practice_new", type="primary"):
        try:
            q = get_random_question(topic=topic_key, difficulty=difficulty)
            st.session_state["practice_question"] = q["question"]
            st.session_state["practice_answer"] = q["answer"]
            st.session_state["practice_topic"] = q.get("topic_display", "")
            st.session_state["practice_topic_key"] = q.get("topic", "")
            st.session_state["practice_difficulty"] = q.get("difficulty", difficulty)
            st.session_state["show_answer"] = False
            st.session_state["show_working"] = False
            st.session_state["mark_result"] = None
            st.session_state["practice_hints"] = None
            st.session_state["practice_hint_level"] = 0
            st.session_state["practice_explanation"] = None
            st.session_state["practice_mcq_options"] = None
            st.session_state["practice_mcq_selected"] = None
            st.session_state["similar_question"] = None
            st.session_state["similar_mark_result"] = None
            st.session_state["similar_show_answer"] = False
            st.session_state["similar_show_working"] = False
        except Exception as e:
            st.error(f"Error loading question: {str(e)}")
            return

    if "practice_question" not in st.session_state:
        st.info("Press the button above to get started!")
        return

    # Topic & difficulty badge
    if st.session_state.get("practice_topic"):
        diff = st.session_state.get('practice_difficulty', 'medium')
        st.caption(
            f"{st.session_state['practice_topic']}  |  {diff.title()}"
        )

    st.markdown("##### Your Question")
    st.markdown(_escape_dollars(st.session_state["practice_question"]))

    # ── Hint System ──
    with st.expander("Need a hint?", expanded=False):
        if st.button("Get Hint", key="practice_hint_btn"):
            if st.session_state.get("practice_hints") is None:
                with st.spinner("Generating hints..."):
                    try:
                        correct_answer = get_final_answer(st.session_state["practice_answer"])
                        hints = generate_hints(
                            st.session_state["practice_question"], correct_answer
                        )
                        st.session_state["practice_hints"] = hints
                        st.session_state["practice_hint_level"] = 1
                    except Exception:
                        st.error("Could not generate hints right now. Please try again.")
            else:
                current = st.session_state.get("practice_hint_level", 0)
                if current < 3:
                    st.session_state["practice_hint_level"] = current + 1

        hint_level = st.session_state.get("practice_hint_level", 0)
        hints = st.session_state.get("practice_hints")
        if hints and hint_level > 0:
            for i in range(hint_level):
                st.info(f"**Hint {i + 1}:** {hints[i]}")
            if hint_level < 3:
                st.caption("Click 'Get Hint' again for the next hint.")

    # ── Answer Input ──
    if answer_mode == "Multiple Choice":
        _render_mcq_answer()
    else:
        _render_short_answer()

    # ── Show Answer / Working buttons ──
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Show Answer", key="practice_answer_btn"):
            st.session_state["show_answer"] = True
    with col_b:
        if st.button("Show Working", key="practice_working_btn"):
            st.session_state["show_working"] = True

    # ── Display original answer / working BEFORE similar question ──
    if st.session_state.get("show_answer", False):
        final = get_final_answer(st.session_state["practice_answer"])
        st.success(f"**Answer:** {final}")

    if st.session_state.get("show_working", False):
        st.markdown("##### Full Working")
        steps = _format_working(st.session_state["practice_answer"])
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:** {_escape_dollars(step)}")

    # ── Display marking result + Explain My Mistake ──
    if st.session_state.get("mark_result"):
        result = st.session_state["mark_result"]
        if result["is_correct"]:
            st.success(f"**Correct!** {result['feedback']}")
        else:
            st.error(f"**Incorrect.** {result['feedback']}")

            if st.button("What did I do wrong?", key="practice_explain_btn"):
                with st.spinner("Let me look at your work..."):
                    try:
                        correct_answer = get_final_answer(st.session_state["practice_answer"])
                        student_ans = st.session_state.get("practice_student_answer", "")
                        if st.session_state.get("practice_mcq_selected"):
                            student_ans = st.session_state["practice_mcq_selected"]
                        explanation = explain_mistake(
                            st.session_state["practice_question"],
                            correct_answer,
                            student_ans,
                        )
                        st.session_state["practice_explanation"] = explanation
                    except Exception:
                        st.error("Could not generate an explanation right now. Please try again.")

            if st.session_state.get("practice_explanation"):
                st.markdown("---")
                st.markdown("##### Here's what happened:")
                st.markdown(st.session_state["practice_explanation"])

            # ── Similar Question (optional) ──
            st.markdown("---")
            st.markdown("**Want to try a similar question?**")
            if st.button("Try a Similar One", key="practice_similar_btn"):
                with st.spinner("Creating a similar question..."):
                    try:
                        similar = generate_similar_question(
                            st.session_state["practice_question"],
                            st.session_state.get("practice_topic_key", ""),
                            st.session_state.get("practice_difficulty", "medium"),
                        )
                        st.session_state["similar_question"] = similar
                        st.session_state["similar_mark_result"] = None
                        st.session_state["similar_show_answer"] = False
                        st.session_state["similar_show_working"] = False
                    except Exception:
                        st.error("Could not generate a similar question right now. Please try again.")

    # ── Render Similar Question if generated ──
    if st.session_state.get("similar_question"):
        _render_similar_question()


def _render_mcq_answer():
    """Render MCQ options for the current practice question."""
    if st.session_state.get("practice_mcq_options") is None:
        with st.spinner("Generating answer choices..."):
            correct_answer = get_final_answer(st.session_state["practice_answer"])
            options = generate_mcq_options(
                st.session_state["practice_question"], correct_answer
            )
            st.session_state["practice_mcq_options"] = options

    options = st.session_state["practice_mcq_options"]
    if not options:
        return

    selected = st.radio(
        "Pick one:",
        [f"**{opt['label']}.** {opt['text']}" for opt in options],
        key="practice_mcq_radio",
    )

    if st.button("Submit", key="practice_mcq_submit", type="primary"):
        selected_idx = next(
            i for i, opt in enumerate(options)
            if f"**{opt['label']}.** {opt['text']}" == selected
        )
        chosen = options[selected_idx]
        st.session_state["practice_mcq_selected"] = chosen["text"]

        is_correct = chosen["is_correct"]
        st.session_state["mark_result"] = {
            "is_correct": is_correct,
            "score": 1 if is_correct else 0,
            "feedback": "Well done! You chose the right answer!" if is_correct
            else f"Not quite. The correct answer is: {get_final_answer(st.session_state['practice_answer'])}",
        }
        _update_weak_topics(
            st.session_state.get("practice_topic", "General"),
            is_correct,
        )


def _render_short_answer():
    """Render the short answer text input and check button."""
    student_answer = st.text_input(
        "Your answer:",
        key="practice_student_answer",
        placeholder="e.g. 42",
    )

    if st.button("Check!", key="practice_check_btn", type="primary"):
        if not student_answer.strip():
            st.warning("Type your answer first.")
        else:
            with st.spinner("Marking your answer..."):
                correct_answer = get_final_answer(st.session_state["practice_answer"])
                mark_result = auto_mark_answer(
                    st.session_state["practice_question"],
                    correct_answer,
                    student_answer.strip(),
                )
                st.session_state["mark_result"] = mark_result
                _update_weak_topics(
                    st.session_state.get("practice_topic", "General"),
                    mark_result["is_correct"],
                )


def _render_similar_question():
    """Render a similar question block for retry practice."""
    similar = st.session_state["similar_question"]
    st.markdown("---")
    st.markdown("##### Try This One Too!")
    st.markdown(_escape_dollars(similar["question"]))

    sim_answer = st.text_input(
        "Your answer:",
        key="similar_student_answer",
        placeholder="e.g. 42",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check!", key="similar_check_btn", type="primary"):
            if not sim_answer.strip():
                st.warning("Type your answer first.")
            else:
                with st.spinner("Checking..."):
                    mark_result = auto_mark_answer(
                        similar["question"],
                        similar["answer"],
                        sim_answer.strip(),
                    )
                    st.session_state["similar_mark_result"] = mark_result
                    _update_weak_topics(
                        st.session_state.get("practice_topic", "General"),
                        mark_result["is_correct"],
                    )
    with col2:
        if st.button("Show Answer", key="similar_answer_btn"):
            st.session_state["similar_show_answer"] = True

    if st.session_state.get("similar_mark_result"):
        r = st.session_state["similar_mark_result"]
        if r["is_correct"]:
            st.success(f"**Correct!** {r['feedback']}")
        else:
            st.error(f"**Incorrect.** {r['feedback']}")

    if st.session_state.get("similar_show_answer", False):
        st.success(f"**Answer:** {similar['answer']}")
        if similar.get("solution"):
            with st.expander("Solution"):
                steps = _format_working(similar["solution"])
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i}:** {_escape_dollars(step)}")


def _render_generated_practice(topic_key, topics):
    """Render the AI-generated question practice sub-section."""

    if not topic_key:
        st.info("Pick a topic above so the AI can create a question for you!")
        return

    difficulty = st.select_slider(
        "How hard?",
        options=["easy", "medium", "hard"],
        value="medium",
        key="gen_difficulty",
    )

    gen_answer_mode = st.radio(
        "How do you want to answer?",
        ["Type My Answer", "Multiple Choice"],
        key="gen_answer_mode",
        horizontal=True,
    )

    if st.button("Create a Question!", key="gen_new", type="primary"):
        with st.spinner("Generating a new practice question..."):
            try:
                result = generate_practice_question(topic_key, difficulty)
                st.session_state["gen_question"] = result["question"]
                st.session_state["gen_solution"] = result["solution"]
                st.session_state["gen_answer"] = result["answer"]
                st.session_state["gen_topic"] = result["topic_display"]
                st.session_state["gen_topic_key"] = result.get("topic", topic_key)
                st.session_state["gen_result_difficulty"] = result["difficulty"]
                st.session_state["gen_show_answer"] = False
                st.session_state["gen_show_solution"] = False
                st.session_state["gen_mark_result"] = None
                st.session_state["gen_hints"] = None
                st.session_state["gen_hint_level"] = 0
                st.session_state["gen_explanation"] = None
                st.session_state["gen_mcq_options"] = None
                st.session_state["gen_mcq_selected"] = None
                st.session_state["gen_similar"] = None
                st.session_state["gen_similar_mark_result"] = None
                st.session_state["gen_similar_show_answer"] = False
            except Exception as e:
                st.error(f"Error generating question: {str(e)}")
                return

    if "gen_question" not in st.session_state:
        st.info("Press the button above to get started!")
        return

    diff = st.session_state.get('gen_result_difficulty', 'medium')
    st.caption(
        f"{st.session_state.get('gen_topic', '')}  |  {diff.title()}"
    )
    st.markdown("##### Your Question")
    st.markdown(_escape_dollars(st.session_state["gen_question"]))

    # ── Hint System ──
    with st.expander("Need a hint?", expanded=False):
        if st.button("Get Hint", key="gen_hint_btn"):
            if st.session_state.get("gen_hints") is None:
                with st.spinner("Generating hints..."):
                    try:
                        hints = generate_hints(
                            st.session_state["gen_question"],
                            st.session_state["gen_answer"],
                        )
                        st.session_state["gen_hints"] = hints
                        st.session_state["gen_hint_level"] = 1
                    except Exception:
                        st.error("Could not generate hints right now. Please try again.")
            else:
                current = st.session_state.get("gen_hint_level", 0)
                if current < 3:
                    st.session_state["gen_hint_level"] = current + 1

        hint_level = st.session_state.get("gen_hint_level", 0)
        hints = st.session_state.get("gen_hints")
        if hints and hint_level > 0:
            for i in range(hint_level):
                st.info(f"**Hint {i + 1}:** {hints[i]}")
            if hint_level < 3:
                st.caption("Click 'Get Hint' again for the next hint.")

    # ── Answer Input ──
    if gen_answer_mode == "Multiple Choice":
        _render_gen_mcq_answer()
    else:
        _render_gen_short_answer()

    # ── Show Answer / Solution buttons ──
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Show Answer", key="gen_answer_btn"):
            st.session_state["gen_show_answer"] = True
    with col_b:
        if st.button("Show Solution", key="gen_solution_btn"):
            st.session_state["gen_show_solution"] = True

    # ── Display answer / solution BEFORE similar question ──
    if st.session_state.get("gen_show_answer", False):
        st.success(f"**Answer:** {st.session_state['gen_answer']}")

    if st.session_state.get("gen_show_solution", False):
        st.markdown("##### Solution")
        steps = _format_working(st.session_state["gen_solution"])
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:** {_escape_dollars(step)}")

    # ── Display marking result + Explain My Mistake ──
    if st.session_state.get("gen_mark_result"):
        result = st.session_state["gen_mark_result"]
        if result["is_correct"]:
            st.success(f"**Correct!** {result['feedback']}")
        else:
            st.error(f"**Incorrect.** {result['feedback']}")

            if st.button("What did I do wrong?", key="gen_explain_btn"):
                with st.spinner("Let me look at your work..."):
                    try:
                        student_ans = st.session_state.get("gen_student_answer", "")
                        if st.session_state.get("gen_mcq_selected"):
                            student_ans = st.session_state["gen_mcq_selected"]
                        explanation = explain_mistake(
                            st.session_state["gen_question"],
                            st.session_state["gen_answer"],
                            student_ans,
                        )
                        st.session_state["gen_explanation"] = explanation
                    except Exception:
                        st.error("Could not generate an explanation right now. Please try again.")

            if st.session_state.get("gen_explanation"):
                st.markdown("---")
                st.markdown("##### Here's what happened:")
                st.markdown(st.session_state["gen_explanation"])

            # ── Similar Question (optional) ──
            st.markdown("---")
            st.markdown("**Want to try a similar question?**")
            if st.button("Try a Similar One", key="gen_similar_btn"):
                with st.spinner("Creating a similar question..."):
                    try:
                        similar = generate_similar_question(
                            st.session_state["gen_question"],
                            st.session_state.get("gen_topic_key", ""),
                            st.session_state.get("gen_result_difficulty", "medium"),
                        )
                        st.session_state["gen_similar"] = similar
                        st.session_state["gen_similar_mark_result"] = None
                        st.session_state["gen_similar_show_answer"] = False
                    except Exception:
                        st.error("Could not generate a similar question right now. Please try again.")

    # ── Render Similar Question if generated ──
    if st.session_state.get("gen_similar"):
        _render_gen_similar_question()


def _render_gen_mcq_answer():
    """Render MCQ options for the generated practice question."""
    if st.session_state.get("gen_mcq_options") is None:
        with st.spinner("Generating answer choices..."):
            try:
                options = generate_mcq_options(
                    st.session_state["gen_question"],
                    st.session_state["gen_answer"],
                )
                st.session_state["gen_mcq_options"] = options
            except Exception:
                st.error("Could not generate choices right now. Please try again.")
                return

    options = st.session_state["gen_mcq_options"]
    if not options:
        return

    selected = st.radio(
        "Pick one:",
        [f"**{opt['label']}.** {opt['text']}" for opt in options],
        key="gen_mcq_radio",
    )

    if st.button("Submit", key="gen_mcq_submit", type="primary"):
        selected_idx = next(
            i for i, opt in enumerate(options)
            if f"**{opt['label']}.** {opt['text']}" == selected
        )
        chosen = options[selected_idx]
        st.session_state["gen_mcq_selected"] = chosen["text"]

        is_correct = chosen["is_correct"]
        st.session_state["gen_mark_result"] = {
            "is_correct": is_correct,
            "score": 1 if is_correct else 0,
            "feedback": "Well done! You chose the right answer!" if is_correct
            else f"Not quite. The correct answer is: {st.session_state['gen_answer']}",
        }
        _update_weak_topics(
            st.session_state.get("gen_topic", "General"),
            is_correct,
        )


def _render_gen_short_answer():
    """Render short answer input for generated practice."""
    gen_student_answer = st.text_input(
        "Your answer:",
        key="gen_student_answer",
        placeholder="e.g. 42",
    )

    if st.button("Check!", key="gen_check_btn", type="primary"):
        if not gen_student_answer.strip():
            st.warning("Type your answer first.")
        else:
            with st.spinner("Marking your answer..."):
                mark_result = auto_mark_answer(
                    st.session_state["gen_question"],
                    st.session_state["gen_answer"],
                    gen_student_answer.strip(),
                )
                st.session_state["gen_mark_result"] = mark_result
                _update_weak_topics(
                    st.session_state.get("gen_topic", "General"),
                    mark_result["is_correct"],
                )


def _render_gen_similar_question():
    """Render a similar question block for the generated practice."""
    similar = st.session_state["gen_similar"]
    st.markdown("---")
    st.markdown("##### Try This One Too!")
    st.markdown(_escape_dollars(similar["question"]))

    sim_answer = st.text_input(
        "Your answer:",
        key="gen_similar_student_answer",
        placeholder="e.g. 42",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check!", key="gen_similar_check_btn", type="primary"):
            if not sim_answer.strip():
                st.warning("Type your answer first.")
            else:
                with st.spinner("Checking..."):
                    mark_result = auto_mark_answer(
                        similar["question"],
                        similar["answer"],
                        sim_answer.strip(),
                    )
                    st.session_state["gen_similar_mark_result"] = mark_result
                    _update_weak_topics(
                        st.session_state.get("gen_topic", "General"),
                        mark_result["is_correct"],
                    )
    with col2:
        if st.button("Show Answer", key="gen_similar_answer_btn"):
            st.session_state["gen_similar_show_answer"] = True

    if st.session_state.get("gen_similar_mark_result"):
        r = st.session_state["gen_similar_mark_result"]
        if r["is_correct"]:
            st.success(f"**Correct!** {r['feedback']}")
        else:
            st.error(f"**Incorrect.** {r['feedback']}")

    if st.session_state.get("gen_similar_show_answer", False):
        st.success(f"**Answer:** {similar['answer']}")
        if similar.get("solution"):
            with st.expander("Solution"):
                steps = _format_working(similar["solution"])
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i}:** {_escape_dollars(step)}")
