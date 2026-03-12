"""
Streamlit UI components for the PSLE Math Study Bot.

Provides two main tabs:
1. Ask a Question - RAG-powered Q&A with topic filtering and citations
2. Practice Mode - Random questions + AI-generated practice questions
"""

import streamlit as st
from src.generation import answer_question
from src.practice import get_random_question, get_final_answer, generate_practice_question
from src.topic_classifier import get_all_topics


def render_qa_tab():
    """Render the 'Ask a Math Question' tab with topic selection and citations."""
    st.header("Ask a Math Question")
    
    # Topic selection
    st.subheader("📚 Select PSLE Topic")
    
    topics = get_all_topics()
    
    # Create topic selection with radio buttons
    topic_options = ["All Topics"] + [t["name"] for t in topics]
    topic_descriptions = ["Search across all topics"] + [t["description"] for t in topics]
    
    selected_index = st.radio(
        "Choose a topic to get relevant examples:",
        range(len(topic_options)),
        format_func=lambda i: f"**{topic_options[i]}** — {topic_descriptions[i]}",
        key="topic_selector",
    )
    
    # Get selected topic key (None for "All Topics")
    if selected_index == 0:
        selected_topic = None
        topic_key = None
    else:
        selected_topic = topics[selected_index - 1]
        topic_key = selected_topic["key"]
    
    # Display topic info
    if selected_topic:
        st.info(
            f"🎯 **Topic: {selected_topic['name']}**\n\n"
            f"*{selected_topic['description']}*\n\n"
            f"The bot will find similar problems from this topic to help answer your question."
        )
    else:
        st.info("🌐 **All Topics Mode**: The bot will search across all PSLE topics for relevant examples.")
    
    # Question input
    st.subheader("✍️ Your Question")
    
    # Topic-specific placeholder examples
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
        "e.g. Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. How many eggs does she sell daily?",
    )
    
    question = st.text_area(
        "Type your PSLE Math question below:",
        height=120,
        placeholder=placeholder,
    )

    if st.button("Get Answer", key="qa_submit", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Finding similar problems and generating answer..."):
            result = answer_question(question.strip(), topic=topic_key)

        st.markdown("---")
        
        # Confidence and retrieval info
        if result["num_docs_retrieved"] > 0:
            confidence_pct = round(result["confidence"] * 100)
            if result["low_confidence"]:
                st.warning(
                    f"⚠️ Found {result['num_docs_retrieved']} examples from **{result['topic_display']}** "
                    f"(confidence: {confidence_pct}% — may not be closely related)"
                )
            else:
                st.success(
                    f"✅ Found {result['num_docs_retrieved']} relevant examples from "
                    f"**{result['topic_display']}** (confidence: {confidence_pct}%)"
                )
        
        # Display answer
        st.subheader("💡 Answer")
        st.markdown(result["answer"])

        # Display structured citations
        if result.get("citations"):
            with st.expander("📖 Sources & Citations", expanded=False):
                st.write(
                    f"The tutor referenced these **{len(result['citations'])}** similar problems "
                    f"from the GSM8K dataset:"
                )
                for i, cite in enumerate(result["citations"], 1):
                    similarity_pct = round(cite["score"] * 100)
                    st.markdown(
                        f"**{i}. [{cite['topic']}]** {cite['source']} "
                        f"(similarity: {similarity_pct}%)"
                    )
                    st.markdown(f"   *Question:* {cite['question']}")
                    if cite["method"]:
                        st.markdown(f"   *Method:* {cite['method']}")
                    st.markdown("---")


def render_practice_tab():
    """Render the 'Practice Mode' tab with random questions and question generation."""
    st.header("Practice Mode")
    
    topics = get_all_topics()
    
    # Sub-tabs for different practice modes
    practice_mode = st.radio(
        "Choose practice mode:",
        ["Random Question (from GSM8K)", "Generate New Question (AI-created)"],
        key="practice_mode_select",
        horizontal=True,
    )
    
    # Topic filter for practice
    topic_options = ["Any Topic"] + [t["name"] for t in topics]
    topic_keys = [None] + [t["key"] for t in topics]
    
    selected_topic_idx = st.selectbox(
        "Filter by topic:",
        range(len(topic_options)),
        format_func=lambda i: topic_options[i],
        key="practice_topic_filter",
    )
    selected_topic_key = topic_keys[selected_topic_idx]
    
    st.markdown("---")
    
    if practice_mode == "Random Question (from GSM8K)":
        _render_random_practice(selected_topic_key)
    else:
        _render_generated_practice(selected_topic_key, topics)


def _render_random_practice(topic_key):
    """Render the random question practice sub-section."""
    st.info("💪 Get a random question from the GSM8K test set to practice solving on your own!")

    if st.button("Give me a question", key="practice_new", type="primary"):
        try:
            q = get_random_question(topic=topic_key)
            st.session_state["practice_question"] = q["question"]
            st.session_state["practice_answer"] = q["answer"]
            st.session_state["practice_topic"] = q.get("topic_display", "")
            st.session_state["show_answer"] = False
            st.session_state["show_working"] = False
        except Exception as e:
            st.error(f"Error loading question: {str(e)}")
            return

    if "practice_question" not in st.session_state:
        st.info("👆 Click the button above to get a practice question.")
        return

    # Display question
    if st.session_state.get("practice_topic"):
        st.caption(f"Topic: {st.session_state['practice_topic']}")
    
    st.subheader("❓ Question")
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
        st.subheader("📝 Full Working")
        st.markdown(st.session_state["practice_answer"])


def _render_generated_practice(topic_key, topics):
    """Render the AI-generated question practice sub-section."""
    st.info(
        "🤖 **AI Question Generator** creates fresh practice questions using the LLM. "
        "Great for unlimited practice on any topic!"
    )
    
    # Need a topic selected for generation
    if not topic_key:
        st.warning("Please select a specific topic above to generate practice questions.")
        return
    
    # Difficulty selector
    difficulty = st.select_slider(
        "Difficulty:",
        options=["easy", "medium", "hard"],
        value="medium",
        key="gen_difficulty",
    )
    
    if st.button("Generate Question", key="gen_new", type="primary"):
        with st.spinner("Generating a new practice question..."):
            try:
                result = generate_practice_question(topic_key, difficulty)
                st.session_state["gen_question"] = result["question"]
                st.session_state["gen_solution"] = result["solution"]
                st.session_state["gen_answer"] = result["answer"]
                st.session_state["gen_topic"] = result["topic_display"]
                st.session_state["gen_difficulty"] = result["difficulty"]
                st.session_state["gen_show_answer"] = False
                st.session_state["gen_show_solution"] = False
            except Exception as e:
                st.error(f"Error generating question: {str(e)}")
                return
    
    if "gen_question" not in st.session_state:
        st.info("👆 Select a topic and click Generate to create a practice question.")
        return
    
    # Display generated question
    st.caption(
        f"Topic: {st.session_state.get('gen_topic', '')} · "
        f"Difficulty: {st.session_state.get('gen_difficulty', 'medium')}"
    )
    st.subheader("❓ Question")
    st.write(st.session_state["gen_question"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show Answer", key="gen_answer_btn"):
            st.session_state["gen_show_answer"] = True
    
    with col2:
        if st.button("Show Full Solution", key="gen_solution_btn"):
            st.session_state["gen_show_solution"] = True
    
    if st.session_state.get("gen_show_answer", False):
        st.success(f"**Answer:** {st.session_state['gen_answer']}")
    
    if st.session_state.get("gen_show_solution", False):
        st.subheader("📝 Solution")
        st.markdown(st.session_state["gen_solution"])
