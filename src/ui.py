import streamlit as st
from src.generation import answer_question
from src.practice import get_random_question, get_final_answer
from src.topic_classifier import get_all_topics


def render_qa_tab():
    """Render the 'Ask a Math Question' tab with topic selection."""
    st.header("Ask a Math Question")
    
    # Topic selection
    st.subheader("📚 Select PSLE Topic")
    
    topics = get_all_topics()
    
    # Create topic selection with radio buttons
    topic_options = ["All Topics"] + [f"{t['name']}" for t in topics]
    topic_descriptions = ["Search across all topics"] + [f"{t['description']}" for t in topics]
    
    selected_index = st.radio(
        "Choose a topic to get relevant examples:",
        range(len(topic_options)),
        format_func=lambda i: f"**{topic_options[i]}** - {topic_descriptions[i]}",
        key="topic_selector"
    )
    
    # Get selected topic key (None for "All Topics")
    if selected_index == 0:
        selected_topic = None
        topic_key = None
    else:
        selected_topic = topics[selected_index - 1]
        topic_key = selected_topic['key']
    
    # Display topic info
    if selected_topic:
        st.info(f"🎯 **Topic: {selected_topic['name']}**\n\n"
                f"*{selected_topic['description']}*\n\n"
                f"The bot will find similar problems from this topic to help answer your question.")
    else:
        st.info("🌐 **All Topics Mode**: The bot will search across all PSLE topics for relevant examples.")
    
    # Question input
    st.subheader("✍️ Your Question")
    
    # Provide topic-specific example questions
    if topic_key == "percentage":
        placeholder = "e.g. A shirt costs $60 and is sold at a 20% discount. What is the sale price?"
    elif topic_key == "fractions_decimals":
        placeholder = "e.g. Ali has 24 apples. He gives 1/3 of them to Ben. How many apples does Ali have left?"
    elif topic_key == "ratio_proportion":
        placeholder = "e.g. The ratio of boys to girls in a class is 3:2. If there are 15 boys, how many girls are there?"
    elif topic_key == "rate":
        placeholder = "e.g. 5 notebooks cost $15. How much does 1 notebook cost?"
    elif topic_key == "measurement":
        placeholder = "e.g. Find the area of a rectangle with length 8m and width 5m."
    elif topic_key == "data_handling":
        placeholder = "e.g. The scores of 5 students are 70, 80, 85, 90, 95. What is the average score?"
    else:
        placeholder = "e.g. Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. How many eggs does she sell daily?"
    
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

        # Display answer
        st.markdown("---")
        st.subheader("💡 Answer")
        
        if result["num_docs_retrieved"] > 0:
            st.success(f"✅ Found {result['num_docs_retrieved']} relevant examples from **{result['topic_display']}**")
        
        st.markdown(result["answer"])

        # Display source examples used
        if result["sources"]:
            with st.expander("📖 Example Problems Used", expanded=False):
                st.write(f"The tutor used these {len(result['sources'])} similar problems as reference:")
                for i, src in enumerate(result["sources"], 1):
                    st.write(f"**{i}.** [{src['topic']}] {src['question']}")


def render_practice_tab():
    """Render the 'Practice Mode' tab."""
    st.header("Practice Mode")
    
    st.info("💪 **Practice Mode** gives you random questions from GSM8K dataset to practice solving on your own!")

    if st.button("Give me a question", key="practice_new", type="primary"):
        q = get_random_question()
        st.session_state["practice_question"] = q["question"]
        st.session_state["practice_answer"] = q["answer"]
        st.session_state["show_answer"] = False
        st.session_state["show_working"] = False

    if "practice_question" not in st.session_state:
        st.info("👆 Click the button above to get a practice question.")
        return

    st.markdown("---")
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
