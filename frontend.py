import streamlit as st
from orchestrator import run_agent
from tinydb import TinyDB

db = TinyDB("memory/memory_store.json")
topics = [t["name"] for t in db.all()]

st.title("🧠 AthenaCore: Multi-Agent Collaboration with Memory")

topic = st.selectbox(
    "Choose or start a topic",
    options=topics + ["New Topic"]
)

if topic == "New Topic":
    topic = st.text_input("Enter new topic name")

agent_choice = st.selectbox(
    "Run agent",
    ["Research", "Summarizer", "Devil", "Insight"]
)

query = st.text_area(
    "Enter query or context (only for Research Agent):",
    ""
)

if st.button("Run Agent") and topic:
    result = run_agent(agent_choice, topic, query)

    st.subheader(f"{agent_choice} Agent Output")
    st.write(result)

    st.subheader("📜 Shared Topic Log")
    log = db.search(lambda x: x["name"] == topic)
    if log:
        for entry in log[0]["log"][::-1]:
            st.markdown(f"**{entry['agent']}**: {entry['content']}")
