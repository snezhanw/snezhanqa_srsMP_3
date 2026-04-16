import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, SerperDevTool

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"]
)

file_tool = FileReadTool()
search_tool = SerperDevTool()

st.set_page_config(page_title="KazNU Multi-Agent System", layout="wide")

def load_css():
    st.markdown("""
    <style>
    .stApp {background-color: #fff0f6;}
    h1, h2, h3 {color: #d63384;}
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        border-radius: 12px;
        padding: 8px 16px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

st.title("🌸 KazNU Multi-Agent AI System")

st.image("banner.jpg", use_container_width=True)

st.sidebar.header("⚙️ Agent Settings")

user_country = st.text_input("Country")
req_type = st.selectbox("Scenario", ["General", "Study", "Housing", "Leisure"])

knowledge = st.text_area("Knowledge Base", value="Campus rules and regulations")

uploaded_file = st.file_uploader("Upload infrastructure file", type=["txt"])

infra_path = None

if uploaded_file:
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    infra_path = "temp.txt"
else:
    infra_path = "infrastructure.txt"

st.markdown("---")

if st.button("🚀 Run System"):

    if not infra_path:
        st.error("No infrastructure file")
        st.stop()

    analyst = Agent(
        role="Analyst",
        goal="Analyze student request",
        backstory="Expert in student adaptation",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    guide = Agent(
        role="Guide",
        goal="Create campus navigation plan",
        backstory="Uses campus infrastructure data",
        tools=[file_tool],
        llm=llm,
        memory=True,
        verbose=True
    )

    controller = Agent(
        role="Controller",
        goal="Validate output safety",
        backstory="Ensures compliance with rules",
        llm=llm,
        verbose=True
    )

    tasks = []

    t1 = Task(
        description=f"Analyze student from {user_country}. Knowledge: {knowledge}",
        expected_output="Student profile summary",
        agent=analyst
    )
    tasks.append(t1)

    if req_type == "General":
        tasks.append(Task(
            description="Ask clarification questions",
            expected_output="List of questions",
            agent=analyst
        ))

    t2 = Task(
        description=f"Use file {infra_path} to create plan for {req_type}",
        expected_output="Campus route plan",
        agent=guide
    )
    tasks.append(t2)

    t3 = Task(
        description="Finalize result in structured format",
        expected_output="Final report",
        agent=controller,
        human_input=True
    )
    tasks.append(t3)

    crew = Crew(
        agents=[analyst, guide, controller],
        tasks=tasks,
        process=Process.sequential,
        memory=True
    )

    result = crew.kickoff()

    st.subheader("📌 Result")
    st.markdown(result.raw)