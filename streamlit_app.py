import streamlit as st
import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. ENV
load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

if not os.environ["GOOGLE_API_KEY"]:
    st.error("❌ Нет GOOGLE_API_KEY")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

file_tool = FileReadTool()
search_tool = SerperDevTool()

# 2. UI
st.set_page_config(page_title="KazNU Multi-Agent", layout="wide")

def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. MEMORY
if "history" not in st.session_state:
    st.session_state.history = []

# 4. SIDEBAR
st.sidebar.title("Навигация")
st.sidebar.info("Multi-Agent System")

st.title("🌸 KazNU Multi-Agent System")
st.image("banner.jpg", use_container_width=True)

# 5. AGENT CONFIG
st.header("⚙️ Конфигурация")

with st.expander("Редактирование"):
    col1, col2 = st.columns(2)

    with col1:
        r_analyst = st.text_input("Роль аналитика", "Культурный аналитик")
        g_analyst = st.text_input("Цель аналитика", "Анализ студента")

    with col2:
        r_guide = st.text_input("Роль гида", "Навигатор кампуса")
        g_guide = st.text_input("Цель гида", "Маршрут")

# 6. INPUT
st.header("📝 Вход")

user_question = st.text_area("Вопрос", "Сделай маршрут по кампусу")

col1, col2 = st.columns(2)

with col1:
    user_country = st.text_input("Страна", "Южная Корея")
    req_type = st.selectbox("Тип", ["Общий", "Учеба", "Жилье", "Досуг"])

    k_base = st.text_area("Knowledge", "СББП помогает студентам")

with col2:
    uploaded_file = st.file_uploader("Файл infrastructure.txt", type=["txt"])

    infra_path = None
    infra_text = ""

    if uploaded_file:
        with open("temp_infra.txt", "wb") as f:
            f.write(uploaded_file.getbuffer())
        infra_path = "temp_infra.txt"
        st.success("Файл загружен")

    elif os.path.exists("data/infrastructure.txt"):
        infra_path = "data/infrastructure.txt"
        st.info("Используется файл по умолчанию")

if infra_path:
    with open(infra_path, encoding="utf-8") as f:
        infra_text = f.read()

rules_text = ""
if os.path.exists("rules.txt"):
    with open("rules.txt", encoding="utf-8") as f:
        rules_text = f.read()

# 7. RUN
st.header("🚀 Запуск")

if st.button("Сгенерировать"):

    if not infra_path:
        st.error("Нет файла инфраструктуры")
        st.stop()

    st.session_state.history.append({
        "question": user_question,
        "country": user_country,
        "type": req_type
    })

    analyst = Agent(
        role=r_analyst,
        goal=g_analyst,
        backstory="Анализ студентов",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    guide = Agent(
        role=r_guide,
        goal=g_guide,
        backstory="Маршруты кампуса",
        tools=[file_tool],
        llm=llm,
        memory=True,
        verbose=True
    )

    critic = Agent(
        role="Контролер",
        goal="Безопасность",
        backstory="Проверка правил",
        llm=llm,
        verbose=True
    )

    tasks = []

    tasks.append(Task(
        description=f"""
Вопрос: {user_question}
Страна: {user_country}
Тип: {req_type}
Knowledge: {k_base}
Rules: {rules_text}
        """,
        expected_output="Анализ",
        agent=analyst
    ))

    if req_type == "Общий":
        tasks.append(Task(
            description="Задай уточняющие вопросы",
            expected_output="Вопросы",
            agent=analyst
        ))

    tasks.append(Task(
        description=f"""
Инфраструктура:
{infra_text}

Сделай маршрут для {req_type}
        """,
        expected_output="Маршрут",
        agent=guide
    ))

    tasks.append(Task(
        description="Финальный отчет",
        expected_output="Готовый гид",
        agent=critic
    ))

    crew = Crew(
        agents=[analyst, guide, critic],
        tasks=tasks,
        process=Process.sequential,
        memory=True
    )

    with st.spinner("Генерация..."):
        result = crew.kickoff()

    st.subheader("Результат")

    if st.checkbox("Подтвердить"):
        st.markdown(f"<div class='card'>{result.raw}</div>", unsafe_allow_html=True)
        st.balloons()

# 8. HISTORY
st.header("История")

for i in st.session_state.history:
    st.markdown(f"<div class='card'>❓ {i['question']}<br>🌍 {i['country']}<br>📌 {i['type']}</div>", unsafe_allow_html=True)