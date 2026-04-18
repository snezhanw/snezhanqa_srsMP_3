import streamlit as st
import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, SerperDevTool

# 1. ENV
load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

if not os.environ["GOOGLE_API_KEY"]:
    st.error("❌ Нет GOOGLE_API_KEY")
    st.stop()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"]
)

file_tool = FileReadTool()
search_tool = SerperDevTool()

# 2. UI
st.set_page_config(page_title="KazNU Multi-Agent", layout="wide")

def load_css():
    if os.path.exists("style.css"):
        with open("style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. MEMORY (Streamlit + CrewAI)
if "history" not in st.session_state:
    st.session_state.history = []

# 4. SIDEBAR
st.sidebar.title("Навигация")
st.sidebar.info("Multi-Agent System")

st.title("🌸 KazNU Multi-Agent System")

if os.path.exists("banner.jpg"):
    st.image("banner.jpg", use_container_width=True)

# 5. AGENT CONFIG
st.header("⚙️ Конфигурация")

with st.expander("Редактирование"):
    col1, col2 = st.columns(2)

    with col1:
        r_analyst = st.text_input("Роль аналитика", "Культурный аналитик")
        g_analyst = st.text_input("Цель аналитика", "Анализ профиля студента и его трудностей")

    with col2:
        r_guide = st.text_input("Роль гида", "Навигатор кампуса")
        g_guide = st.text_input("Цель гида", "Создание персонального маршрута адаптации")

# 6. INPUT
st.header("📝 Вход")

user_question = st.text_area("Вопрос", "Сделай маршрут по кампусу")

col1, col2 = st.columns(2)

with col1:
    user_country = st.text_input("Страна", "Южная Корея")
    req_type = st.selectbox("Тип", ["Общий", "Учеба", "Жилье", "Медицина", "Документы", "Досуг"])

    k_base = st.text_area("Knowledge (база знаний)", "СББП помогает студентам адаптироваться. Есть общежития, медпункт, библиотека.")

with col2:
    uploaded_file = st.file_uploader("Файл инфраструктуры", type=["txt"])

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

# читаем инфраструктуру
if infra_path:
    with open(infra_path, encoding="utf-8") as f:
        infra_text = f.read()

# KNOWLEDGE FILE
rules_text = ""
if os.path.exists("knowledge/rules.txt"):
    with open("knowledge/rules.txt", encoding="utf-8") as f:
        rules_text = f.read()

# 7. RUN
st.header("🚀 Запуск")

if st.button("Сгенерировать"):

    if not infra_path:
        st.error("❌ Нет файла инфраструктуры")
        st.stop()

    # сохраняем в memory
    st.session_state.history.append({
        "question": user_question,
        "country": user_country,
        "type": req_type
    })

    # === AGENTS ===
    analyst = Agent(
        role=r_analyst,
        goal=g_analyst,
        backstory="Эксперт по культурной адаптации иностранных студентов",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    guide = Agent(
        role=r_guide,
        goal=g_guide,
        backstory="Специалист по инфраструктуре кампуса",
        tools=[file_tool],
        llm=llm,
        memory=True,
        verbose=True
    )

    critic = Agent(
        role="Контролер качества",
        goal="Проверка корректности и культурной аккуратности рекомендаций",
        backstory="Проверяет соответствие рекомендаций правилам и культуре",
        llm=llm,
        verbose=True
    )

    tasks = []

    # === TASK 1: ANALYSIS (Memory + Knowledge + Tools) ===
    tasks.append(Task(
        description=f"""
Проанализируй студента:

Вопрос: {user_question}
Страна: {user_country}
Тип: {req_type}

История прошлых запросов:
{st.session_state.history}

Используй Knowledge и Rules:
Knowledge: {k_base}
Rules: {rules_text}

При необходимости используй search tool.

Определи:
- профиль студента
- возможные трудности
- культурные особенности
        """,
        expected_output="Анализ профиля студента и список потенциальных проблем",
        agent=analyst
    ))

    # === CONDITIONAL TASK ===
    if req_type == "Общий" or len(user_question) < 30:
        tasks.append(Task(
            description="""
Информации недостаточно.

Сгенерируй короткий уточняющий вопрос.
Варианты: проживание, учеба, медицина, документы, досуг.
            """,
            expected_output="1 уточняющий вопрос",
            agent=analyst
        ))

    # === TASK 2: ROUTE (Files + Tools) ===
    tasks.append(Task(
        description=f"""
Используй file tool для анализа файла инфраструктуры: {infra_path}

Инфраструктура:
{infra_text}

Создай персональный маршрут для типа: {req_type}

Учитывай анализ предыдущего агента.
        """,
        expected_output="Подробный маршрут адаптации",
        agent=guide
    ))

    # === TASK 3: FINAL (Knowledge) ===
    tasks.append(Task(
        description=f"""
Сформируй финальный персональный гид.

Используй:
- предыдущие результаты
- Knowledge
- Rules

Сделай ответ:
- понятным
- культурно аккуратным
- структурированным
        """,
        expected_output="Готовый персональный гид",
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

    # === HITL ===
    st.subheader("🧑‍⚖️ Проверка результата (HITL)")

    decision = st.radio("Подтверждение:", ["Одобрить", "Отклонить"])

    if decision == "Одобрить":
        st.markdown(f"<div class='card'>{result.raw}</div>", unsafe_allow_html=True)
        st.success("✅ Одобрено")
        st.balloons()

    else:
        st.warning("❌ Результат отклонён пользователем")

    # === LOGS ===
    st.subheader("📊 Лог выполнения")
    st.write(result)

# 8. HISTORY
st.header("📚 История")

for i in st.session_state.history:
    st.markdown(
        f"<div class='card'>❓ {i['question']}<br>🌍 {i['country']}<br>📌 {i['type']}</div>",
        unsafe_allow_html=True
    )