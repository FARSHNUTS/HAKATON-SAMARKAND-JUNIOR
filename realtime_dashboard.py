import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Настройка страницы
st.set_page_config(
    page_title="Мониторинг Статистики Безопасности",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Путь к вашему файлу
FILE_PATH = "hackathon_result.csv"

# =================================================================
# 2. Функции для загрузки и предварительных расчетов (Кэширование для скорости)
# =================================================================

@st.cache_data
def load_data(file_path):
    """Загружает и очищает данные."""
    df = pd.read_csv(file_path)
    # Очистка: Фильтруем ID = -1, если это шум/неопознанные объекты
    df_filtered = df[df['id'] >= 0].copy()
    return df_filtered

@st.cache_data
def calculate_people_count(df):
    """Рассчитывает количество уникальных людей в каждом кадре."""
    people_count_per_frame = df.groupby('frame')['id'].nunique().rename('people_count')
    df_with_count = df.merge(people_count_per_frame, on='frame')
    return df_with_count

# Загрузка полных данных и расчет количества людей в кадре
df_data_full = load_data(FILE_PATH)
df_data = calculate_people_count(df_data_full)

# =================================================================
# 3. Боковая панель с фильтрами (st.sidebar)
# =================================================================
st.sidebar.header("Параметры Фильтрации ⚙️")

# 3.1. Фильтр: Количество людей в кадре
min_people = df_data['people_count'].min()
max_people = df_data['people_count'].max()

selected_min_people = st.sidebar.slider(
    "1. Мин. количество людей в кадре",
    min_value=min_people,
    max_value=max_people,
    value=min_people,
    step=1
)

# 3.2. Фильтр по Роли (role)
roles = df_data['role'].unique().tolist()
selected_roles = st.sidebar.multiselect(
    "2. Фильтр по Роли", 
    options=roles, 
    default=roles 
)

# 3.3. Фильтр по ID Объекта (id)
ids = sorted(df_data['id'].unique().tolist())
selected_ids = st.sidebar.multiselect(
    "3. Фильтр по ID Объекта", 
    options=ids, 
    default=ids 
)

# 3.4. Фильтр по Инциденту Опасности (danger)
danger_options = ['Все', 'Только опасные (1)', 'Только безопасные (0)']
selected_danger = st.sidebar.selectbox(
    "4. Фильтр по Опасности", 
    options=danger_options, 
    index=0
)

# =================================================================
# 4. Применение всех фильтров
# =================================================================

df_filtered = df_data[
    (df_data['people_count'] >= selected_min_people) & 
    (df_data['role'].isin(selected_roles)) & 
    (df_data['id'].isin(selected_ids))
]

if selected_danger == 'Только опасные (1)':
    df_filtered = df_filtered[df_filtered['danger'] == 1]
elif selected_danger == 'Только безопасные (0)':
    df_filtered = df_filtered[df_filtered['danger'] == 0]

# Проверка, остались ли данные после фильтрации
if df_filtered.empty:
    st.error("⚠️ Данные не найдены! Измените параметры фильтрации.")
    st.stop()


# =================================================================
# 5. Основная часть дашборда
# =================================================================
st.title("📊 Панель Безопасности и Мониторинга Персонала")
st.markdown(f"Отображаются данные: **{df_filtered.shape[0]:,}** строк.")
st.markdown("---")

# 5.1. Расчет и вывод ключевых показателей (KPI)
total_frames = df_filtered['frame'].max() if not df_filtered.empty else 0
unique_subjects = df_filtered['id'].nunique()
total_incidents = df_filtered[df_filtered['danger'] == 1].shape[0]

if total_frames > 0:
    danger_rate = (df_filtered['danger'].sum() / total_frames) * 100
else:
    danger_rate = 0.0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Всего Обработано Кадров 🖼️", value=f"{total_frames:,}")
    
with col2:
    st.metric(label="Уникальных Объектов в Кадре 👤", value=unique_subjects)
    
with col3:
    st.metric(
        label="Процент Опасных Событий ⚠️", 
        value=f"{danger_rate:.2f}%", 
        delta=f"Всего: {total_incidents} инцидентов", 
        delta_color="inverse"
    )
    
st.markdown("---")

# =================================================================
# 5.2. Объединенный Интерактивный График
# =================================================================

# 5.2.1. Создание фигур

# Фигура A: Тренд Обнаружения Падения
df_timeline = df_filtered.groupby('time_sec')['danger'].max().reset_index()
fig_timeline = px.line(
    df_timeline, 
    x='time_sec', 
    y='danger', 
    title='Тренд Обнаружения Падения / Инцидентов Опасности по Времени',
    template="plotly_white"
)
fig_timeline.update_layout(yaxis_title="Наличие инцидента (1=Да)")
fig_timeline.update_yaxes(
    tick0=0,        
    dtick=1,        
    range=[0, 1.1]  
)


# Фигура B: Количество людей в кадре
df_people_timeline = df_filtered.groupby('time_sec')['people_count'].mean().reset_index()
df_people_timeline.columns = ['time_sec', 'average_people_count']
fig_people = px.line(
    df_people_timeline, 
    x='time_sec', 
    y='average_people_count', 
    title='Среднее число объектов в кадре в секунду',
    template="plotly_white",
    line_shape='hv'
)
fig_people.update_layout(yaxis_title="Количество объектов")


# 5.2.2. Элемент переключения
chart_options = {
    "Обнаружение Падения (Тренд) ⚠️": fig_timeline, 
    "Люди в Кадре по Времени 👤": fig_people,
}

# Используем st.radio для выбора типа графика
selected_chart = st.radio(
    "Выберите График для Отображения",
    list(chart_options.keys()),
    index=0, 
    horizontal=True
)

# 5.2.3. Отображение выбранной фигуры
st.plotly_chart(chart_options[selected_chart], width='stretch')


# 5.3. Таблица с последними событиями (теперь показывается весь отфильтрованный набор данных)
st.markdown("---")
st.subheader("Детализация")

st.write(f"Отображаются все **{df_filtered.shape[0]:,}** отфильтрованных строк. Используйте кнопку 'Expand' для полноэкранного просмотра:")

st.dataframe(
    df_filtered, # <-- Передаем весь фрейм данных
    width='stretch',
    # Убираем ограничение высоты, чтобы при развертывании отображались все данные.
)