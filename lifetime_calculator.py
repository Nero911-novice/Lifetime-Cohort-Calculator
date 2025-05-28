import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_title="📈 Lifetime & LTV Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка стилей
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifetimeCalculator:
    """Профессиональный класс для расчета Lifetime, LTV и связанных метрик."""
    
    def __init__(self):
        self.retention_keywords = ['retention', '%', 'ret', 'удержан', 'остал', 'процент']
    
    @st.cache_data
    def read_file(_self, file_data) -> Tuple[pd.DataFrame, str]:
        """Читает CSV или Excel файл с кэшированием."""
        try:
            # Получаем данные из uploaded file
            if file_data.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_data)
                return df, "success"
            
            # Для CSV пробуем разные параметры
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        file_data.seek(0)  # Возвращаем указатель в начало
                        df = pd.read_csv(file_data, encoding=encoding, sep=sep)
                        if len(df.columns) >= 2 and len(df) > 0:
                            return df, "success"
                    except Exception:
                        continue
            
            return pd.DataFrame(), "Не удалось прочитать файл. Проверьте формат и кодировку."
        
        except Exception as e:
            return pd.DataFrame(), f"Ошибка чтения файла: {str(e)}"
    
    def find_retention_column(self, df: pd.DataFrame, col_name: Optional[str] = None) -> Tuple[str, str]:
        """Находит колонку с retention данными."""
        if col_name and col_name in df.columns:
            return col_name, "success"
        
        # Автопоиск по ключевым словам
        candidates = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in self.retention_keywords):
                candidates.append(col)
        
        if candidates:
            return candidates[0], "success"
        
        # Берем первую числовую колонку
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return numeric_cols[0], f"warning: Использована колонка '{numeric_cols[0]}'"
        
        return "", "Не найдено подходящих числовых колонок"
    
    def validate_retention_data(self, retention_series: pd.Series) -> Tuple[pd.Series, str]:
        """Валидирует и очищает данные retention."""
        try:
            retention = retention_series.copy().fillna(0)
            
            # Преобразуем в числовой тип
            if retention.dtype not in [np.float64, np.int64]:
                retention = pd.to_numeric(retention, errors='coerce').fillna(0)
            
            # Если значения больше 1, считаем проценты
            if retention.max() > 1:
                retention = retention / 100.0
            
            # Валидация
            if retention.max() > 1.5:
                return pd.Series(), "Некорректные значения retention (> 150%)"
            
            if len(retention[retention > 0]) == 0:
                return pd.Series(), "Все значения retention равны нулю"
            
            return retention, "success"
        
        except Exception as e:
            return pd.Series(), f"Ошибка валидации: {str(e)}"
    
    def calculate_metrics(self, retention: pd.Series, arppu: float, cac: float) -> Dict[str, Any]:
        """Рассчитывает все основные метрики."""
        try:
            # Основные метрики
            lifetime = retention.sum()
            ltv = arppu * lifetime
            ltv_cac = ltv / cac if cac > 0 else np.inf
            roi_percent = (ltv_cac - 1) * 100
            
            # Кумулятивные значения
            cumulative_retention = retention.cumsum()
            cumulative_ltv = arppu * cumulative_retention
            
            # Период окупаемости
            payback_period = None
            for i, cum_ltv in enumerate(cumulative_ltv):
                if cum_ltv >= cac:
                    payback_period = i + 1
                    break
            
            # Дополнительные метрики для аналитика
            monthly_ltv = arppu * retention
            churn_rates = 1 - (retention / retention.shift(1).fillna(1))
            churn_rates = churn_rates.fillna(0)
            
            # Прогноз на будущие периоды (простая экстраполяция)
            if len(retention) >= 3:
                # Используем убывающую экспоненту для прогноза
                last_periods = retention.tail(3)
                decay_rate = (last_periods.iloc[-1] / last_periods.iloc[0]) ** (1/2)
                
                future_retention = []
                last_value = retention.iloc[-1]
                for i in range(12):  # прогноз на год вперед
                    last_value *= decay_rate
                    future_retention.append(max(last_value, 0.001))  # минимум 0.1%
                
                future_retention = pd.Series(future_retention)
                extended_lifetime = lifetime + future_retention.sum()
                extended_ltv = arppu * extended_lifetime
            else:
                future_retention = pd.Series()
                extended_lifetime = lifetime
                extended_ltv = ltv
            
            return {
                'lifetime': lifetime,
                'ltv': ltv,
                'ltv_cac': ltv_cac,
                'roi_percent': roi_percent,
                'payback_period': payback_period,
                'cumulative_ltv': cumulative_ltv,
                'monthly_ltv': monthly_ltv,
                'churn_rates': churn_rates,
                'retention': retention,
                'future_retention': future_retention,
                'extended_lifetime': extended_lifetime,
                'extended_ltv': extended_ltv,
                'status': 'success'
            }
        
        except Exception as e:
            return {'status': f'Ошибка расчета: {str(e)}'}
    
    def create_sensitivity_analysis(self, ltv: float, base_cac: float, base_arppu: float) -> pd.DataFrame:
        """Создает расширенный анализ чувствительности."""
        try:
            # Анализ по CAC
            cac_range = np.linspace(max(1, base_cac * 0.5), base_cac * 2, 15)
            # Анализ по ARPPU
            arppu_range = np.linspace(max(1, base_arppu * 0.7), base_arppu * 1.5, 15)
            
            sensitivity_data = []
            
            # Чувствительность по CAC
            for cac in cac_range:
                ltv_cac_ratio = ltv / cac
                sensitivity_data.append({
                    'Параметр': 'CAC',
                    'Значение': int(cac),
                    'LTV/CAC': round(ltv_cac_ratio, 2),
                    'ROI %': round((ltv_cac_ratio - 1) * 100, 1),
                    'Статус': 'Прибыльно' if ltv_cac_ratio > 1 else 'Убыточно',
                    'Отклонение от базы %': round((cac / base_cac - 1) * 100, 1)
                })
            
            # Чувствительность по ARPPU
            lifetime = ltv / base_arppu  # восстанавливаем lifetime
            for arppu in arppu_range:
                new_ltv = arppu * lifetime
                ltv_cac_ratio = new_ltv / base_cac
                sensitivity_data.append({
                    'Параметр': 'ARPPU',
                    'Значение': int(arppu),
                    'LTV/CAC': round(ltv_cac_ratio, 2),
                    'ROI %': round((ltv_cac_ratio - 1) * 100, 1),
                    'Статус': 'Прибыльно' if ltv_cac_ratio > 1 else 'Убыточно',
                    'Отклонение от базы %': round((arppu / base_arppu - 1) * 100, 1)
                })
            
            return pd.DataFrame(sensitivity_data)
        
        except Exception as e:
            logger.error(f"Ошибка анализа чувствительности: {str(e)}")
            return pd.DataFrame()

# Создаем экземпляр калькулятора
@st.cache_resource
def get_calculator():
    return LifetimeCalculator()

calculator = get_calculator()

def create_plotly_charts(metrics: Dict) -> Tuple[Any, Any, Any]:
    """Создает интерактивные графики с Plotly."""
    retention = metrics['retention']
    cumulative_ltv = metrics['cumulative_ltv']
    monthly_ltv = metrics['monthly_ltv']
    
    months = list(range(1, len(retention) + 1))
    
    # График 1: Retention кривая
    fig_retention = go.Figure()
    fig_retention.add_trace(go.Scatter(
        x=months, 
        y=retention,
        mode='lines+markers',
        name='Retention %',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    # Добавляем прогноз если есть
    if len(metrics['future_retention']) > 0:
        future_months = list(range(len(retention) + 1, len(retention) + len(metrics['future_retention']) + 1))
        fig_retention.add_trace(go.Scatter(
            x=future_months,
            y=metrics['future_retention'],
            mode='lines+markers',
            name='Прогноз retention',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig_retention.update_layout(
        title="Кривая удержания пользователей",
        xaxis_title="Месяц",
        yaxis_title="Retention (доля)",
        hovermode='x unified'
    )
    
    # График 2: Рост LTV
    fig_ltv = go.Figure()
    fig_ltv.add_trace(go.Scatter(
        x=months,
        y=cumulative_ltv,
        mode='lines+markers',
        name='Кумулятивный LTV',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        fill='tonexty'
    ))
    
    fig_ltv.update_layout(
        title="Накопление LTV по месяцам",
        xaxis_title="Месяц", 
        yaxis_title="LTV (руб.)",
        hovermode='x unified'
    )
    
    # График 3: Месячный LTV
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=months,
        y=monthly_ltv,
        name='Месячный LTV',
        marker_color='#764ba2'
    ))
    
    fig_monthly.update_layout(
        title="Месячный вклад в LTV",
        xaxis_title="Месяц",
        yaxis_title="LTV за месяц (руб.)",
        hovermode='x unified'
    )
    
    return fig_retention, fig_ltv, fig_monthly

def export_results_to_excel(metrics: Dict, sensitivity_df: pd.DataFrame, filename: str = "ltv_analysis.xlsx"):
    """Экспорт результатов в Excel."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Основные метрики
        main_metrics = pd.DataFrame({
            'Метрика': ['Lifetime (мес)', 'LTV (руб)', 'LTV/CAC', 'ROI (%)', 'Период окупаемости (мес)'],
            'Значение': [
                round(metrics['lifetime'], 2),
                round(metrics['ltv'], 0),
                round(metrics['ltv_cac'], 2),
                round(metrics['roi_percent'], 1),
                metrics['payback_period'] or 'Не окупается'
            ]
        })
        main_metrics.to_excel(writer, sheet_name='Основные метрики', index=False)
        
        # Детальные данные по месяцам
        monthly_data = pd.DataFrame({
            'Месяц': range(1, len(metrics['retention']) + 1),
            'Retention': metrics['retention'].round(4),
            'Кумулятивный LTV': metrics['cumulative_ltv'].round(0),
            'Месячный LTV': metrics['monthly_ltv'].round(0),
            'Churn Rate': metrics['churn_rates'].round(4)
        })
        monthly_data.to_excel(writer, sheet_name='Помесячные данные', index=False)
        
        # Анализ чувствительности
        sensitivity_df.to_excel(writer, sheet_name='Анализ чувствительности', index=False)
        
        # Прогноз
        if len(metrics['future_retention']) > 0:
            forecast_data = pd.DataFrame({
                'Месяц': range(len(metrics['retention']) + 1, len(metrics['retention']) + len(metrics['future_retention']) + 1),
                'Прогноз Retention': metrics['future_retention'].round(4)
            })
            forecast_data.to_excel(writer, sheet_name='Прогноз', index=False)
    
    return output.getvalue()

def main():
    """Главная функция приложения."""
    
    # Заголовок
    st.markdown('<h1 class="main-header">📈 Lifetime & LTV Calculator</h1>', unsafe_allow_html=True)
    st.markdown("**Профессиональный инструмент для анализа когортной retention и расчета ключевых метрик**")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки анализа")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "📁 Загрузите файл с retention данными",
            type=['csv', 'xlsx', 'xls'],
            help="Поддерживаются форматы: CSV, Excel"
        )
        
        if uploaded_file:
            # Чтение файла
            df, status = calculator.read_file(uploaded_file)
            
            if status == "success":
                st.success("✅ Файл загружен успешно!")
                
                # Предварительный просмотр данных
                with st.expander("👀 Предварительный просмотр данных"):
                    st.dataframe(df.head())
                
                # Выбор колонки
                retention_col = st.selectbox(
                    "📊 Колонка с retention данными",
                    ["Автопоиск"] + list(df.columns),
                    help="Выберите колонку с процентами удержания"
                )
                
                if retention_col == "Автопоиск":
                    retention_col = None
            else:
                st.error(f"❌ Ошибка загрузки: {status}")
                return
        else:
            st.info("👆 Загрузите файл для начала анализа")
            
            # Показываем пример данных
            st.markdown("### 📋 Пример формата данных:")
            example_data = pd.DataFrame({
                'Месяц': range(1, 13),
                '% от первого месяца': [100, 51.7, 42.4, 36.8, 32.1, 28.9, 26.4, 24.3, 22.6, 21.1, 19.8, 18.7]
            })
            st.dataframe(example_data)
            return
        
        st.markdown("---")
        
        # Параметры бизнеса
        st.subheader("💼 Бизнес-параметры")
        
        col1, col2 = st.columns(2)
        with col1:
            arppu = st.number_input(
                "ARPPU (руб.)",
                min_value=0.01,
                value=71.0,
                step=1.0,
                help="Средний доход на одного платящего пользователя"
            )
        
        with col2:
            cac = st.number_input(
                "CAC (руб.)",
                min_value=0.01,
                value=184.0,
                step=1.0,
                help="Стоимость привлечения одного клиента"
            )
        
        # Дополнительные настройки
        st.markdown("---")
        st.subheader("🔧 Дополнительные настройки")
        
        show_forecast = st.checkbox("📈 Показать прогноз", value=True)
        show_sensitivity = st.checkbox("🎯 Анализ чувствительности", value=True)
        
        # Кнопка расчета
        calculate_button = st.button("🚀 Рассчитать метрики", type="primary", use_container_width=True)
    
    # Основная область контента
    if uploaded_file and calculate_button:
        with st.spinner("⏳ Выполняем расчеты..."):
            # Поиск колонки retention
            col_name, col_status = calculator.find_retention_column(df, retention_col)
            
            if not col_name:
                st.error(f"❌ {col_status}")
                st.info(f"Доступные колонки: {', '.join(df.columns)}")
                return
            
            if col_status.startswith("warning"):
                st.warning(f"⚠️ {col_status}")
            
            # Валидация данных
            retention_data, validation_status = calculator.validate_retention_data(df[col_name])
            
            if validation_status != "success":
                st.error(f"❌ {validation_status}")
                return
            
            # Расчет метрик
            metrics = calculator.calculate_metrics(retention_data, arppu, cac)
            
            if metrics['status'] != 'success':
                st.error(f"❌ {metrics['status']}")
                return
        
        # Отображение результатов
        st.success("✅ Расчеты выполнены успешно!")
        
        # Основные метрики в карточках
        st.markdown("## 📊 Ключевые метрики")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⏱️ Lifetime",
                f"{metrics['lifetime']:.2f} мес",
                help="Среднее время жизни клиента"
            )
        
        with col2:
            st.metric(
                "💎 LTV",
                f"{metrics['ltv']:,.0f} ₽",
                delta=f"{metrics['extended_ltv'] - metrics['ltv']:,.0f} ₽ прогноз" if show_forecast else None,
                help="Общий доход с одного клиента"
            )
        
        with col3:
            ratio_color = "normal" if 1 <= metrics['ltv_cac'] <= 3 else ("inverse" if metrics['ltv_cac'] < 1 else "normal")
            st.metric(
                "📈 LTV/CAC",
                f"{metrics['ltv_cac']:.2f}",
                delta=f"{metrics['roi_percent']:+.1f}% ROI",
                help="Отношение дохода к затратам на привлечение"
            )
        
        with col4:
            payback_text = f"{metrics['payback_period']} мес" if metrics['payback_period'] else "Не окупается"
            st.metric(
                "💰 Окупаемость",
                payback_text,
                help="Время окупаемости клиента"
            )
        
        # Интерпретация результатов
        st.markdown("## 🎯 Интерпретация результатов")
        
        if metrics['ltv_cac'] < 1:
            st.error("🔴 **Убыточная модель**: LTV < CAC. Требуется оптимизация привлечения или увеличение ARPPU.")
        elif metrics['ltv_cac'] < 3:
            st.warning("🟡 **Окупается, но есть риски**: LTV/CAC = 1-3. Рекомендуется улучшение retention или снижение CAC.")
        else:
            st.success("🟢 **Отличная модель**: LTV/CAC > 3. Можно масштабировать привлечение клиентов.")
        
        # Графики
        st.markdown("## 📈 Визуализация данных")
        
        fig_retention, fig_ltv, fig_monthly = create_plotly_charts(metrics)
        
        tab1, tab2, tab3 = st.tabs(["📉 Retention кривая", "📈 Рост LTV", "📊 Месячный LTV"])
        
        with tab1:
            st.plotly_chart(fig_retention, use_container_width=True)
        
        with tab2:
            st.plotly_chart(fig_ltv, use_container_width=True)
        
        with tab3:
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Детальные данные
        st.markdown("## 📋 Детальные данные")
        
        detailed_data = pd.DataFrame({
            'Месяц': range(1, len(metrics['retention']) + 1),
            'Retention %': (metrics['retention'] * 100).round(2),
            'Кумулятивный LTV': metrics['cumulative_ltv'].round(0),
            'Месячный LTV': metrics['monthly_ltv'].round(0),
            'Churn Rate %': (metrics['churn_rates'] * 100).round(2)
        })
        
        st.dataframe(detailed_data, use_container_width=True)
        
        # Анализ чувствительности
        if show_sensitivity:
            st.markdown("## 🎯 Анализ чувствительности")
            sensitivity_df = calculator.create_sensitivity_analysis(metrics['ltv'], cac, arppu)
            
            if not sensitivity_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 По CAC")
                    cac_data = sensitivity_df[sensitivity_df['Параметр'] == 'CAC']
                    st.dataframe(cac_data, use_container_width=True)
                
                with col2:
                    st.subheader("📊 По ARPPU")
                    arppu_data = sensitivity_df[sensitivity_df['Параметр'] == 'ARPPU']
                    st.dataframe(arppu_data, use_container_width=True)
        
        # Экспорт результатов
        st.markdown("## 💾 Экспорт результатов")
        
        excel_data = export_results_to_excel(metrics, sensitivity_df if show_sensitivity else pd.DataFrame())
        
        st.download_button(
            label="📥 Скачать Excel отчет",
            data=excel_data,
            file_name=f"ltv_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Справочная информация
    with st.expander("📚 Справка по метрикам и методологии"):
        st.markdown("""
        ### 📖 Основные понятия
        
        - **Lifetime** — среднее время жизни клиента (сумма retention по месяцам)
        - **LTV** (Lifetime Value) — общий доход с одного клиента за весь период жизни
        - **CAC** (Customer Acquisition Cost) — стоимость привлечения одного клиента
        - **ARPPU** (Average Revenue Per Paying User) — средний доход на платящего пользователя
        - **Retention** — доля пользователей, которые остались активными в определенный период
        
        ### 🧮 Формулы расчета
        
        ```
        Lifetime = Σ(Retention по месяцам)
        LTV = ARPPU × Lifetime
        LTV/CAC = LTV ÷ CAC
        ROI = (LTV/CAC - 1) × 100%
        ```
        
        ### 🎯 Интерпретация LTV/CAC
        
        - **< 1.0** — Убыточная модель (тратим больше, чем зарабатываем)
        - **1.0-3.0** — Окупается, но требует оптимизации
        - **> 3.0** — Отличная модель для масштабирования
        
        ### 📊 Рекомендации по улучшению
        
        **Для увеличения LTV:**
        - Улучшение retention (продуктовые фичи, engagement)
        - Увеличение ARPPU (upsell, cross-sell, ценообразование)
        
        **For снижения CAC:**
        - Оптимизация рекламных каналов
        - Улучшение конверсии
        - Развитие органических каналов
        """)

if __name__ == "__main__":
    main()
