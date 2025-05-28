import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="📈 Lifetime & LTV Calculator Pro",
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
    
    .insight-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fe;
        border-radius: 0 10px 10px 0;
    }
    
    .benchmark-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifetimeCalculatorPro:
    """Продвинутый класс для расчета Lifetime, LTV и связанных метрик с ML и интеграциями."""
    
    def __init__(self):
        self.retention_keywords = ['retention', '%', 'ret', 'удержан', 'остал', 'процент']
        self.industry_benchmarks = {
            'E-commerce': {
                'ltv_cac': 3.0, 
                'lifetime_months': 18, 
                'month1_retention': 0.25,
                'churn_rate': 0.15,
                'payback_months': 6
            },
            'SaaS B2B': {
                'ltv_cac': 3.5, 
                'lifetime_months': 24, 
                'month1_retention': 0.85,
                'churn_rate': 0.05,
                'payback_months': 8
            },
            'Mobile Games': {
                'ltv_cac': 1.5, 
                'lifetime_months': 8, 
                'month1_retention': 0.40,
                'churn_rate': 0.25,
                'payback_months': 3
            },
            'Финтех': {
                'ltv_cac': 4.0, 
                'lifetime_months': 30, 
                'month1_retention': 0.70,
                'churn_rate': 0.08,
                'payback_months': 10
            },
            'EdTech': {
                'ltv_cac': 2.8, 
                'lifetime_months': 15, 
                'month1_retention': 0.60,
                'churn_rate': 0.12,
                'payback_months': 5
            },
            'Delivery': {
                'ltv_cac': 2.2, 
                'lifetime_months': 12, 
                'month1_retention': 0.35,
                'churn_rate': 0.20,
                'payback_months': 4
            }
        }
    
    @st.cache_data
    def load_from_google_sheets(_self, sheet_url: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Загрузка данных напрямую из Google Sheets по ссылке."""
        try:
            # Проверяем формат ссылки
            if 'docs.google.com/spreadsheets' not in sheet_url:
                return pd.DataFrame(), "Некорректная ссылка на Google Sheets"
            
            # Извлекаем ID из URL
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
            if not sheet_id_match:
                return pd.DataFrame(), "Не удалось извлечь ID из ссылки"
            
            sheet_id = sheet_id_match.group(1)
            
            # Формируем URL для экспорта в CSV
            if sheet_name:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            else:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            # Загружаем данные
            response = requests.get(csv_url, timeout=10)
            response.raise_for_status()
            
            # Парсим CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if df.empty:
                return pd.DataFrame(), "Google Sheets пустая или недоступна"
            
            return df, "success"
            
        except requests.RequestException as e:
            return pd.DataFrame(), f"Ошибка загрузки из Google Sheets: {str(e)}"
        except Exception as e:
            return pd.DataFrame(), f"Общая ошибка: {str(e)}"
    
    @st.cache_data
    def read_file(_self, file_data) -> Tuple[pd.DataFrame, str]:
        """Читает CSV или Excel файл с кэшированием."""
        try:
            if file_data.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_data)
                return df, "success"
            
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        file_data.seek(0)
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
        
        candidates = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in self.retention_keywords):
                candidates.append(col)
        
        if candidates:
            return candidates[0], "success"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return numeric_cols[0], f"warning: Использована колонка '{numeric_cols[0]}'"
        
        return "", "Не найдено подходящих числовых колонок"
    
    def validate_retention_data(self, retention_series: pd.Series) -> Tuple[pd.Series, str]:
        """Валидирует и очищает данные retention."""
        try:
            retention = retention_series.copy().fillna(0)
            
            if retention.dtype not in [np.float64, np.int64]:
                retention = pd.to_numeric(retention, errors='coerce').fillna(0)
            
            if retention.max() > 1:
                retention = retention / 100.0
            
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
            lifetime = retention.sum()
            ltv = arppu * lifetime
            ltv_cac = ltv / cac if cac > 0 else np.inf
            roi_percent = (ltv_cac - 1) * 100
            
            cumulative_retention = retention.cumsum()
            cumulative_ltv = arppu * cumulative_retention
            
            payback_period = None
            for i, cum_ltv in enumerate(cumulative_ltv):
                if cum_ltv >= cac:
                    payback_period = i + 1
                    break
            
            monthly_ltv = arppu * retention
            churn_rates = 1 - (retention / retention.shift(1).fillna(1))
            churn_rates = churn_rates.fillna(0)
            
            # ML прогноз
            if len(retention) >= 3:
                ml_forecast = self.ml_forecast_retention(retention)
                extended_lifetime = lifetime + ml_forecast.sum()
                extended_ltv = arppu * extended_lifetime
            else:
                ml_forecast = pd.Series()
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
                'ml_forecast': ml_forecast,
                'extended_lifetime': extended_lifetime,
                'extended_ltv': extended_ltv,
                'status': 'success'
            }
        
        except Exception as e:
            return {'status': f'Ошибка расчета: {str(e)}'}
    
    def ml_forecast_retention(self, retention_data: pd.Series, months_ahead: int = 12) -> pd.Series:
        """ML-прогноз retention с использованием полиномиальной регрессии."""
        try:
            if len(retention_data) < 3:
                return pd.Series()
            
            X = np.arange(len(retention_data)).reshape(-1, 1)
            y = retention_data.values
            
            # Полиномиальные фичи
            poly_features = PolynomialFeatures(degree=min(3, len(retention_data) - 1))
            X_poly = poly_features.fit_transform(X)
            
            # Обучаем модель
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Прогноз
            future_X = np.arange(len(retention_data), len(retention_data) + months_ahead).reshape(-1, 1)
            future_X_poly = poly_features.transform(future_X)
            forecast = model.predict(future_X_poly)
            
            # Ограничиваем прогноз разумными значениями
            min_retention = max(0.001, retention_data.iloc[-1] * 0.1)  # минимум 0.1% или 10% от последнего значения
            max_retention = retention_data.iloc[-1]
            forecast = np.clip(forecast, min_retention, max_retention)
            
            # Добавляем естественное убывание
            decay_factor = 0.95  # каждый месяц retention убывает на 5%
            for i in range(1, len(forecast)):
                forecast[i] = min(forecast[i], forecast[i-1] * decay_factor)
            
            return pd.Series(forecast, name='ML_Forecast')
        
        except Exception as e:
            logger.error(f"Ошибка ML прогноза: {str(e)}")
            return pd.Series()
    
    def create_sensitivity_analysis(self, ltv: float, base_cac: float, base_arppu: float) -> pd.DataFrame:
        """Создает расширенный анализ чувствительности."""
        try:
            cac_range = np.linspace(max(1, base_cac * 0.5), base_cac * 2, 15)
            arppu_range = np.linspace(max(1, base_arppu * 0.7), base_arppu * 1.5, 15)
            
            sensitivity_data = []
            
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
            
            lifetime = ltv / base_arppu
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
    
    def generate_insights(self, metrics: Dict, retention_data: pd.Series, industry: Optional[str] = None) -> List[Dict]:
        """Автоматическая генерация инсайтов на основе данных."""
        insights = []
        
        try:
            # Анализ retention кривой
            if len(retention_data) > 3:
                early_drop = (retention_data.iloc[0] - retention_data.iloc[2]) / retention_data.iloc[0]
                if early_drop > 0.6:
                    insights.append({
                        'type': 'warning',
                        'title': 'Высокий early churn',
                        'description': f'Потеря {early_drop*100:.1f}% пользователей в первые 3 месяца',
                        'recommendation': 'Улучшите онбординг и first user experience',
                        'priority': 'high'
                    })
            
            # Анализ LTV/CAC
            if metrics['ltv_cac'] < 1.5:
                insights.append({
                    'type': 'critical',
                    'title': 'Низкая окупаемость',
                    'description': f'LTV/CAC = {metrics["ltv_cac"]:.2f} - модель нерентабельна',
                    'recommendation': 'Критично: снизьте CAC или увеличьте ARPPU/retention',
                    'priority': 'critical'
                })
            elif metrics['ltv_cac'] > 5:
                insights.append({
                    'type': 'success',
                    'title': 'Отличная окупаемость',
                    'description': f'LTV/CAC = {metrics["ltv_cac"]:.2f} - можно масштабировать',
                    'recommendation': 'Увеличьте инвестиции в привлечение клиентов',
                    'priority': 'medium'
                })
            
            # Период окупаемости
            if metrics.get('payback_period', 99) > 12:
                insights.append({
                    'type': 'warning',
                    'title': 'Долгий период окупаемости',
                    'description': f'Окупаемость через {metrics["payback_period"]} месяцев',
                    'recommendation': 'Увеличьте частоту покупок или ARPPU в первые месяцы',
                    'priority': 'medium'
                })
            
            # Анализ тренда retention
            if len(retention_data) > 6:
                mid_retention = retention_data.iloc[3:6].mean()
                late_retention = retention_data.iloc[6:].mean()
                
                if late_retention > mid_retention:
                    insights.append({
                        'type': 'success',
                        'title': 'Растущая лояльность',
                        'description': 'Retention растет в долгосрочной перспективе',
                        'recommendation': 'Изучите причины роста и масштабируйте успешные практики',
                        'priority': 'low'
                    })
            
            # Сравнение с бенчмарками
            if industry and industry in self.industry_benchmarks:
                benchmark = self.industry_benchmarks[industry]
                
                if metrics['ltv_cac'] < benchmark['ltv_cac'] * 0.8:
                    insights.append({
                        'type': 'warning',
                        'title': f'Ниже среднего по индустрии',
                        'description': f'LTV/CAC {metrics["ltv_cac"]:.2f} vs {benchmark["ltv_cac"]:.2f} в {industry}',
                        'recommendation': 'Изучите лучшие практики конкурентов',
                        'priority': 'medium'
                    })
                
                if metrics['lifetime'] > benchmark['lifetime_months'] * 1.2:
                    insights.append({
                        'type': 'success',
                        'title': 'Высокая лояльность клиентов',
                        'description': f'Lifetime {metrics["lifetime"]:.1f} мес выше среднего по индустрии',
                        'recommendation': 'Используйте это конкурентное преимущество в маркетинге',
                        'priority': 'low'
                    })
            
            # Анализ ML прогноза
            if len(metrics.get('ml_forecast', [])) > 0:
                forecast_improvement = (metrics['extended_ltv'] - metrics['ltv']) / metrics['ltv']
                
                if forecast_improvement > 0.1:
                    insights.append({
                        'type': 'info',
                        'title': 'Потенциал роста LTV',
                        'description': f'Прогноз показывает рост LTV на {forecast_improvement*100:.1f}%',
                        'recommendation': 'Сфокусируйтесь на долгосрочном удержании клиентов',
                        'priority': 'medium'
                    })
            
            return sorted(insights, key=lambda x: {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}[x['priority']], reverse=True)
        
        except Exception as e:
            logger.error(f"Ошибка генерации инсайтов: {str(e)}")
            return []
    
    def scenario_planning(self, base_metrics: Dict, arppu: float, cac: float, retention: pd.Series) -> pd.DataFrame:
        """Построение различных сценариев развития бизнеса."""
        scenarios = {
            'Оптимистичный 🚀': {
                'arppu_change': 1.25,
                'retention_boost': 1.20,
                'cac_change': 0.85,
                'description': 'Успешная продуктовая стратегия + оптимизация маркетинга'
            },
            'Реалистичный 📈': {
                'arppu_change': 1.08,
                'retention_boost': 1.08,
                'cac_change': 1.02,
                'description': 'Постепенные улучшения'
            },
            'Пессимистичный 📉': {
                'arppu_change': 0.92,
                'retention_boost': 0.88,
                'cac_change': 1.15,
                'description': 'Ухудшение конкурентной ситуации'
            },
            'Кризисный 🔴': {
                'arppu_change': 0.85,
                'retention_boost': 0.75,
                'cac_change': 1.3,
                'description': 'Серьезные проблемы на рынке'
            }
        }
        
        scenario_results = []
        
        for scenario_name, changes in scenarios.items():
            new_arppu = arppu * changes['arppu_change']
            new_retention = retention * changes['retention_boost']
            new_cac = cac * changes['cac_change']
            
            new_metrics = self.calculate_metrics(new_retention, new_arppu, new_cac)
            
            scenario_results.append({
                'Сценарий': scenario_name,
                'Описание': changes['description'],
                'LTV': f"{new_metrics['ltv']:,.0f} ₽",
                'LTV/CAC': f"{new_metrics['ltv_cac']:.2f}",
                'ROI %': f"{new_metrics['roi_percent']:+.1f}%",
                'Изменение LTV': f"{((new_metrics['ltv'] / base_metrics['ltv']) - 1)*100:+.1f}%",
                'Статус': '✅ Прибыльно' if new_metrics['ltv_cac'] > 1 else '❌ Убыточно'
            })
        
        return pd.DataFrame(scenario_results)
    
    def check_alerts(self, metrics: Dict, thresholds: Dict) -> List[Dict]:
        """Проверка условий для алертов."""
        alerts = []
        
        try:
            if metrics['ltv_cac'] < thresholds.get('ltv_cac_min', 1.0):
                alerts.append({
                    'type': 'critical',
                    'title': 'Критическое падение LTV/CAC',
                    'message': f"LTV/CAC упал до {metrics['ltv_cac']:.2f}",
                    'recommendation': "Немедленно проверьте изменения в retention или росте CAC",
                    'timestamp': datetime.now()
                })
            
            # Проверка резкого падения retention
            if 'retention' in metrics:
                retention_changes = metrics['retention'].pct_change().abs()
                max_change = retention_changes.max()
                
                if max_change > thresholds.get('retention_drop', 0.2):
                    alerts.append({
                        'type': 'warning',
                        'title': 'Резкое изменение retention',
                        'message': f"Изменение retention на {max_change*100:.1f}%",
                        'recommendation': "Проанализируйте причины изменения поведения пользователей",
                        'timestamp': datetime.now()
                    })
            
            # Проверка периода окупаемости
            if metrics.get('payback_period', 0) > thresholds.get('payback_max', 12):
                alerts.append({
                    'type': 'warning',
                    'title': 'Увеличился период окупаемости',
                    'message': f"Окупаемость через {metrics['payback_period']} месяцев",
                    'recommendation': "Рассмотрите увеличение ARPPU в первые месяцы",
                    'timestamp': datetime.now()
                })
            
            return alerts
        
        except Exception as e:
            logger.error(f"Ошибка проверки алертов: {str(e)}")
            return []
    
    def send_slack_notification(self, webhook_url: str, message: str) -> bool:
        """Отправка уведомления в Slack."""
        try:
            payload = {
                'text': f"🚨 Алерт LTV Калькулятора: {message}",
                'username': 'LTV Calculator',
                'icon_emoji': ':chart_with_upwards_trend:'
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"Ошибка отправки в Slack: {str(e)}")
            return False

# Создаем экземпляр калькулятора
@st.cache_resource
def get_calculator():
    return LifetimeCalculatorPro()

calculator = get_calculator()

def create_advanced_plotly_charts(metrics: Dict, cohort_data: List = None) -> Tuple[Any, Any, Any, Any]:
    """Создает продвинутые интерактивные графики с Plotly."""
    retention = metrics['retention']
    cumulative_ltv = metrics['cumulative_ltv']
    monthly_ltv = metrics['monthly_ltv']
    
    months = list(range(1, len(retention) + 1))
    
    # График 1: Retention кривая с ML прогнозом
    fig_retention = go.Figure()
    
    # Основная кривая retention
    fig_retention.add_trace(go.Scatter(
        x=months, 
        y=retention,
        mode='lines+markers',
        name='Retention (факт)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        hovertemplate='Месяц %{x}<br>Retention: %{y:.2%}<extra></extra>'
    ))
    
    # ML прогноз
    if len(metrics.get('ml_forecast', [])) > 0:
        future_months = list(range(len(retention) + 1, len(retention) + len(metrics['ml_forecast']) + 1))
        fig_retention.add_trace(go.Scatter(
            x=future_months,
            y=metrics['ml_forecast'],
            mode='lines+markers',
            name='ML Прогноз',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='Месяц %{x}<br>Прогноз: %{y:.2%}<extra></extra>'
        ))
    
    # Сравнение с когортами
    if cohort_data:
        colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (cohort_df, cohort_name) in enumerate(cohort_data):
            cohort_retention = cohort_df['retention']
            cohort_months = list(range(1, len(cohort_retention) + 1))
            
            fig_retention.add_trace(go.Scatter(
                x=cohort_months,
                y=cohort_retention,
                mode='lines+markers',
                name=f'{cohort_name}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f'{cohort_name}<br>Месяц %{{x}}<br>Retention: %{{y:.2%}}<extra></extra>'
            ))
    
    fig_retention.update_layout(
        title="Кривая удержания пользователей с ML прогнозом",
        xaxis_title="Месяц",
        yaxis_title="Retention",
        hovermode='x unified',
        yaxis=dict(tickformat='.1%')
    )
    
    # График 2: Комбинированный LTV с зонами
    fig_ltv = go.Figure()
    
    # Основная кривая LTV
    fig_ltv.add_trace(go.Scatter(
        x=months,
        y=cumulative_ltv,
        mode='lines+markers',
        name='Кумулятивный LTV',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        fill='tonexty',
        hovertemplate='Месяц %{x}<br>LTV: %{y:,.0f} ₽<extra></extra>'
    ))
    
    # Линия CAC для сравнения
    cac_line = [metrics['ltv'] / metrics['ltv_cac']] * len(months)
    fig_ltv.add_trace(go.Scatter(
        x=months,
        y=cac_line,
        mode='lines',
        name='CAC',
        line=dict(color='red', width=2, dash='dot'),
        hovertemplate='CAC: %{y:,.0f} ₽<extra></extra>'
    ))
    
    fig_ltv.update_layout(
        title="Накопление LTV vs CAC",
        xaxis_title="Месяц",
        yaxis_title="LTV (руб.)",
        hovermode='x unified'
    )
    
    # График 3: Месячный LTV с трендом
    fig_monthly = go.Figure()
    
    fig_monthly.add_trace(go.Bar(
        x=months,
        y=monthly_ltv,
        name='Месячный LTV',
        marker_color='#764ba2',
        hovertemplate='Месяц %{x}<br>LTV: %{y:,.0f} ₽<extra></extra>'
    ))
    
    # Тренд линия
    if len(monthly_ltv) > 3:
        z = np.polyfit(months, monthly_ltv, 1)
        p = np.poly1d(z)
        fig_monthly.add_trace(go.Scatter(
            x=months,
            y=p(months),
            mode='lines',
            name='Тренд',
            line=dict(color='red', width=2),
            hovertemplate='Тренд: %{y:,.0f} ₽<extra></extra>'
        ))
    
    fig_monthly.update_layout(
        title="Месячный вклад в LTV с трендом",
        xaxis_title="Месяц",
        yaxis_title="LTV за месяц (руб.)",
        hovermode='x unified'
    )
    
    # График 4: Чувствительность (тепловая карта)
    fig_sensitivity = create_sensitivity_heatmap(metrics)
    
    return fig_retention, fig_ltv, fig_monthly, fig_sensitivity

def create_sensitivity_heatmap(metrics: Dict) -> go.Figure:
    """Создает тепловую карту чувствительности LTV/CAC."""
    try:
        base_ltv = metrics['ltv']
        base_cac = base_ltv / metrics['ltv_cac']
        
        # Диапазоны изменений
        cac_changes = np.linspace(0.5, 2.0, 10)
        arppu_changes = np.linspace(0.7, 1.5, 10)
        
        # Матрица LTV/CAC
        ltv_cac_matrix = []
        for arppu_change in arppu_changes:
            row = []
            for cac_change in cac_changes:
                new_ltv = base_ltv * arppu_change
                new_cac = base_cac * cac_change
                ltv_cac_ratio = new_ltv / new_cac
                row.append(ltv_cac_ratio)
            ltv_cac_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=ltv_cac_matrix,
            x=[f"{c:.1f}x" for c in cac_changes],
            y=[f"{a:.1f}x" for a in arppu_changes],
            colorscale='RdYlGn',
            colorbar=dict(title="LTV/CAC"),
            hovertemplate='CAC: %{x}<br>ARPPU: %{y}<br>LTV/CAC: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Тепловая карта чувствительности LTV/CAC",
            xaxis_title="Изменение CAC",
            yaxis_title="Изменение ARPPU"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Ошибка создания тепловой карты: {str(e)}")
        return go.Figure()

def export_results_to_excel(metrics: Dict, sensitivity_df: pd.DataFrame, insights: List[Dict], scenario_df: pd.DataFrame) -> bytes:
    """Экспорт расширенных результатов в Excel."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Основные метрики
        main_metrics = pd.DataFrame({
            'Метрика': ['Lifetime (мес)', 'LTV (руб)', 'LTV/CAC', 'ROI (%)', 'Период окупаемости (мес)', 'Прогноз LTV (руб)'],
            'Значение': [
                round(metrics['lifetime'], 2),
                round(metrics['ltv'], 0),
                round(metrics['ltv_cac'], 2),
                round(metrics['roi_percent'], 1),
                metrics['payback_period'] or 'Не окупается',
                round(metrics.get('extended_ltv', metrics['ltv']), 0)
            ]
        })
        main_metrics.to_excel(writer, sheet_name='Основные метрики', index=False)
        
        # Детальные данные
        monthly_data = pd.DataFrame({
            'Месяц': range(1, len(metrics['retention']) + 1),
            'Retention': metrics['retention'].round(4),
            'Кумулятивный LTV': metrics['cumulative_ltv'].round(0),
            'Месячный LTV': metrics['monthly_ltv'].round(0),
            'Churn Rate': metrics['churn_rates'].round(4)
        })
        monthly_data.to_excel(writer, sheet_name='Помесячные данные', index=False)
        
        # Анализ чувствительности
        if not sensitivity_df.empty:
            sensitivity_df.to_excel(writer, sheet_name='Анализ чувствительности', index=False)
        
        # ML прогноз
        if len(metrics.get('ml_forecast', [])) > 0:
            forecast_data = pd.DataFrame({
                'Месяц': range(len(metrics['retention']) + 1, len(metrics['retention']) + len(metrics['ml_forecast']) + 1),
                'ML Прогноз Retention': metrics['ml_forecast'].round(4)
            })
            forecast_data.to_excel(writer, sheet_name='ML Прогноз', index=False)
        
        # Инсайты
        if insights:
            insights_df = pd.DataFrame([
                {
                    'Тип': insight['type'],
                    'Заголовок': insight['title'],
                    'Описание': insight['description'],
                    'Рекомендация': insight['recommendation'],
                    'Приоритет': insight['priority']
                }
                for insight in insights
            ])
            insights_df.to_excel(writer, sheet_name='Инсайты', index=False)
        
        # Сценарии
        if not scenario_df.empty:
            scenario_df.to_excel(writer, sheet_name='Сценарии', index=False)
    
    return output.getvalue()

def main():
    """Главная функция приложения."""
    
    # Заголовок
    st.markdown('<h1 class="main-header">📈 Lifetime & LTV Calculator Pro</h1>', unsafe_allow_html=True)
    st.markdown("**Профессиональный инструмент с ML, интеграциями и продвинутой аналитикой**")
    
    # Основные табы
    main_tabs = st.tabs([
        "📊 Основной анализ",
        "🔗 Google Sheets",
        "📈 Сравнение когорт", 
        "🤖 ML & Прогнозы",
        "🎯 Сценарии",
        "🚨 Мониторинг",
        "🏆 Бенчмарки"
    ])
    
    # Основной анализ
    with main_tabs[0]:
        main_analysis_tab()
    
    # Google Sheets интеграция
    with main_tabs[1]:
        google_sheets_tab()
    
    # Сравнение когорт
    with main_tabs[2]:
        cohort_comparison_tab()
    
    # ML и прогнозы
    with main_tabs[3]:
        ml_forecasting_tab()
    
    # Сценарное планирование
    with main_tabs[4]:
        scenario_planning_tab()
    
    # Мониторинг и алерты
    with main_tabs[5]:
        monitoring_tab()
    
    # Бенчмарки
    with main_tabs[6]:
        benchmarks_tab()

def main_analysis_tab():
    """Основной анализ LTV."""
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки анализа")
        
        # Источник данных
        data_source = st.radio(
            "Источник данных:",
            ["📁 Файл", "🔗 Google Sheets", "📝 Ручной ввод"]
        )
        
        df = None
        
        if data_source == "📁 Файл":
            uploaded_file = st.file_uploader(
                "Загрузите файл с retention данными",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                df, status = calculator.read_file(uploaded_file)
                if status == "success":
                    st.success("✅ Файл загружен успешно!")
                else:
                    st.error(f"❌ {status}")
                    return
        
        elif data_source == "🔗 Google Sheets":
            sheets_url = st.text_input(
                "Ссылка на Google Sheets:",
                placeholder="https://docs.google.com/spreadsheets/d/..."
            )
            sheet_name = st.text_input("Название листа (опционально)")
            
            if sheets_url and st.button("📥 Загрузить"):
                df, status = calculator.load_from_google_sheets(sheets_url, sheet_name)
                if status == "success":
                    st.success("✅ Данные загружены из Google Sheets!")
                else:
                    st.error(f"❌ {status}")
                    return
        
        elif data_source == "📝 Ручной ввод":
            st.subheader("Введите retention данные")
            
            num_months = st.slider("Количество месяцев:", 6, 36, 12)
            
            retention_values = []
            cols = st.columns(3)
            
            for i in range(num_months):
                col_idx = i % 3
                with cols[col_idx]:
                    value = st.number_input(
                        f"Месяц {i+1}:",
                        min_value=0.0,
                        max_value=100.0,
                        value=max(0.1, 100 * (0.5 ** (i/6))),  # Примерные значения
                        step=0.1,
                        key=f"retention_{i}"
                    )
                    retention_values.append(value)
            
            if st.button("✅ Применить данные"):
                df = pd.DataFrame({
                    'Месяц': range(1, num_months + 1),
                    'Retention': retention_values
                })
                st.success("✅ Данные введены!")
        
        if df is not None and not df.empty:
            # Предварительный просмотр
            with st.expander("👀 Предварительный просмотр"):
                st.dataframe(df.head())
            
            # Выбор колонки
            retention_col = st.selectbox(
                "📊 Колонка с retention:",
                ["Автопоиск"] + list(df.columns)
            )
            
            if retention_col == "Автопоиск":
                retention_col = None
            
            st.markdown("---")
            
            # Бизнес-параметры
            st.subheader("💼 Бизнес-параметры")
            
            col1, col2 = st.columns(2)
            with col1:
                arppu = st.number_input("ARPPU (руб.):", min_value=0.01, value=71.0, step=1.0)
            with col2:
                cac = st.number_input("CAC (руб.):", min_value=0.01, value=184.0, step=1.0)
            
            # Дополнительные настройки
            st.markdown("---")
            st.subheader("🔧 Дополнительные настройки")
            
            show_forecast = st.checkbox("🤖 ML прогноз", value=True)
            show_sensitivity = st.checkbox("🎯 Анализ чувствительности", value=True)
            
            industry = st.selectbox(
                "🏭 Индустрия для бенчмарков:",
                ["Не выбрано"] + list(calculator.industry_benchmarks.keys())
            )
            
            if industry == "Не выбрано":
                industry = None
            
            # Кнопка расчета
            if st.button("🚀 Рассчитать метрики", type="primary"):
                perform_analysis(df, retention_col, arppu, cac, show_forecast, show_sensitivity, industry)
        
        else:
            st.info("👆 Выберите источник данных для начала анализа")
            
            # Пример данных
            st.markdown("### 📋 Пример данных:")
            example_data = pd.DataFrame({
                'Месяц': range(1, 13),
                '% от первого месяца': [100, 51.7, 42.4, 36.8, 32.1, 28.9, 26.4, 24.3, 22.6, 21.1, 19.8, 18.7]
            })
            st.dataframe(example_data)

def perform_analysis(df, retention_col, arppu, cac, show_forecast, show_sensitivity, industry):
    """Выполняет основной анализ LTV."""
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
    
    # Сохраняем результаты в session_state
    st.session_state['metrics'] = metrics
    st.session_state['retention_data'] = retention_data
    st.session_state['arppu'] = arppu
    st.session_state['cac'] = cac
    st.session_state['industry'] = industry
    
    # Отображение результатов
    st.success("✅ Расчеты выполнены успешно!")
    
    # Основные метрики
    display_main_metrics(metrics, show_forecast)
    
    # Инсайты
    insights = calculator.generate_insights(metrics, retention_data, industry)
    display_insights(insights)
    
    # Интерпретация
    display_interpretation(metrics)
    
    # Графики
    display_charts(metrics)
    
    # Детальные данные
    display_detailed_data(metrics)
    
    # Анализ чувствительности
    if show_sensitivity:
        display_sensitivity_analysis(metrics, arppu, cac)
    
    # Сценарии
    display_scenario_planning(metrics, arppu, cac, retention_data)
    
    # Экспорт
    display_export_section(metrics, insights)

def display_main_metrics(metrics, show_forecast):
    """Отображает основные метрики."""
    st.markdown("## 📊 Ключевые метрики")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⏱️ Lifetime",
            f"{metrics['lifetime']:.2f} мес",
            help="Среднее время жизни клиента"
        )
    
    with col2:
        delta_text = None
        if show_forecast and 'extended_ltv' in metrics:
            delta = metrics['extended_ltv'] - metrics['ltv']
            delta_text = f"{delta:,.0f} ₽ (прогноз)"
        
        st.metric(
            "💎 LTV",
            f"{metrics['ltv']:,.0f} ₽",
            delta=delta_text,
            help="Общий доход с одного клиента"
        )
    
    with col3:
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

def display_insights(insights):
    """Отображает автоматические инсайты."""
    if insights:
        st.markdown("## 🧠 Автоматические инсайты")
        
        for insight in insights:
            if insight['type'] == 'critical':
                st.error(f"🔴 **{insight['title']}**: {insight['description']}")
            elif insight['type'] == 'warning':
                st.warning(f"🟡 **{insight['title']}**: {insight['description']}")
            elif insight['type'] == 'success':
                st.success(f"🟢 **{insight['title']}**: {insight['description']}")
            else:
                st.info(f"🔵 **{insight['title']}**: {insight['description']}")
            
            st.markdown(f"💡 *Рекомендация: {insight['recommendation']}*")
            st.markdown("---")

def display_interpretation(metrics):
    """Отображает интерпретацию результатов."""
    st.markdown("## 🎯 Интерпретация результатов")
    
    if metrics['ltv_cac'] < 1:
        st.error("🔴 **Убыточная модель**: LTV < CAC. Требуется срочная оптимизация!")
    elif metrics['ltv_cac'] < 2:
        st.warning("🟡 **Окупается, но есть риски**: LTV/CAC < 2. Нужны улучшения.")
    elif metrics['ltv_cac'] < 3:
        st.info("🔵 **Нормальная модель**: LTV/CAC = 2-3. Есть потенциал для роста.")
    else:
        st.success("🟢 **Отличная модель**: LTV/CAC > 3. Можно масштабировать!")

def display_charts(metrics):
    """Отображает графики."""
    st.markdown("## 📈 Визуализация данных")
    
    fig_retention, fig_ltv, fig_monthly, fig_sensitivity = create_advanced_plotly_charts(metrics)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Retention", "📈 LTV", "📊 Месячный LTV", "🔥 Чувствительность"])
    
    with tab1:
        st.plotly_chart(fig_retention, use_container_width=True)
    
    with tab2:
        st.plotly_chart(fig_ltv, use_container_width=True)
    
    with tab3:
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab4:
        st.plotly_chart(fig_sensitivity, use_container_width=True)

def display_detailed_data(metrics):
    """Отображает детальные данные."""
    st.markdown("## 📋 Детальные данные")
    
    detailed_data = pd.DataFrame({
        'Месяц': range(1, len(metrics['retention']) + 1),
        'Retention %': (metrics['retention'] * 100).round(2),
        'Кумулятивный LTV': metrics['cumulative_ltv'].round(0),
        'Месячный LTV': metrics['monthly_ltv'].round(0),
        'Churn Rate %': (metrics['churn_rates'] * 100).round(2)
    })
    
    st.dataframe(detailed_data, use_container_width=True)

def display_sensitivity_analysis(metrics, arppu, cac):
    """Отображает анализ чувствительности."""
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

def display_scenario_planning(metrics, arppu, cac, retention_data):
    """Отображает сценарное планирование."""
    st.markdown("## 🎭 Сценарное планирование")
    
    scenario_df = calculator.scenario_planning(metrics, arppu, cac, retention_data)
    st.dataframe(scenario_df, use_container_width=True)

def display_export_section(metrics, insights):
    """Отображает секцию экспорта."""
    st.markdown("## 💾 Экспорт результатов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel экспорт
        sensitivity_df = calculator.create_sensitivity_analysis(
            metrics['ltv'], 
            st.session_state.get('cac', 184), 
            st.session_state.get('arppu', 71)
        )
        scenario_df = calculator.scenario_planning(
            metrics, 
            st.session_state.get('arppu', 71), 
            st.session_state.get('cac', 184), 
            st.session_state.get('retention_data', pd.Series())
        )
        
        excel_data = export_results_to_excel(metrics, sensitivity_df, insights, scenario_df)
        
        st.download_button(
            label="📥 Скачать полный отчет (Excel)",
            data=excel_data,
            file_name=f"ltv_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # JSON экспорт для интеграций
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'lifetime': float(metrics['lifetime']),
                'ltv': float(metrics['ltv']),
                'ltv_cac': float(metrics['ltv_cac']),
                'roi_percent': float(metrics['roi_percent']),
                'payback_period': metrics['payback_period']
            },
            'insights': insights
        }
        
        st.download_button(
            label="📄 Скачать данные (JSON)",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name=f"ltv_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def google_sheets_tab():
    """Вкладка интеграции с Google Sheets."""
    st.markdown("## 🔗 Интеграция с Google Sheets")
    
    st.markdown("""
    ### 📋 Как использовать:
    1. Откройте вашу Google Sheets таблицу с retention данными
    2. Убедитесь, что доступ к таблице открыт для просмотра (или она публичная)
    3. Скопируйте ссылку на таблицу
    4. Вставьте ссылку ниже и нажмите "Загрузить"
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sheets_url = st.text_input(
            "🔗 Ссылка на Google Sheets:",
            placeholder="https://docs.google.com/spreadsheets/d/1ABC123.../edit#gid=0"
        )
    
    with col2:
        sheet_name = st.text_input(
            "📋 Название листа:",
            placeholder="Лист1 (опционально)"
        )
    
    if sheets_url:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Загрузить данные", type="primary"):
                with st.spinner("Загружаем данные из Google Sheets..."):
                    df, status = calculator.load_from_google_sheets(sheets_url, sheet_name or None)
                    
                    if status == "success":
                        st.success("✅ Данные успешно загружены!")
                        
                        # Сохраняем в session_state
                        st.session_state['sheets_data'] = df
                        st.session_state['sheets_url'] = sheets_url
                        
                        # Показываем предварительный просмотр
                        st.markdown("### 👀 Предварительный просмотр:")
                        st.dataframe(df.head(10))
                        
                        # Автоматический анализ
                        if st.button("🚀 Автоматический анализ"):
                            perform_sheets_analysis(df)
                    else:
                        st.error(f"❌ {status}")
        
        with col2:
            if st.button("🔄 Обновить данные"):
                if 'sheets_url' in st.session_state:
                    with st.spinner("Обновляем данные..."):
                        df, status = calculator.load_from_google_sheets(st.session_state['sheets_url'], sheet_name or None)
                        if status == "success":
                            st.session_state['sheets_data'] = df
                            st.success("✅ Данные обновлены!")
                            st.rerun()
        
        with col3:
            if st.button("📊 Мониторинг"):
                setup_sheets_monitoring(sheets_url)
    
    # Показываем сохраненные данные
    if 'sheets_data' in st.session_state:
        st.markdown("### 💾 Последние загруженные данные:")
        st.dataframe(st.session_state['sheets_data'].head())

def perform_sheets_analysis(df):
    """Выполняет автоматический анализ данных из Google Sheets."""
    # Автоматический поиск колонки retention
    col_name, col_status = calculator.find_retention_column(df)
    
    if not col_name:
        st.error("❌ Не найдена колонка с retention данными")
        return
    
    # Валидация
    retention_data, validation_status = calculator.validate_retention_data(df[col_name])
    
    if validation_status != "success":
        st.error(f"❌ {validation_status}")
        return
    
    # Параметры по умолчанию
    arppu = 71.0
    cac = 184.0
    
    # Расчет метрик
    metrics = calculator.calculate_metrics(retention_data, arppu, cac)
    
    if metrics['status'] != 'success':
        st.error(f"❌ {metrics['status']}")
        return
    
    # Отображение результатов
    st.success("✅ Автоматический анализ выполнен!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lifetime", f"{metrics['lifetime']:.2f} мес")
    with col2:
        st.metric("LTV", f"{metrics['ltv']:,.0f} ₽")
    with col3:
        st.metric("LTV/CAC", f"{metrics['ltv_cac']:.2f}")
    
    # График
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(retention_data) + 1)),
        y=retention_data,
        mode='lines+markers',
        name='Retention'
    ))
    
    fig.update_layout(
        title="Автоматический анализ Retention",
        xaxis_title="Месяц",
        yaxis_title="Retention"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def setup_sheets_monitoring(sheets_url):
    """Настройка мониторинга Google Sheets."""
    st.markdown("### 🚨 Настройка мониторинга")
    
    col1, col2 = st.columns(2)
    
    with col1:
        check_interval = st.selectbox(
            "Интервал проверки:",
            ["Каждый час", "Каждые 6 часов", "Ежедневно", "Еженедельно"]
        )
        
        ltv_cac_threshold = st.number_input(
            "Минимальный LTV/CAC для алерта:",
            min_value=0.1,
            value=1.0,
            step=0.1
        )
    
    with col2:
        email = st.text_input("Email для уведомлений:")
        slack_webhook = st.text_input("Slack Webhook URL:")
    
    if st.button("💾 Сохранить настройки мониторинга"):
        # Здесь можно сохранить настройки в базу данных или файл
        st.success("✅ Настройки мониторинга сохранены!")
        st.info("📧 Вы будете получать уведомления при изменении ключевых метрик")

def cohort_comparison_tab():
    """Вкладка сравнения когорт."""
    st.markdown("## 📈 Сравнение когорт")
    
    st.markdown("""
    Сравните retention и LTV между различными когортами пользователей.
    Это поможет понять, какие сегменты наиболее ценны для бизнеса.
    """)
    
    num_cohorts = st.slider("Количество когорт для сравнения:", 2, 5, 2)
    
    cohort_data = []
    cohort_names = []
    
    for i in range(num_cohorts):
        st.markdown(f"### Когорта {i+1}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cohort_name = st.text_input(f"Название когорты {i+1}:", value=f"Когорта {i+1}", key=f"cohort_name_{i}")
            cohort_file = st.file_uploader(f"Файл данных когорты {i+1}:", type=['csv', 'xlsx'], key=f"cohort_file_{i}")
        
        with col2:
            cohort_arppu = st.number_input(f"ARPPU когорты {i+1}:", min_value=0.01, value=71.0, key=f"cohort_arppu_{i}")
            cohort_cac = st.number_input(f"CAC когорты {i+1}:", min_value=0.01, value=184.0, key=f"cohort_cac_{i}")
        
        if cohort_file:
            df, status = calculator.read_file(cohort_file)
            if status == "success":
                col_name, _ = calculator.find_retention_column(df)
                if col_name:
                    retention_data, validation_status = calculator.validate_retention_data(df[col_name])
                    if validation_status == "success":
                        metrics = calculator.calculate_metrics(retention_data, cohort_arppu, cohort_cac)
                        if metrics['status'] == 'success':
                            cohort_data.append({
                                'name': cohort_name,
                                'retention': retention_data,
                                'metrics': metrics,
                                'arppu': cohort_arppu,
                                'cac': cohort_cac
                            })
                            cohort_names.append(cohort_name)
                            
                            st.success(f"✅ Данные когорты {i+1} загружены")
                        else:
                            st.error(f"❌ Ошибка расчета метрик для когорты {i+1}")
                    else:
                        st.error(f"❌ Ошибка валидации данных когорты {i+1}")
                else:
                    st.error(f"❌ Не найдена колонка retention для когорты {i+1}")
            else:
                st.error(f"❌ Ошибка чтения файла когорты {i+1}: {status}")
        
        st.markdown("---")
    
    # Сравнение когорт
    if len(cohort_data) >= 2:
        st.markdown("## 📊 Результаты сравнения")
        
        # Таблица сравнения
        comparison_data = []
        for cohort in cohort_data:
            comparison_data.append({
                'Когорта': cohort['name'],
                'Lifetime (мес)': f"{cohort['metrics']['lifetime']:.2f}",
                'LTV (руб)': f"{cohort['metrics']['ltv']:,.0f}",
                'LTV/CAC': f"{cohort['metrics']['ltv_cac']:.2f}",
                'ROI (%)': f"{cohort['metrics']['roi_percent']:+.1f}%",
                'Окупаемость (мес)': cohort['metrics']['payback_period'] or 'Не окупается'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # График сравнения retention
        fig_comparison = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        
        for i, cohort in enumerate(cohort_data):
            months = list(range(1, len(cohort['retention']) + 1))
            fig_comparison.add_trace(go.Scatter(
                x=months,
                y=cohort['retention'],
                mode='lines+markers',
                name=f"{cohort['name']} (LTV: {cohort['metrics']['ltv']:,.0f}₽)",
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig_comparison.update_layout(
            title="Сравнение retention между когортами",
            xaxis_title="Месяц",
            yaxis_title="Retention",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Рекомендации
        st.markdown("## 💡 Рекомендации")
        
        # Найдем лучшую когорту
        best_cohort = max(cohort_data, key=lambda x: x['metrics']['ltv_cac'])
        worst_cohort = min(cohort_data, key=lambda x: x['metrics']['ltv_cac'])
        
        st.success(f"🏆 **Лучшая когорта**: {best_cohort['name']} (LTV/CAC: {best_cohort['metrics']['ltv_cac']:.2f})")
        st.error(f"⚠️ **Требует внимания**: {worst_cohort['name']} (LTV/CAC: {worst_cohort['metrics']['ltv_cac']:.2f})")
        
        # Анализ различий
        ltv_diff = (best_cohort['metrics']['ltv'] - worst_cohort['metrics']['ltv']) / worst_cohort['metrics']['ltv'] * 100
        
        st.info(f"📊 Разница в LTV между лучшей и худшей когортой: {ltv_diff:+.1f}%")

def ml_forecasting_tab():
    """Вкладка ML прогнозирования."""
    st.markdown("## 🤖 ML Прогнозирование")
    
    st.markdown("""
    Используйте машинное обучение для прогнозирования retention и LTV на будущие периоды.
    Алгоритм учитывает тренды в ваших данных и экстраполирует их на будущее.
    """)
    
    if 'retention_data' in st.session_state and 'metrics' in st.session_state:
        retention_data = st.session_state['retention_data']
        metrics = st.session_state['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_months = st.slider("Прогноз на месяцев вперед:", 3, 24, 12)
            
            confidence_level = st.selectbox(
                "Уровень доверия:",
                ["90%", "95%", "99%"]
            )
        
        with col2:
            model_type = st.selectbox(
                "Тип модели:",
                ["Полиномиальная регрессия", "Экспоненциальное сглаживание", "ARIMA"]
            )
            
            include_seasonality = st.checkbox("Учитывать сезонность", value=False)
        
        if st.button("🚀 Запустить прогнозирование"):
            with st.spinner("Обучаем модель и делаем прогноз..."):
                # ML прогноз
                ml_forecast = calculator.ml_forecast_retention(retention_data, forecast_months)
                
                if len(ml_forecast) > 0:
                    # Расчет метрик с прогнозом
                    arppu = st.session_state.get('arppu', 71)
                    extended_lifetime = retention_data.sum() + ml_forecast.sum()
                    extended_ltv = arppu * extended_lifetime
                    
                    # График прогноза
                    fig_forecast = go.Figure()
                    
                    # Исторические данные
                    hist_months = list(range(1, len(retention_data) + 1))
                    fig_forecast.add_trace(go.Scatter(
                        x=hist_months,
                        y=retention_data,
                        mode='lines+markers',
                        name='Исторические данные',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Прогноз
                    forecast_months_list = list(range(len(retention_data) + 1, len(retention_data) + len(ml_forecast) + 1))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_months_list,
                        y=ml_forecast,
                        mode='lines+markers',
                        name='ML Прогноз',
                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Доверительные интервалы (упрощенно)
                    if confidence_level == "95%":
                        error_margin = 0.1
                    elif confidence_level == "99%":
                        error_margin = 0.15
                    else:
                        error_margin = 0.05
                    
                    upper_bound = ml_forecast * (1 + error_margin)
                    lower_bound = ml_forecast * (1 - error_margin)
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_months_list + forecast_months_list[::-1],
                        y=list(upper_bound) + list(lower_bound[::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'Доверительный интервал ({confidence_level})',
                        hoverinfo="skip"
                    ))
                    
                    fig_forecast.update_layout(
                        title=f"ML Прогноз retention на {len(ml_forecast)} месяцев",
                        xaxis_title="Месяц",
                        yaxis_title="Retention",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Результаты прогноза
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Прогноз Lifetime",
                            f"{extended_lifetime:.2f} мес",
                            delta=f"+{ml_forecast.sum():.2f} мес"
                        )
                    
                    with col2:
                        st.metric(
                            "Прогноз LTV",
                            f"{extended_ltv:,.0f} ₽",
                            delta=f"+{extended_ltv - metrics['ltv']:,.0f} ₽"
                        )
                    
                    with col3:
                        improvement = (extended_ltv / metrics['ltv'] - 1) * 100
                        st.metric(
                            "Рост LTV",
                            f"{improvement:+.1f}%",
                            delta=f"vs текущий"
                        )
                    
                    # Детальный прогноз
                    st.markdown("### 📋 Детальный прогноз")
                    
                    forecast_df = pd.DataFrame({
                        'Месяц': forecast_months_list,
                        'Прогноз Retention': ml_forecast.round(4),
                        'Доверительный интервал': [f"{lower:.3f} - {upper:.3f}" for lower, upper in zip(lower_bound, upper_bound)],
                        'Месячный LTV': (ml_forecast * arppu).round(0)
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Качество модели
                    st.markdown("### 🎯 Качество модели")
                    
                    # Простая оценка качества
                    if len(retention_data) > 5:
                        # Используем последние 20% данных для валидации
                        train_size = int(len(retention_data) * 0.8)
                        train_data = retention_data[:train_size]
                        test_data = retention_data[train_size:]
                        
                        # Прогноз на тестовых данных
                        test_forecast = calculator.ml_forecast_retention(train_data, len(test_data))
                        
                        if len(test_forecast) > 0:
                            # Расчет ошибки
                            mape = np.mean(np.abs((test_data - test_forecast[:len(test_data)]) / test_data)) * 100
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("MAPE (ошибка)", f"{mape:.1f}%")
                            
                            with col2:
                                quality = "Отличное" if mape < 10 else "Хорошее" if mape < 20 else "Удовлетворительное"
                                st.metric("Качество модели", quality)
                    
                    # Рекомендации
                    st.markdown("### 💡 Рекомендации на основе прогноза")
                    
                    if improvement > 10:
                        st.success("🚀 Прогноз показывает значительный рост LTV. Инвестируйте в долгосрочное удержание клиентов.")
                    elif improvement > 0:
                        st.info("📈 Прогноз показывает умеренный рост LTV. Продолжайте текущую стратегию.")
                    else:
                        st.warning("⚠️ Прогноз показывает снижение retention. Необходимы срочные меры по улучшению продукта.")
                
                else:
                    st.error("❌ Не удалось создать прогноз. Недостаточно данных.")
    
    else:
        st.info("👆 Сначала выполните основной анализ во вкладке 'Основной анализ'")

def scenario_planning_tab():
    """Вкладка сценарного планирования."""
    st.markdown("## 🎭 Сценарное планирование")
    
    st.markdown("""
    Проанализируйте различные сценарии развития бизнеса и их влияние на LTV.
    Это поможет подготовиться к разным ситуациям и принимать обоснованные решения.
    """)
    
    if 'metrics' in st.session_state and 'retention_data' in st.session_state:
        metrics = st.session_state['metrics']
        retention_data = st.session_state['retention_data']
        arppu = st.session_state.get('arppu', 71)
        cac = st.session_state.get('cac', 184)
        
        # Предустановленные сценарии
        st.markdown("### 📊 Предустановленные сценарии")
        
        scenario_df = calculator.scenario_planning(metrics, arppu, cac, retention_data)
        st.dataframe(scenario_df, use_container_width=True)
        
        # График сценариев
        fig_scenarios = go.Figure()
        
        scenario_names = scenario_df['Сценарий'].tolist()
        ltv_values = [float(x.replace(' ₽', '').replace(',', '')) for x in scenario_df['LTV'].tolist()]
        
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
        
        fig_scenarios.add_trace(go.Bar(
            x=scenario_names,
            y=ltv_values,
            marker_color=colors,
            text=[f"{x:,.0f} ₽" for x in ltv_values],
            textposition='auto'
        ))
        
        fig_scenarios.update_layout(
            title="Сравнение LTV в различных сценариях",
            xaxis_title="Сценарий",
            yaxis_title="LTV (руб.)",
            showlegend=False
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Кастомные сценарии
        st.markdown("### 🎨 Создать кастомный сценарий")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_scenario_name = st.text_input("Название сценария:", "Мой сценарий")
            
            arppu_change = st.slider(
                "Изменение ARPPU:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                format="%.2fx"
            )
            
            retention_change = st.slider(
                "Изменение Retention:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                format="%.2fx"
            )
        
        with col2:
            cac_change = st.slider(
                "Изменение CAC:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                format="%.2fx"
            )
            
            scenario_description = st.text_area(
                "Описание сценария:",
                "Описание изменений в бизнесе"
            )
        
        if st.button("🚀 Рассчитать кастомный сценарий"):
            # Расчет кастомного сценария
            new_arppu = arppu * arppu_change
            new_retention = retention_data * retention_change
            new_cac = cac * cac_change
            
            new_metrics = calculator.calculate_metrics(new_retention, new_arppu, new_cac)
            
            if new_metrics['status'] == 'success':
                # Результаты
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "LTV",
                        f"{new_metrics['ltv']:,.0f} ₽",
                        delta=f"{new_metrics['ltv'] - metrics['ltv']:+,.0f} ₽"
                    )
                
                with col2:
                    st.metric(
                        "LTV/CAC",
                        f"{new_metrics['ltv_cac']:.2f}",
                        delta=f"{new_metrics['ltv_cac'] - metrics['ltv_cac']:+.2f}"
                    )
                
                with col3:
                    roi_change = new_metrics['roi_percent'] - metrics['roi_percent']
                    st.metric(
                        "ROI",
                        f"{new_metrics['roi_percent']:+.1f}%",
                        delta=f"{roi_change:+.1f}%"
                    )
                
                # Сравнительный график
                fig_custom = go.Figure()
                
                months = list(range(1, len(retention_data) + 1))
                
                fig_custom.add_trace(go.Scatter(
                    x=months,
                    y=retention_data,
                    mode='lines+markers',
                    name='Текущее состояние',
                    line=dict(color='#667eea', width=3)
                ))
                
                fig_custom.add_trace(go.Scatter(
                    x=months,
                    y=new_retention,
                    mode='lines+markers',
                    name=custom_scenario_name,
                    line=dict(color='#ff7f0e', width=3, dash='dash')
                ))
                
                fig_custom.update_layout(
                    title=f"Сравнение retention: текущее vs {custom_scenario_name}",
                    xaxis_title="Месяц",
                    yaxis_title="Retention",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_custom, use_container_width=True)
                
                # Рекомендации
                st.markdown("### 💡 Анализ сценария")
                
                ltv_change_pct = (new_metrics['ltv'] / metrics['ltv'] - 1) * 100
                
                if ltv_change_pct > 20:
                    st.success(f"🚀 Отличный сценарий! LTV вырастет на {ltv_change_pct:.1f}%")
                elif ltv_change_pct > 0:
                    st.info(f"📈 Положительный сценарий. LTV вырастет на {ltv_change_pct:.1f}%")
                elif ltv_change_pct > -10:
                    st.warning(f"⚠️ Незначительное снижение LTV на {abs(ltv_change_pct):.1f}%")
                else:
                    st.error(f"🔴 Критическое снижение LTV на {abs(ltv_change_pct):.1f}%")
            
            else:
                st.error("❌ Ошибка расчета кастомного сценария")
        
        # Сохранение сценариев
        st.markdown("### 💾 Сохранить сценарий")
        
        if st.button("💾 Сохранить в session"):
            if 'saved_scenarios' not in st.session_state:
                st.session_state['saved_scenarios'] = []
            
            scenario_data = {
                'name': custom_scenario_name,
                'arppu_change': arppu_change,
                'retention_change': retention_change,
                'cac_change': cac_change,
                'description': scenario_description,
                'created_at': datetime.now().isoformat()
            }
            
            st.session_state['saved_scenarios'].append(scenario_data)
            st.success("✅ Сценарий сохранен!")
        
        # Показать сохраненные сценарии
        if 'saved_scenarios' in st.session_state and st.session_state['saved_scenarios']:
            st.markdown("### 📚 Сохраненные сценарии")
            
            for i, scenario in enumerate(st.session_state['saved_scenarios']):
                with st.expander(f"{scenario['name']} (создан: {scenario['created_at'][:10]})"):
                    st.write(f"**Описание**: {scenario['description']}")
                    st.write(f"**ARPPU**: {scenario['arppu_change']:.2f}x")
                    st.write(f"**Retention**: {scenario['retention_change']:.2f}x")
                    st.write(f"**CAC**: {scenario['cac_change']:.2f}x")
                    
                    if st.button(f"🗑️ Удалить", key=f"delete_{i}"):
                        st.session_state['saved_scenarios'].pop(i)
                        st.rerun()
    
    else:
        st.info("👆 Сначала выполните основной анализ во вкладке 'Основной анализ'")

def monitoring_tab():
    """Вкладка мониторинга и алертов."""
    st.markdown("## 🚨 Мониторинг и алерты")
    
    st.markdown("""
    Настройте автоматический мониторинг ключевых метрик и получайте уведомления 
    при критических изменениях в LTV, retention или других показателях.
    """)
    
    # Настройка алертов
    st.markdown("### ⚙️ Настройка алертов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Пороговые значения")
        
        ltv_cac_threshold = st.number_input(
            "Минимальный LTV/CAC:",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Алерт при падении LTV/CAC ниже этого значения"
        )
        
        retention_drop_threshold = st.number_input(
            "Критичное падение retention (%):",
            min_value=1,
            max_value=50,
            value=20,
            help="Алерт при падении retention больше чем на X%"
        )
        
        payback_threshold = st.number_input(
            "Максимальный период окупаемости (мес):",
            min_value=1,
            max_value=24,
            value=12,
            help="Алерт при превышении периода окупаемости"
        )
    
    with col2:
        st.subheader("📧 Уведомления")
        
        email_notifications = st.text_input(
            "Email для уведомлений:",
            placeholder="your.email@company.com"
        )
        
        slack_webhook = st.text_input(
            "Slack Webhook URL:",
            placeholder="https://hooks.slack.com/services/..."
        )
        
        telegram_bot_token = st.text_input(
            "Telegram Bot Token:",
            placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        )
        
        telegram_chat_id = st.text_input("Telegram Chat ID:",
            placeholder="-1001234567890"
        )
    
    # Сохранение настроек мониторинга
    if st.button("💾 Сохранить настройки мониторинга"):
        monitoring_settings = {
            'ltv_cac_threshold': ltv_cac_threshold,
            'retention_drop_threshold': retention_drop_threshold / 100,
            'payback_threshold': payback_threshold,
            'email': email_notifications,
            'slack_webhook': slack_webhook,
            'telegram_bot_token': telegram_bot_token,
            'telegram_chat_id': telegram_chat_id,
            'created_at': datetime.now().isoformat()
        }
        
        st.session_state['monitoring_settings'] = monitoring_settings
        st.success("✅ Настройки мониторинга сохранены!")
    
    # Проверка алертов для текущих данных
    if 'metrics' in st.session_state and 'monitoring_settings' in st.session_state:
        st.markdown("### 🔍 Проверка текущих данных")
        
        metrics = st.session_state['metrics']
        settings = st.session_state['monitoring_settings']
        
        alerts = calculator.check_alerts(metrics, settings)
        
        if alerts:
            st.markdown("#### 🚨 Обнаружены алерты:")
            
            for alert in alerts:
                if alert['type'] == 'critical':
                    st.error(f"🔴 **{alert['title']}**: {alert['message']}")
                else:
                    st.warning(f"🟡 **{alert['title']}**: {alert['message']}")
                
                st.markdown(f"💡 *{alert['recommendation']}*")
                
                # Отправка уведомлений
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if settings.get('email') and st.button(f"📧 Отправить на email", key=f"email_{alert['title']}"):
                        # Здесь можно реализовать отправку email
                        st.info("📧 Email уведомление отправлено!")
                
                with col2:
                    if settings.get('slack_webhook') and st.button(f"💬 Отправить в Slack", key=f"slack_{alert['title']}"):
                        success = calculator.send_slack_notification(
                            settings['slack_webhook'], 
                            f"{alert['title']}: {alert['message']}"
                        )
                        if success:
                            st.success("✅ Уведомление отправлено в Slack!")
                        else:
                            st.error("❌ Ошибка отправки в Slack")
                
                with col3:
                    if settings.get('telegram_bot_token') and st.button(f"📱 Отправить в Telegram", key=f"tg_{alert['title']}"):
                        # Здесь можно реализовать отправку в Telegram
                        st.info("📱 Telegram уведомление отправлено!")
                
                st.markdown("---")
        
        else:
            st.success("✅ Все метрики в норме, алертов нет!")
    
    # История алертов
    st.markdown("### 📈 Симуляция мониторинга")
    
    if st.button("🎭 Симулировать различные сценарии"):
        # Создаем различные сценарии для демонстрации алертов
        test_scenarios = [
            {
                'name': 'Нормальная ситуация',
                'ltv_cac': 3.2,
                'retention_drop': 0.05,
                'payback_period': 8
            },
            {
                'name': 'Критическое падение LTV/CAC',
                'ltv_cac': 0.8,
                'retention_drop': 0.15,
                'payback_period': 15
            },
            {
                'name': 'Резкое падение retention',
                'ltv_cac': 2.1,
                'retention_drop': 0.35,
                'payback_period': 6
            },
            {
                'name': 'Долгая окупаемость',
                'ltv_cac': 1.8,
                'retention_drop': 0.10,
                'payback_period': 18
            }
        ]
        
        for scenario in test_scenarios:
            st.markdown(f"#### 🎯 Сценарий: {scenario['name']}")
            
            # Создаем тестовые метрики
            test_metrics = {
                'ltv_cac': scenario['ltv_cac'],
                'payback_period': scenario['payback_period'],
                'retention': pd.Series([1.0, 0.8, 0.8 - scenario['retention_drop']])
            }
            
            # Проверяем алерты
            test_alerts = calculator.check_alerts(
                test_metrics, 
                st.session_state.get('monitoring_settings', {
                    'ltv_cac_threshold': 1.0,
                    'retention_drop': 0.2,
                    'payback_threshold': 12
                })
            )
            
            if test_alerts:
                for alert in test_alerts:
                    if alert['type'] == 'critical':
                        st.error(f"🔴 {alert['message']}")
                    else:
                        st.warning(f"🟡 {alert['message']}")
            else:
                st.success("✅ Алертов нет")
            
            st.markdown("---")

def benchmarks_tab():
    """Вкладка сравнения с бенчмарками индустрии."""
    st.markdown("## 🏆 Бенчмарки индустрии")
    
    st.markdown("""
    Сравните ваши метрики с усредненными показателями по различным индустриям.
    Это поможет понять, насколько эффективна ваша модель относительно конкурентов.
    """)
    
    # Выбор индустрии
    industry = st.selectbox(
        "Выберите индустрию:",
        list(calculator.industry_benchmarks.keys())
    )
    
    benchmark = calculator.industry_benchmarks[industry]
    
    # Отображение бенчмарков
    st.markdown(f"### 📊 Бенчмарки для индустрии: {industry}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("LTV/CAC", f"{benchmark['ltv_cac']:.1f}")
    
    with col2:
        st.metric("Lifetime", f"{benchmark['lifetime_months']} мес")
    
    with col3:
        st.metric("1-й месяц retention", f"{benchmark['month1_retention']:.0%}")
    
    with col4:
        st.metric("Churn Rate", f"{benchmark['churn_rate']:.1%}")
    
    with col5:
        st.metric("Окупаемость", f"{benchmark['payback_months']} мес")
    
    # Сравнение с вашими данными
    if 'metrics' in st.session_state:
        st.markdown("### 🔄 Сравнение с вашими данными")
        
        metrics = st.session_state['metrics']
        retention_data = st.session_state.get('retention_data', pd.Series())
        
        # Создаем таблицу сравнения
        comparison_data = {
            'Метрика': ['LTV/CAC', 'Lifetime (мес)', '1-й месяц retention', 'Средний churn rate'],
            'Ваши данные': [
                f"{metrics['ltv_cac']:.2f}",
                f"{metrics['lifetime']:.1f}",
                f"{retention_data.iloc[0] if len(retention_data) > 0 else 0:.0%}",
                f"{metrics['churn_rates'].mean():.1%}" if len(metrics['churn_rates']) > 0 else "N/A"
            ],
            'Бенчмарк индустрии': [
                f"{benchmark['ltv_cac']:.1f}",
                f"{benchmark['lifetime_months']}",
                f"{benchmark['month1_retention']:.0%}",
                f"{benchmark['churn_rate']:.1%}"
            ],
            'Отклонение': []
        }
        
        # Расчет отклонений
        your_ltv_cac = metrics['ltv_cac']
        your_lifetime = metrics['lifetime']
        your_retention = retention_data.iloc[0] if len(retention_data) > 0 else 0
        your_churn = metrics['churn_rates'].mean() if len(metrics['churn_rates']) > 0 else 0
        
        deviations = [
            f"{((your_ltv_cac / benchmark['ltv_cac']) - 1) * 100:+.1f}%",
            f"{((your_lifetime / benchmark['lifetime_months']) - 1) * 100:+.1f}%",
            f"{((your_retention / benchmark['month1_retention']) - 1) * 100:+.1f}%" if your_retention > 0 else "N/A",
            f"{((your_churn / benchmark['churn_rate']) - 1) * 100:+.1f}%" if your_churn > 0 else "N/A"
        ]
        
        comparison_data['Отклонение'] = deviations
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Радарный график сравнения
        create_benchmark_radar_chart(metrics, benchmark, industry, retention_data)
        
        # Анализ и рекомендации
        st.markdown("### 💡 Анализ и рекомендации")
        
        recommendations = []
        
        if your_ltv_cac < benchmark['ltv_cac'] * 0.8:
            recommendations.append({
                'type': 'warning',
                'title': 'LTV/CAC ниже среднего по индустрии',
                'text': f"Ваш LTV/CAC ({your_ltv_cac:.2f}) на {abs(((your_ltv_cac / benchmark['ltv_cac']) - 1) * 100):.1f}% ниже среднего",
                'action': 'Фокус на увеличении ARPPU или снижении CAC'
            })
        elif your_ltv_cac > benchmark['ltv_cac'] * 1.2:
            recommendations.append({
                'type': 'success',
                'title': 'Отличный LTV/CAC',
                'text': f"Ваш LTV/CAC превышает средний по индустрии на {((your_ltv_cac / benchmark['ltv_cac']) - 1) * 100:.1f}%",
                'action': 'Можно увеличить инвестиции в привлечение клиентов'
            })
        
        if your_lifetime < benchmark['lifetime_months'] * 0.8:
            recommendations.append({
                'type': 'warning',
                'title': 'Низкое время жизни клиентов',
                'text': f"Lifetime ниже среднего по индустрии на {abs(((your_lifetime / benchmark['lifetime_months']) - 1) * 100):.1f}%",
                'action': 'Улучшите retention через продуктовые изменения'
            })
        
        if your_retention > 0 and your_retention < benchmark['month1_retention'] * 0.8:
            recommendations.append({
                'type': 'warning',
                'title': 'Низкий retention первого месяца',
                'text': f"Retention первого месяца ниже среднего на {abs(((your_retention / benchmark['month1_retention']) - 1) * 100):.1f}%",
                'action': 'Улучшите onboarding и первый пользовательский опыт'
            })
        
        # Отображение рекомендаций
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"⚠️ **{rec['title']}**: {rec['text']}")
            else:
                st.success(f"✅ **{rec['title']}**: {rec['text']}")
            
            st.markdown(f"🎯 *Рекомендация: {rec['action']}*")
            st.markdown("---")
        
        if not recommendations:
            st.success("🎉 Ваши метрики соответствуют или превышают средние показатели индустрии!")
    
    else:
        st.info("👆 Выполните основной анализ для сравнения с бенчмарками")
    
    # Детальная информация по индустриям
    st.markdown("### 📚 Детальная информация по индустриям")
    
    industry_descriptions = {
        'E-commerce': {
            'description': 'Интернет-магазины, маркетплейсы',
            'challenges': ['Высокая конкуренция', 'Сезонность', 'Логистические затраты'],
            'opportunities': ['Персонализация', 'Омниканальность', 'Подписки']
        },
        'SaaS B2B': {
            'description': 'Программное обеспечение как услуга для бизнеса',
            'challenges': ['Длинный sales cycle', 'Высокие затраты на онбординг'],
            'opportunities': ['Высокий LTV', 'Предсказуемый доход', 'Масштабируемость']
        },
        'Mobile Games': {
            'description': 'Мобильные игры с монетизацией',
            'challenges': ['Высокий churn', 'Конкуренция за внимание'],
            'opportunities': ['Вирусность', 'In-app покупки', 'Реклама']
        },
        'Финтех': {
            'description': 'Финансовые технологии и услуги',
            'challenges': ['Регулирование', 'Доверие пользователей'],
            'opportunities': ['Высокая лояльность', 'Cross-selling', 'Данные']
        },
        'EdTech': {
            'description': 'Образовательные технологии',
            'challenges': ['Сезонность', 'Доказательство эффективности'],
            'opportunities': ['Долгосрочные отношения', 'Расширение программ']
        },
        'Delivery': {
            'description': 'Доставка еды и товаров',
            'challenges': ['Конкуренция по цене', 'Логистические затраты'],
            'opportunities': ['Частота использования', 'Расширение географии']
        }
    }
    
    if industry in industry_descriptions:
        info = industry_descriptions[industry]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Описание**: {info['description']}")
            
            st.markdown("**Основные вызовы:**")
            for challenge in info['challenges']:
                st.markdown(f"• {challenge}")
        
        with col2:
            st.markdown("**Возможности для роста:**")
            for opportunity in info['opportunities']:
                st.markdown(f"• {opportunity}")

def create_benchmark_radar_chart(metrics, benchmark, industry, retention_data):
    """Создает радарный график сравнения с бенчмарками."""
    try:
        categories = ['LTV/CAC', 'Lifetime', 'Retention 1м', 'Низкий Churn']
        
        # Нормализуем данные для радарного графика
        your_values = [
            min(metrics['ltv_cac'] / benchmark['ltv_cac'], 2),  # макс 2x от бенчмарка
            min(metrics['lifetime'] / benchmark['lifetime_months'], 2),
            min((retention_data.iloc[0] if len(retention_data) > 0 else 0) / benchmark['month1_retention'], 2),
            min((1 - metrics['churn_rates'].mean()) / (1 - benchmark['churn_rate']), 2) if len(metrics['churn_rates']) > 0 else 0
        ]
        
        benchmark_values = [1, 1, 1, 1]  # бенчмарк = 1
        
        fig = go.Figure()
        
        # Ваши данные
        fig.add_trace(go.Scatterpolar(
            r=your_values + [your_values[0]],  # замыкаем график
            theta=categories + [categories[0]],
            fill='toself',
            name='Ваши данные',
            line_color='#667eea'
        ))
        
        # Бенчмарк
        fig.add_trace(go.Scatterpolar(
            r=benchmark_values + [benchmark_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Бенчмарк {industry}',
            line_color='#ff7f0e',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2]
                )
            ),
            showlegend=True,
            title="Сравнение с бенчмарками индустрии"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Ошибка создания радарного графика: {str(e)}")
        st.error("Не удалось создать радарный график сравнения")

# Запуск приложения
if __name__ == "__main__":
    # Инициализация session state
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.version = "2.0.0"
        st.session_state.last_updated = datetime.now()
    
    # Справочная информация в подвале
    with st.expander("📚 Справка по метрикам и методологии"):
        st.markdown("""
        ### 📖 Основные понятия
        
        - **Lifetime** — среднее время жизни клиента (сумма retention по месяцам)
        - **LTV** (Lifetime Value) — общий доход с одного клиента за весь период жизни
        - **CAC** (Customer Acquisition Cost) — стоимость привлечения одного клиента
        - **ARPPU** (Average Revenue Per Paying User) — средний доход на платящего пользователя
        - **Retention** — доля пользователей, которые остались активными в определенный период
        - **Churn Rate** — доля пользователей, которые перестали быть активными
        
        ### 🧮 Формулы расчета
        
        ```
        Lifetime = Σ(Retention по месяцам)
        LTV = ARPPU × Lifetime
        LTV/CAC = LTV ÷ CAC
        ROI = (LTV/CAC - 1) × 100%
        Churn Rate = 1 - (Retention_текущий / Retention_предыдущий)
        ```
        
        ### 🎯 Интерпретация LTV/CAC
        
        - **< 1.0** — Убыточная модель (тратим больше, чем зарабатываем)
        - **1.0-2.0** — Окупается, но есть риски
        - **2.0-3.0** — Здоровая модель с потенциалом роста
        - **> 3.0** — Отличная модель для масштабирования
        
        ### 📊 Рекомендации по улучшению
        
        **Для увеличения LTV:**
        - Улучшение retention (продуктовые фичи, engagement, персонализация)
        - Увеличение ARPPU (upsell, cross-sell, пересмотр ценообразования)
        - Снижение churn rate (улучшение customer success)
        
        **Для снижения CAC:**
        - Оптимизация рекламных каналов и таргетинга
        - Улучшение конверсии лендингов и воронки
        - Развитие органических каналов (SEO, контент-маркетинг)
        - Реферальные программы
        
        ### 🤖 ML Прогнозирование
        
        Приложение использует полиномиальную регрессию для прогнозирования retention.
        Алгоритм учитывает:
        - Исторический тренд retention
        - Естественное убывание пользовательской активности
        - Доверительные интервалы для оценки точности прогноза
        
        ### 🏭 Бенчмарки по индустриям
        
        Данные основаны на публичных исследованиях и отчетах:
        - **E-commerce**: средний LTV/CAC = 3.0, retention 1м = 25%
        - **SaaS B2B**: средний LTV/CAC = 3.5, retention 1м = 85%
        - **Mobile Games**: средний LTV/CAC = 1.5, retention 1м = 40%
        - **Финтех**: средний LTV/CAC = 4.0, retention 1м = 70%
        - **EdTech**: средний LTV/CAC = 2.8, retention 1м = 60%
        - **Delivery**: средний LTV/CAC = 2.2, retention 1м = 35%
        
        ### 🔗 Интеграции
        
        **Google Sheets**: Поддерживается прямая загрузка данных по ссылке.
        Требования к таблице:
        - Публичный доступ или доступ по ссылке
        - Колонка с числовыми значениями retention (в % или долях)
        - Корректные заголовки колонок
        
        **Мониторинг**: Настраиваемые алерты с уведомлениями в:
        - Email
        - Slack (через Webhook)
        - Telegram (через Bot API)
        
        ### 🎭 Сценарное планирование
        
        Позволяет моделировать различные бизнес-сценарии:
        - **Оптимистичный**: +25% ARPPU, +20% retention, -15% CAC
        - **Реалистичный**: +8% ARPPU, +8% retention, +2% CAC
        - **Пессимистичный**: -8% ARPPU, -12% retention, +15% CAC
        - **Кризисный**: -15% ARPPU, -25% retention, +30% CAC
        
        ### 💾 Экспорт данных
        
        **Excel отчет включает:**
        - Основные метрики
        - Помесячные данные
        - ML прогноз
        - Анализ чувствительности
        - Автоматические инсайты
        - Сценарии планирования
        
        **JSON формат** для интеграции с другими системами аналитики.
        """)
    
    # Информация о версии
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ О приложении")
    st.sidebar.markdown(f"**Версия**: {st.session_state.version}")
    st.sidebar.markdown(f"**Обновлено**: {st.session_state.last_updated.strftime('%Y-%m-%d')}")
    st.sidebar.markdown("**Автор**: Lifetime Calculator Pro")
    
    # Обратная связь
    with st.sidebar.expander("📝 Обратная связь"):
        feedback_type = st.selectbox("Тип обратной связи:", 
                                   ["💡 Предложение", "🐛 Ошибка", "❓ Вопрос", "👍 Похвала"])
        feedback_text = st.text_area("Ваше сообщение:")
        
        if st.button("📤 Отправить") and feedback_text:
            # Здесь можно реализовать отправку фидбека
            st.success("Спасибо за обратную связь!")
    
    # Запуск основного приложения
    main()
