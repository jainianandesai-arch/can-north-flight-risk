# ══════════════════════════════════════════════════════════════════════════
# CAN NORTH FINANCIAL — FLIGHT RISK INTELLIGENCE TOOL
# app.py
#
# DEMO VERSION — Trained on synthetic data
# Users download sample data, tweak it, upload it, see predictions change
# ══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Flight Risk Intelligence — CAN North Financial',
    page_icon='🏦',
    layout='wide'
)

# ── LOAD MODEL ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('can_north_model.pkl')
    scaler   = joblib.load('can_north_scaler.pkl')
    features = joblib.load('can_north_features.pkl')
    return model, scaler, features

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────
def prepare_features(df, feature_names):
    df = df.copy()
    df['persona_direction_encoded'] = df['persona_direction'].map(
        {'More flexible': 0, 'No change': 1, 'More office': 2})
    df['role_level_encoded'] = df['role_level'].map(
        {'Analyst': 1, 'Senior Analyst': 2, 'Manager': 3,
         'Director': 4, 'VP': 5})
    div_dummies = pd.get_dummies(df['division'], prefix='div')
    loc_dummies = pd.get_dummies(df['location'], prefix='loc')
    selected = [
        'engagement_2025', 'satisfaction_2025', 'career_growth_2025',
        'wellbeing_2025', 'recognition_2025', 'mgr_effectiveness_2025',
        'belonging_2025', 'performance_2025', 'salary_vs_market_2025',
        'absence_days_2025',
        'engagement_trend', 'satisfaction_trend', 'career_growth_trend',
        'performance_trend', 'absence_trend', 'salary_trend',
        'rto_risk_index', 'commute_time_change_min', 'org_disruption_score',
        'manager_stability', 'persona_direction_encoded', 'transit_dependent',
        'tenure_years', 'age', 'promotions_3yr',
        'role_level_encoded', 'commute_km',
    ]
    X = pd.concat([df[selected], div_dummies, loc_dummies], axis=1)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X

# ── SCORE EMPLOYEES ───────────────────────────────────────────────────────
def score_employees(df, model, scaler, feature_names):
    X        = prepare_features(df, feature_names)
    X_scaled = scaler.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df['flight_risk_score'] = (probs * 100).round(1)
    df['risk_category']     = pd.cut(
        probs,
        bins=[0, 0.30, 0.60, 0.80, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical'])
    return df

# ── INTERVENTION LOGIC ────────────────────────────────────────────────────
def primary_driver(row):
    if row['engagement_2025'] < 35:           return 'Critical engagement'
    elif row['engagement_trend'] < -20:       return 'Engagement freefall'
    elif row['satisfaction_2025'] < 35:       return 'Low satisfaction'
    elif row['rto_risk_index'] > 60:          return 'RTO impact'
    elif row['salary_vs_market_2025'] < 40:   return 'Below market pay'
    elif row['manager_stability'] < 35:       return 'Manager instability'
    else:                                      return 'Multiple factors'

def recommend_action(row):
    if row['rto_risk_index'] > 55 and row['engagement_trend'] < -15:
        return 'Offer flexibility + career conversation'
    elif row['salary_vs_market_2025'] < 40:
        return 'Compensation review'
    elif row['manager_stability'] < 35:
        return 'Manager change or skip-level meeting'
    elif row['engagement_2025'] < 35:
        return 'Urgent 1:1 with HR business partner'
    elif row['career_growth_2025'] < 40:
        return 'Career development conversation'
    else:
        return 'Stay interview this month'

# ── VALIDATE UPLOAD ───────────────────────────────────────────────────────
def validate_upload(df):
    required = [
        'employee_id', 'division', 'role_level', 'location',
        'engagement_2025', 'satisfaction_2025', 'engagement_trend',
        'rto_risk_index', 'persona_direction'
    ]
    return [c for c in required if c not in df.columns]

# ── GENERATE SUMMARY REPORT ───────────────────────────────────────────────
def generate_summary(df_scored):
    high_risk = df_scored[
        df_scored['risk_category'].isin(['Critical', 'High'])]
    stable    = df_scored[df_scored['risk_category'] == 'Low']
    at_risk   = len(high_risk)
    cost      = at_risk * 78000
    retained  = int(at_risk * 0.6)
    saved     = retained * 78000
    int_cost  = at_risk * 500 + (at_risk // 3) * 2000

    lines = []
    lines.append('=' * 65)
    lines.append('FLIGHT RISK INTELLIGENCE — EXPLORATORY SUMMARY REPORT')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('Synthetic data — demonstration purposes only')
    lines.append('=' * 65)

    lines.append('\nDATASET OVERVIEW')
    lines.append('-' * 40)
    lines.append(f'Total employees scored:   {len(df_scored):,}')
    lines.append(f'Critical risk:            '
                 f'{(df_scored["risk_category"]=="Critical").sum()}')
    lines.append(f'High risk:                '
                 f'{(df_scored["risk_category"]=="High").sum()}')
    lines.append(f'Medium risk:              '
                 f'{(df_scored["risk_category"]=="Medium").sum()}')
    lines.append(f'Low / Stable:             '
                 f'{(df_scored["risk_category"]=="Low").sum()}')
    lines.append(f'Total at risk:            {at_risk}')
    lines.append(f'Est. cost if all leave:   ${cost:,.0f}')

    lines.append('\nKEY SIGNALS — AT RISK VS STABLE')
    lines.append('-' * 40)
    lines.append(f'{"Signal":<30} {"At Risk":>10} {"Stable":>10} '
                 f'{"Gap":>10}')
    lines.append(f'{"-"*30} {"-"*10} {"-"*10} {"-"*10}')

    signals = [
        ('engagement_2025',       'Engagement 2025'),
        ('satisfaction_2025',     'Satisfaction 2025'),
        ('career_growth_2025',    'Career growth 2025'),
        ('engagement_trend',      'Engagement trend'),
        ('satisfaction_trend',    'Satisfaction trend'),
        ('rto_risk_index',        'RTO risk index'),
        ('org_disruption_score',  'Org disruption'),
        ('salary_vs_market_2025', 'Salary vs market'),
        ('manager_stability',     'Manager stability'),
    ]

    for col, label in signals:
        if col in df_scored.columns:
            hr = high_risk[col].mean() if len(high_risk) > 0 else 0
            st = stable[col].mean()    if len(stable)    > 0 else 0
            lines.append(
                f'{label:<30} {hr:>10.1f} {st:>10.1f} {st-hr:>+10.1f}')

    lines.append('\nRISK BY DIVISION')
    lines.append('-' * 40)
    for div in sorted(df_scored['division'].unique()):
        grp  = df_scored[df_scored['division'] == div]
        rate = grp['risk_category'].isin(
            ['Critical','High']).mean() * 100
        lines.append(f'  {div:<30} {rate:.1f}% at risk '
                     f'({len(grp)} employees)')

    lines.append('\nRISK BY ROLE LEVEL')
    lines.append('-' * 40)
    for role in ['Analyst','Senior Analyst','Manager','Director','VP']:
        grp = df_scored[df_scored['role_level'] == role]
        if len(grp) > 0:
            rate = grp['risk_category'].isin(
                ['Critical','High']).mean() * 100
            lines.append(f'  {role:<20} {rate:.1f}% at risk '
                         f'({len(grp)} employees)')

    lines.append('\nRTO IMPACT')
    lines.append('-' * 40)
    for persona in ['More office', 'No change', 'More flexible']:
        grp = df_scored[df_scored['persona_direction'] == persona]
        if len(grp) > 0:
            rate = grp['risk_category'].isin(
                ['Critical','High']).mean() * 100
            lines.append(f'  {persona:<20} {rate:.1f}% at risk '
                         f'({len(grp)} employees)')

    lines.append('\nCRITICAL ZONE')
    lines.append('-' * 40)
    critical_zone = df_scored[
        (df_scored['engagement_2025'] < 40) &
        (df_scored['career_growth_2025'] < 40) &
        (df_scored['rto_risk_index'] > 60)
    ]
    lines.append(
        f'Employees with eng<40, career<40, RTO>60: '
        f'{len(critical_zone)}')

    lines.append('\nINTERVENTION ROI')
    lines.append('-' * 40)
    lines.append(f'Cost if all {at_risk} at-risk leave: ${cost:,.0f}')
    lines.append(f'Intervention cost estimate:          ${int_cost:,.0f}')
    lines.append(f'Retain {retained} (60%):             ${saved:,.0f} saved')
    lines.append(f'Net saving:                          '
                 f'${saved - int_cost:,.0f}')
    lines.append(f'ROI:                                 '
                 f'{saved // max(int_cost, 1)}x')

    lines.append('\n' + '=' * 65)
    lines.append('IMPORTANT DISCLAIMER')
    lines.append('=' * 65)
    lines.append(
        'All data in this report is synthetically generated.')
    lines.append(
        'CAN North Financial is a fictional organization.')
    lines.append(
        'This report is for demonstration purposes only.')
    lines.append(
        'Methodology and source code are proprietary.')
    lines.append('=' * 65)

    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main():

    # ── HEADER ────────────────────────────────────────────────────────────
    st.title('🏦 CAN North Financial')
    st.subheader('Employee Flight Risk Intelligence System')
    st.markdown(
        '*People Analytics · Workforce Intelligence · Predictive HR*')
    st.divider()

    # ── CONTEXT BANNER ────────────────────────────────────────────────────
    st.info("""
    **DEMONSTRATION VERSION — Synthetic Data**

    This tool is trained on synthetically generated data representing
    CAN North Financial — a fictional Canadian financial services
    organization. All employee records, scores, and outcomes are
    computationally generated. No real employee data was used.

    **How to use this demo:**
    Download the sample dataset below → open it in Excel →
    tweak any values → save as CSV → upload it here →
    see how the model responds in real time.

    **Try this:** Lower someone's engagement_2025 to 15,
    set their engagement_trend to -40, set rto_risk_index to 85.
    Watch their score jump to Critical.

    **Note:** If you upload your own organization's data the tool
    will run but predictions reflect CAN North's learned patterns —
    not your organization's. Meaningful deployment requires
    retraining on your own data.
    """)

    # ── LOAD MODEL ────────────────────────────────────────────────────────
    try:
        model, scaler, feature_names = load_model()
        st.success('Model loaded — ready to score employees')
    except Exception:
        st.error(
            'Model files not found. Run train_pipeline.py first.')
        st.stop()

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('About this tool')
        st.markdown("""
        **What it does:**
        Predicts which employees are most likely
        to leave in the next 90 days.

        **Signals it uses:**
        - Engagement and satisfaction trends
        - Return-to-office impact
        - Manager stability
        - Career growth confidence
        - Organizational disruption
        - Salary vs market position

        **Model:**
        Logistic Regression
        Winner of a 5-model race
        98% recall on held-out test data

        **Primary metric:**
        Recall — catching real quitters
        before they hand in their notice

        **Data:**
        Synthetic — demonstration only
        """)
        st.divider()
        st.markdown('**How to explore:**')
        st.markdown("""
        1. Download the sample CSV
        2. Open in Excel
        3. Change any values:
           - Lower engagement → higher risk
           - Raise RTO index → higher risk
           - Lower satisfaction → higher risk
        4. Save as CSV
        5. Upload and run
        6. See predictions change in real time
        """)
        st.divider()
        st.caption(
            f'Run date: {datetime.now().strftime("%Y-%m-%d")}')
        st.caption('Methodology and source code proprietary')

    # ── SAMPLE DATA DOWNLOAD ──────────────────────────────────────────────
    st.header('Step 1 — Download the sample dataset')
    st.markdown("""
    This CSV represents 200 current employees at CAN North Financial
    with 3 years of engagement, performance, commute, and
    organizational data. Download it, open in Excel, tweak any
    values, save, and upload below.
    """)

    if os.path.exists('can_north_q2_2026.csv'):
        sample_df = pd.read_csv('can_north_q2_2026.csv')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Employees', f'{len(sample_df):,}')
        with col2:
            st.metric('Features', f'{sample_df.shape[1]}')
        with col3:
            st.metric('Period', '2023 – 2025')

        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='⬇ Download Sample Dataset (CSV)',
            data=sample_csv,
            file_name='can_north_sample_employees.csv',
            mime='text/csv',
            help='Download, tweak in Excel, upload below'
        )

        with st.expander('Preview — first 5 employees'):
            st.dataframe(
                sample_df[[
                    'employee_id', 'division', 'role_level',
                    'location', 'engagement_2025', 'satisfaction_2025',
                    'engagement_trend', 'rto_risk_index'
                ]].head(5),
                use_container_width=True)
    else:
        st.warning(
            'Sample file not found. '
            'Run create_sample_data.py to generate it.')

    st.divider()

    # ── FILE UPLOAD ───────────────────────────────────────────────────────
    st.header('Step 2 — Upload your employee data')
    st.markdown(
        'Upload the sample file as-is or your modified version. '
        'The model scores every employee instantly.')

    uploaded_file = st.file_uploader(
        'Choose your CSV file',
        type='csv',
        help='Upload the sample file or your modified version')

    if uploaded_file is None:
        st.info('Waiting for file upload...')
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.success(f'File uploaded — {len(df):,} employees loaded')

    missing_cols = validate_upload(df)
    if missing_cols:
        st.warning(
            f'Some expected columns are missing: {missing_cols}. '
            f'For best results use the sample dataset provided above. '
            f'Results may not be meaningful with this file.')

    if 'left' in df.columns:
        df = df.drop(columns=['left'])
        st.info('Outcome column removed — running in scoring mode')

    with st.expander('Preview uploaded data'):
        st.dataframe(
            df[[
                'employee_id', 'division', 'role_level', 'location',
                'engagement_2025', 'satisfaction_2025',
                'engagement_trend', 'rto_risk_index'
            ]].head(10),
            use_container_width=True)

    st.divider()

    # ── RUN MODEL ─────────────────────────────────────────────────────────
    st.header('Step 3 — Run the model')
    st.markdown(
        'The model applies what it learned from CAN North\'s '
        'historical attrition patterns to your uploaded data.')

    if not st.button('Run Flight Risk Model', type='primary'):
        st.stop()

    with st.spinner('Scoring all employees...'):
        df_scored = score_employees(df, model, scaler, feature_names)

    st.success(f'All {len(df_scored):,} employees scored')

    # ── Pre-calculate everything ───────────────────────────────────────────
    critical  = (df_scored['risk_category'] == 'Critical').sum()
    high      = (df_scored['risk_category'] == 'High').sum()
    medium    = (df_scored['risk_category'] == 'Medium').sum()
    low       = (df_scored['risk_category'] == 'Low').sum()
    at_risk   = critical + high
    cost      = at_risk * 78000
    retained  = int(at_risk * 0.6)
    saved     = retained * 78000
    int_cost  = at_risk * 500 + (at_risk // 3) * 2000

    high_risk_df = df_scored[
        df_scored['risk_category'].isin(['Critical', 'High'])].copy()
    stable_df    = df_scored[
        df_scored['risk_category'] == 'Low'].copy()

    if at_risk > 0:
        high_risk_df['Primary Driver']     = high_risk_df.apply(
            primary_driver, axis=1)
        high_risk_df['Recommended Action'] = high_risk_df.apply(
            recommend_action, axis=1)
        top25 = high_risk_df.nlargest(
            25, 'flight_risk_score')[[
            'employee_id', 'division', 'role_level', 'location',
            'engagement_2025', 'satisfaction_2025',
            'engagement_trend', 'rto_risk_index',
            'flight_risk_score', 'risk_category',
            'Primary Driver', 'Recommended Action'
        ]].reset_index(drop=True)
        top25.index = top25.index + 1
    else:
        top25 = pd.DataFrame()

    critical_zone = df_scored[
        (df_scored['engagement_2025'] < 40) &
        (df_scored['career_growth_2025'] < 40) &
        (df_scored['rto_risk_index'] > 60)
    ]

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════

    tab1, tab2, tab3, tab4 = st.tabs([
        '📊 Data Exploration',
        '🎯 Risk Scores',
        '💡 Key Insights',
        '📥 Download Reports'
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA EXPLORATION
    # ══════════════════════════════════════════════════════════════════════

    with tab1:
        st.subheader('What your data is already telling you')
        st.markdown(
            '*Before the model runs — the data speaks for itself*')

        # Dataset overview
        st.markdown('#### Dataset Overview')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total employees', f'{len(df_scored):,}')
        with col2:
            st.metric('Features per employee', f'{df.shape[1]}')
        with col3:
            st.metric('Data period', '2023 → 2025')
        with col4:
            st.metric('At risk', f'{at_risk}',
                      delta='High + Critical',
                      delta_color='inverse')

        st.divider()

        # Engagement distribution
        st.markdown('#### Engagement Score Distribution')
        st.markdown(
            'Engagement is the strongest predictor of flight risk. '
            'Employees below 40 are in the danger zone.')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                'Average engagement',
                f"{df_scored['engagement_2025'].mean():.1f}/100")
        with col2:
            st.metric(
                'Lowest engagement',
                f"{df_scored['engagement_2025'].min():.1f}/100",
                delta='Highest risk', delta_color='inverse')
        with col3:
            st.metric(
                'Highest engagement',
                f"{df_scored['engagement_2025'].max():.1f}/100",
                delta='Most stable')
        with col4:
            danger = (df_scored['engagement_2025'] < 40).sum()
            st.metric(
                'Below 40 — danger zone',
                f'{danger} employees',
                delta='Immediate concern',
                delta_color='inverse')

        st.divider()

        # Key signals table
        st.markdown('#### Key Signals — At Risk vs Stable')
        st.markdown(
            'How much do at-risk and stable employees differ '
            'on each signal? Bigger gap = stronger predictor.')

        signals = [
            ('engagement_2025',       'Engagement 2025',        '/100'),
            ('satisfaction_2025',     'Satisfaction 2025',      '/100'),
            ('career_growth_2025',    'Career growth 2025',     '/100'),
            ('wellbeing_2025',        'Wellbeing 2025',         '/100'),
            ('engagement_trend',      'Engagement trend (3yr)', ' pts'),
            ('satisfaction_trend',    'Satisfaction trend (3yr)',' pts'),
            ('rto_risk_index',        'RTO risk index',         '/100'),
            ('org_disruption_score',  'Org disruption',         '/100'),
            ('salary_vs_market_2025', 'Salary vs market',       '/100'),
            ('manager_stability',     'Manager stability',      '/100'),
        ]

        rows = []
        for col, label, unit in signals:
            if col in df_scored.columns:
                hr = high_risk_df[col].mean() \
                     if len(high_risk_df) > 0 else 0
                st_v = stable_df[col].mean() \
                       if len(stable_df)    > 0 else 0
                gap  = st_v - hr
                rows.append({
                    'Signal':        label,
                    'At Risk (avg)': f"{hr:.1f}{unit}",
                    'Stable (avg)':  f"{st_v:.1f}{unit}",
                    'Gap':           f"{gap:+.1f}{unit}",
                    'Strength':      (
                        'Strong'   if abs(gap) > 20 else
                        'Moderate' if abs(gap) > 10 else
                        'Weak'
                    )
                })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True)

        st.divider()

        # Segment breakdown
        st.markdown('#### Risk Rate by Segment')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('**By Division**')
            rows = []
            for div in sorted(df_scored['division'].unique()):
                grp  = df_scored[df_scored['division'] == div]
                rate = grp['risk_category'].isin(
                    ['Critical','High']).mean() * 100
                rows.append({
                    'Division': div,
                    'At Risk':  f'{rate:.1f}%',
                    'Count':    len(grp)
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True)

        with col2:
            st.markdown('**By Role Level**')
            rows = []
            for role in ['Analyst','Senior Analyst',
                         'Manager','Director','VP']:
                grp = df_scored[df_scored['role_level'] == role]
                if len(grp) > 0:
                    rate = grp['risk_category'].isin(
                        ['Critical','High']).mean() * 100
                    rows.append({
                        'Role':    role,
                        'At Risk': f'{rate:.1f}%',
                        'Count':   len(grp)
                    })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True)

        with col3:
            st.markdown('**By Work Persona**')
            rows = []
            for persona in ['More office','No change','More flexible']:
                grp = df_scored[
                    df_scored['persona_direction'] == persona]
                if len(grp) > 0:
                    rate = grp['risk_category'].isin(
                        ['Critical','High']).mean() * 100
                    rows.append({
                        'Persona':  persona,
                        'At Risk':  f'{rate:.1f}%',
                        'Count':    len(grp)
                    })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True)

        st.divider()

        # Critical zone
        st.markdown('#### Critical Zone')
        st.markdown(
            'Employees with engagement below 40, career growth '
            'below 40, and RTO risk above 60. '
            'Contact these employees immediately — '
            'regardless of any other signal.')

        if len(critical_zone) == 0:
            st.success('No employees in the critical zone.')
        else:
            st.error(
                f'{len(critical_zone)} employees in the critical zone '
                f'— contact them this week.')
            st.dataframe(
                critical_zone[[
                    'employee_id', 'division', 'role_level',
                    'location', 'engagement_2025',
                    'career_growth_2025', 'rto_risk_index',
                    'flight_risk_score'
                ]].sort_values(
                    'flight_risk_score', ascending=False),
                use_container_width=True,
                hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — RISK SCORES
    # ══════════════════════════════════════════════════════════════════════

    with tab2:
        st.subheader('Flight Risk Scores')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Critical', critical,
                      delta='Immediate action',
                      delta_color='inverse')
        with col2:
            st.metric('High', high,
                      delta='This quarter',
                      delta_color='inverse')
        with col3:
            st.metric('Medium', medium,
                      delta='Monitor monthly')
        with col4:
            st.metric('Low — Stable', low)

        if at_risk > 0:
            st.error(
                f'**{at_risk} employees at risk** — '
                f'estimated replacement cost if all leave: '
                f'**${cost:,.0f}**')

        st.divider()

        st.markdown('#### Top 25 — Action Required This Week')
        st.markdown(
            'Ranked by flight risk score. Each row shows '
            'the primary driver and recommended HR action.')

        if at_risk == 0:
            st.success('No employees flagged as High or Critical.')
        else:
            st.dataframe(
                top25,
                use_container_width=True,
                height=500)

        st.divider()

        st.markdown('#### All Employees — Full Scored List')
        st.dataframe(
            df_scored[[
                'employee_id', 'division', 'role_level', 'location',
                'engagement_2025', 'satisfaction_2025',
                'flight_risk_score', 'risk_category'
            ]].sort_values('flight_risk_score', ascending=False),
            use_container_width=True,
            height=400)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — KEY INSIGHTS
    # ══════════════════════════════════════════════════════════════════════

    with tab3:
        st.subheader('Key Insights for Leadership')
        st.markdown(
            '*What the data and model are telling Diana — '
            'in plain English*')

        # RTO insight
        st.markdown('#### The RTO Effect')
        st.markdown(
            'Employees pushed more into office are leaving at '
            'significantly higher rates than those given flexibility. '
            'This is the most actionable finding in this dataset.')

        rto_rows = []
        for persona in ['More office', 'No change', 'More flexible']:
            grp = df_scored[df_scored['persona_direction'] == persona]
            if len(grp) > 0:
                rate = grp['risk_category'].isin(
                    ['Critical','High']).mean() * 100
                avg_eng = grp['engagement_2025'].mean()
                rto_rows.append({
                    'Work Persona':     persona,
                    'At Risk':          f'{rate:.1f}%',
                    'Avg Engagement':   f'{avg_eng:.1f}/100',
                    'Employees':        len(grp),
                    'Interpretation': (
                        'Highest risk — RTO is driving attrition'
                        if persona == 'More office' else
                        'Stable — flexibility is protective'
                        if persona == 'More flexible' else
                        'Moderate — no change in arrangement'
                    )
                })

        st.dataframe(
            pd.DataFrame(rto_rows),
            use_container_width=True,
            hide_index=True)

        st.divider()

        # Engagement trend insight
        st.markdown('#### The Engagement Trend Story')
        st.markdown(
            'The direction of engagement change over 3 years '
            'is nearly as predictive as the current score. '
            'Someone at 50 today who was at 80 two years ago '
            'is more at risk than someone who has always been at 50.')

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                'Avg engagement trend — at risk',
                f"{high_risk_df['engagement_trend'].mean():.1f} pts"
                if len(high_risk_df) > 0 else 'N/A',
                delta='Over 3 years',
                delta_color='inverse')
        with col2:
            st.metric(
                'Avg engagement trend — stable',
                f"{stable_df['engagement_trend'].mean():.1f} pts"
                if len(stable_df) > 0 else 'N/A',
                delta='Over 3 years')

        st.divider()

        # Three recommendations
        st.markdown('#### Three Recommendations for the CEO')

        st.markdown("""
        **1. Flexibility Policy Review** *(Priority: Urgent)*

        Employees pushed more into office leave at significantly
        higher rates than those given flexibility.
        A targeted hybrid policy for high-risk employees could
        reduce attrition by an estimated 30–40%.
        Cost: policy change. Savings potential: significant.

        ---

        **2. Stay Interview Program** *(Priority: High)*

        Launch structured stay interviews for all Critical and
        High risk employees this quarter.
        HR business partners to lead. Focus questions:
        what would make you stay? what has changed in the last year?

        ---

        **3. Quarterly Flight Risk Scoring** *(Priority: Ongoing)*

        Run this model every quarter using refreshed engagement
        survey data. Track movement between risk categories.
        Measure whether interventions are working by monitoring
        score changes quarter over quarter.
        """)

        st.divider()

        # Intervention ROI
        st.markdown('#### Cost of Doing Nothing vs Intervening')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                'Cost if all at-risk leave',
                f'${cost:,.0f}',
                delta='Do nothing',
                delta_color='inverse')
        with col2:
            st.metric(
                'Intervention cost estimate',
                f'${int_cost:,.0f}',
                delta='Stay interviews + reviews')
        with col3:
            st.metric(
                'Conservative net saving',
                f'${saved - int_cost:,.0f}',
                delta=f'Retaining {retained} of {at_risk} employees')

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — DOWNLOAD REPORTS
    # ══════════════════════════════════════════════════════════════════════

    with tab4:
        st.subheader('Download Your Reports')
        st.markdown(
            'Download the full results, the top 25 action list, '
            'and the exploratory summary report.')

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### Full Scored Employee List')
            st.markdown(
                'All employees with flight risk scores, '
                'risk categories, and key signals.')
            full_csv = df_scored.to_csv(
                index=False).encode('utf-8')
            st.download_button(
                label='⬇ Download Full Scored List (CSV)',
                data=full_csv,
                file_name=f'flight_risk_full_'
                          f'{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv')

        with col2:
            st.markdown('#### Top 25 — HR Action List')
            st.markdown(
                'Top 25 at-risk employees with primary driver '
                'and recommended HR intervention.')
            if at_risk > 0:
                top25_csv = top25.to_csv(
                    index=False).encode('utf-8')
                st.download_button(
                    label='⬇ Download Top 25 Action List (CSV)',
                    data=top25_csv,
                    file_name=f'flight_risk_top25_'
                              f'{datetime.now().strftime("%Y%m%d")}'
                              f'.csv',
                    mime='text/csv')
            else:
                st.info('No at-risk employees to download.')

        st.divider()

        # Summary report
        st.markdown('#### Exploratory Summary Report')
        st.markdown(
            'A formatted text report covering dataset overview, '
            'key signals, risk by segment, critical zone, '
            'and intervention ROI. '
            'Ready to share with leadership.')

        summary_text = generate_summary(df_scored)

        st.text_area(
            'Summary Report Preview',
            summary_text,
            height=400)

        st.download_button(
            label='⬇ Download Summary Report (TXT)',
            data=summary_text.encode('utf-8'),
            file_name=f'flight_risk_summary_'
                      f'{datetime.now().strftime("%Y%m%d")}.txt',
            mime='text/plain')

        st.divider()

        # Disclaimer
        st.markdown("""
        ---
        **About this demonstration**

        This tool was built to demonstrate a production-ready
        methodology for employee flight risk prediction in
        financial services organizations.

        All data is synthetic. CAN North Financial is fictional.
        Predictions reflect patterns learned from synthetically
        generated employee data — not real organizational data.

        **Methodology and source code are proprietary.**

        ---
        """)

        st.caption(
            f'Model: Logistic Regression | '
            f'Trained on: 1,400 synthetic employees | '
            f'Recall: 98% on held-out test data | '
            f'Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}')


if __name__ == '__main__':
    main()