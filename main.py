import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
from prediction_helper import predict

st.set_page_config(page_title="Credit Risk Model", page_icon="💳", layout="centered")
st.title("💳 Credit Risk Model")
st.write("Enter applicant details to estimate default risk and credit score.")

with st.form("risk_form"):
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    with row1[0]:
        age = st.number_input("Age", min_value=18, max_value=100, step=1, value=18)
    with row1[1]:
        income = st.number_input("Income", min_value=0, max_value=10000000, step=50000, value=800000)
    with row1[2]:
        loan_amount = st.number_input("Loan Amount", min_value=0, max_value=20000000, value=2000000)

    loan_to_income_ratio = loan_amount / income if income > 0 else 0
    with row2[0]:
        st.metric("Loan to Income Ratio", f"{loan_to_income_ratio:.2f}")
    with row2[1]:
        loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
    with row2[2]:
        avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)

    with row3[0]:
        delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
    with row3[1]:
        credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
    with row3[2]:
        num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

    with row4[0]:
        residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
    with row4[1]:
        loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
    with row4[2]:
        loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

    submitted = st.form_submit_button("Calculate Risk", use_container_width=True)

st.markdown("---")

if submitted:
    probability, credit_score, rating = predict(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    rating_colors = {
        "Poor": "#d9534f",
        "Average": "#f0ad4e",
        "Good": "#c3d825",
        "Excellent": "#5cb85c"
    }

    col1, col2 = st.columns([1, 1], vertical_alignment="center")

    with col1:
        st.metric("Default Probability", f"{probability:.2%}")
        st.metric("Credit Score", credit_score)
        st.markdown(
            f"""
            <div style="
                background-color:{rating_colors.get(rating, '#999')};
                padding:12px;
                border-radius:8px;
                text-align:center;
                font-size:20px;
                font-weight:bold;
                color:white;">
                {rating}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; font-size:42px; font-weight:bold; color:white;">
                {credit_score}
            </div>
            <div style="text-align:center; font-size:14px; color:#aaa; margin-bottom:4px;">
                Range: 300 (Poor) &nbsp;→&nbsp; 900 (Excellent)
            </div>
            """,
            unsafe_allow_html=True
        )

        V_MIN, V_MAX = 300, 900
        R_IN, R_OUT = 0.68, 1.0

        def val_to_theta(v):
            return math.pi * (1 - (v - V_MIN) / (V_MAX - V_MIN))

        def band_polygon(v_start, v_end, n=40):
            th_a, th_b = val_to_theta(v_start), val_to_theta(v_end)
            thetas_outer = np.linspace(th_a, th_b, n)
            thetas_inner = thetas_outer[::-1]
            xs = list(R_OUT * np.cos(thetas_outer)) + list(R_IN * np.cos(thetas_inner))
            ys = list(R_OUT * np.sin(thetas_outer)) + list(R_IN * np.sin(thetas_inner))
            return xs, ys

        bands = [
            (300, 500, "#d9534f", "Poor"),
            (500, 650, "#f0ad4e", "Average"),
            (650, 750, "#c3d825", "Good"),
            (750, 900, "#5cb85c", "Excellent"),
        ]

        fig = go.Figure()

        for v_start, v_end, color, name in bands:
            xs, ys = band_polygon(v_start, v_end)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill='toself', fillcolor=color,
                mode='lines', line=dict(color=color, width=0),
                hoverinfo='skip', showlegend=False
            ))

        annotations = []
        label_r = (R_IN + R_OUT) / 2 + 0.04  # was + 0.08 → pulls labels in a bit
        for v_start, v_end, color, name in bands:
            mid_v = (v_start + v_end) / 2
            th = val_to_theta(mid_v)
            lx, ly = label_r * math.cos(th), label_r * math.sin(th)
            rotation = (90 - math.degrees(th)) * 0.5  # was * 0.7 → less steep, won't run into the corner
            annotations.append(dict(
                x=lx, y=ly, text=f"<b>{name}</b>", showarrow=False,
                textangle=rotation, font=dict(color="white", size=13),
                xanchor="center", yanchor="middle"
            ))

        for tv in [400, 600, 800]:
            th = val_to_theta(tv)
            tx, ty = (R_OUT + 0.09) * math.cos(th), (R_OUT + 0.09) * math.sin(th)
            annotations.append(dict(
                x=tx, y=ty, text=str(tv), showarrow=False,
                font=dict(color="white", size=12),
                xanchor="center", yanchor="middle"
            ))

        frac = (credit_score - V_MIN) / (V_MAX - V_MIN)
        theta_needle = math.pi * (1 - frac)
        needle_len = R_IN * 0.92
        x_tip, y_tip = needle_len * math.cos(theta_needle), needle_len * math.sin(theta_needle)

        fig.add_trace(go.Scatter(
            x=[0, x_tip], y=[0, y_tip], mode='lines',
            line=dict(color='white', width=4), hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(color='white', size=14),
            hoverinfo='skip', showlegend=False
        ))

        fig.update_xaxes(visible=False, range=[-1.4, 1.4])
        fig.update_yaxes(visible=False, range=[-0.25, 1.15], scaleanchor="x", scaleratio=1)

        fig.update_layout(
            annotations=annotations,
            height=290,
            margin=dict(t=10, b=0, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})