import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="GCC Telecom Intelligence",
    page_icon="📡",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("telecom_data.csv")

model = joblib.load("logistic_regression.pkl")

# -------------------------------------------------
# LOAD ICONS
# -------------------------------------------------

tower_icon = Image.open("assets/radio-tower.png")
dashboard_icon = Image.open("assets/dashboard.png")
customer_icon = Image.open("assets/customer.png")
salary_icon = Image.open("assets/salary.png")
heart_icon = Image.open("assets/heart-rate.png")
robot_icon = Image.open("assets/robot.png")

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#F4F8FC;
}

[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #071A52 0%,
        #0A2B75 100%
    );
}

[data-testid="stSidebar"] *{
    color:white;
}

h1{
    color:#1E293B;
    font-weight:800;
}

h2{
    color:#1E293B;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:15px;
    box-shadow:
    0px 4px 14px rgba(0,0,0,0.08);
}

.stButton button{
    background:#2563EB;
    color:white;
    border-radius:12px;
    border:none;
    font-weight:bold;
}

.stButton button:hover{
    background:#1D4ED8;
}

</style>
""",
unsafe_allow_html=True)



# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

from streamlit_option_menu import option_menu

with st.sidebar:

    st.image(tower_icon, width=90)

    st.markdown(
        """
        <h2 style='color:white;
        text-align:center;'>
        GCC Telecom AI
        </h2>
        """,
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Prediction",
            "Customer Insights",
            "Risk Segmentation",
            "Revenue Protection",
            "AI Recommendations",
            "Model Performance",
            "About"
        ],
        icons=[
            "house-fill",
            "people-fill",
            "shield-fill-check",
            "cash-stack",
            "robot",
            "graph-up-arrow",
            "info-circle-fill"
        ],
        default_index=0
    )


if selected == "Prediction":
    st.success("Prediction Page")

elif selected == "Customer Insights":
    st.success("Customer Insights Page")

elif selected == "Risk Segmentation":
    st.success("Risk Segmentation Page")

elif selected == "Revenue Protection":
    st.success("Revenue Protection Page")
    
  
 

    


# -------------------------------------------------
# HEADER
# -------------------------------------------------


col1,col2 = st.columns([8,2])

with col1:
    st.markdown("""
    <h1 style="
    font-size:48px;
    font-weight:800;
    color:#0F172A;
    ">
    GCC Telecom Customer Intelligence
    </h1>

    <p style="
    color:#64748B;
    font-size:18px;
    ">
    AI-powered churn prediction and revenue protection platform
    </p>
    """, unsafe_allow_html=True)

with col2:
    st.image("assets/business.png", width=220)







# -------------------------------------------------
# KPI VALUES
# -------------------------------------------------

total_customers = len(df)

avg_cltv = df["cltv"].astype(float).mean()

avg_health = df["customer_health_score"].astype(float).mean()

high_risk = (df["risk_segment"]=="High Risk").sum()


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with c2:
    st.metric(
        "Average CLTV",
        f"${avg_cltv:,.0f}"
    )

with c3:
    st.metric(
        "Health Score",
        f"{avg_health:.1f}"
    )

with c4:
    st.metric(
        "High Risk Customers",
        f"{high_risk:,}"
    )

st.markdown("""
<style>

div[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:20px;
    border:1px solid #E2E8F0;
    box-shadow:0px 4px 12px rgba(0,0,0,0.06);
}

div[data-testid="metric-container"] label{
    color:#64748B;
}

</style>
""",
unsafe_allow_html=True)


if selected == "Prediction":

    # -------------------------------------------------
    # CUSTOMER SELECTOR
    # -------------------------------------------------

    st.markdown("---")

    customer_id = st.selectbox(
    "Select Customer ID",
    sorted(df["customer_id"].unique())
    )

    customer = df[
    df["customer_id"] == customer_id
    ].iloc[0]

    st.success(
    f"Selected Customer: {customer_id}"
    )

    # -------------------------------------------------
    # CUSTOMER PROFILE
    # -------------------------------------------------

    st.markdown("## 👤 Customer Profile")

    p1,p2,p3,p4 = st.columns(4)

    with p1:
        st.markdown(f"""
            <div style="
            background:white;
            padding:20px;
            border-radius:16px;
            border:1px solid #E5E7EB;
            ">
            <div style="color:#64748B;">Country</div>
            <h4>{customer['country']}</h4>
            </div>
    """,
    unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
            <div style="
            background:white;
            padding:20px;
            border-radius:16px;
            border:1px solid #E5E7EB;
            ">
            <div style="color:#64748B;">City</div>
            <h4>{customer['city']}</h4>
            </div>
    """,
    unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
            <div style="
            background:white;
            padding:20px;
            border-radius:16px;
            border:1px solid #E5E7EB;
            ">
            <div style="color:#64748B;">Customer Type</div>
            <h4>{customer['customer_type']}</h4>
            </div>
    """,
    unsafe_allow_html=True)

    with p4:
        st.markdown(f"""
            <div style="
            background:white;
            padding:20px;
            border-radius:16px;
            border:1px solid #E5E7EB;
            ">
            <div style="color:#64748B;">Contract</div>
            <h4>{customer['contract']}</h4>
            </div>
    """,
    unsafe_allow_html=True)

    # -------------------------------------------------
    # CHURN PREDICTION
    # -------------------------------------------------

    feature_columns = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "roaming_usage",
    "app_logins",
    "avg_monthly_data_usage_gb"
    ]

    prediction_input = pd.DataFrame([{
    "tenure_months": customer["tenure_months"],
    "monthly_charge": customer["monthly_charge"],
    "total_charges": customer["total_charges"],
    "roaming_usage": customer["roaming_usage"],
    "app_logins": customer["app_logins"],
    "avg_monthly_data_usage_gb":
    customer["avg_monthly_data_usage_gb"]
    }])

    prediction_input = prediction_input.fillna(0)

    prediction = model.predict(
    prediction_input
    )

    probability = model.predict_proba(
    prediction_input
    )

    churn_probability = round(
    probability[0][1] * 100,
    2
    )

    # -------------------------------------------------
    # RISK LEVEL
    # -------------------------------------------------

    if churn_probability >= 70:

        predicted_risk = "High Risk"

    elif churn_probability >= 40:

        predicted_risk = "Medium Risk"

    else:

        predicted_risk = "Low Risk"

    # -------------------------------------------------
    # TENURE SEGMENT
    # -------------------------------------------------

    tenure = customer["tenure_months"]

    if tenure <= 12:

        tenure_segment = "New Customer"

    elif tenure <= 36:

        tenure_segment = "Established Customer"

    else:

        tenure_segment = "Loyal Customer"

    # -------------------------------------------------
    # REVENUE AT RISK
    # -------------------------------------------------

    cltv = float(customer["cltv"])

    revenue_at_risk = round(
    cltv * (churn_probability / 100),
    2
    )

    # -------------------------------------------------
    # PREDICTION INTELLIGENCE
    # -------------------------------------------------

    st.markdown("## 🎯 Prediction Intelligence")

    pred1, pred2, pred3 = st.columns(3)

    # -----------------------------
    # CHURN PROBABILITY
    # -----------------------------

    with pred1:

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:18px;
        border:1px solid #E5E7EB;
        text-align:center;
        ">

        <div style="
        color:#64748B;
        font-size:15px;
        ">
        Churn Probability
        </div>

        <h1 style="
        color:#2563EB;
        font-size:42px;
        ">
        {churn_probability:.1f}%
        </h1>

        </div>
    """,
    unsafe_allow_html=True)

    # -----------------------------
    # RISK LEVEL
    # -----------------------------

    with pred2:

        risk_color = "#22C55E"

        if predicted_risk == "High Risk":
            risk_color = "#EF4444"

        elif predicted_risk == "Medium Risk":
            risk_color = "#F59E0B"

        st.markdown(f"""
            <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            border:1px solid #E5E7EB;
            text-align:center;
            ">
    
            <div style="
            color:#64748B;
            font-size:15px;
            ">
            Risk Level
            </div>
    
            <h1 style="
            color:{risk_color};
            font-size:36px;
            ">
            {predicted_risk}
            </h1>
    
            </div>
    """,
    unsafe_allow_html=True)

    # -----------------------------
    # REVENUE AT RISK
    # -----------------------------

    with pred3:

        st.markdown(f"""
            <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            border:1px solid #E5E7EB;
            text-align:center;
            ">

            <div style="
            color:#64748B;
            font-size:15px;
            ">
            Revenue At Risk
            </div>

            <h1 style="
            color:#DC2626;
            font-size:42px;
            ">
            ${revenue_at_risk:,.0f}
            </h1>

            </div>
    """,
    unsafe_allow_html=True)

    # -----------------------------
    # RISK SCORE
    # -----------------------------

    st.markdown("### Risk Score")

    st.progress(
    min(
    int(churn_probability),
    100
    )
    )

    # -----------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------

    if predicted_risk == "High Risk":

        st.error(
            f"""
            Customer has a high probability of churn.
            Potential revenue exposure is
            ${revenue_at_risk:,.0f}.
            Immediate retention action is recommended.
            """
        )

    elif predicted_risk == "Medium Risk":
        st.warning(
            """
            Customer requires monitoring and targeted engagement.
            """
    )

    else:

        st.success(
        """
        Customer currently appears stable and suitable for upsell opportunities.
        """
        )

    # -------------------------------------------------
    # DECISION INTELLIGENCE
    # -------------------------------------------------

    st.markdown("---")

    recommendations = []

    if customer["complaint_count"] >= 3:
         recommendations.append(
    "Assign dedicated relationship manager"
    )
    
    if customer["payment_delay_days"] >= 10:
        recommendations.append(
    "Offer flexible billing plan"
    )
    
    if customer["customer_health_score"] < 85:
        recommendations.append(
        "Launch customer retention campaign"
        )
    
    if customer["app_logins"] < 10:
        recommendations.append(
        "Increase digital engagement"
        )
    
    if customer["tenure_months"] < 12:
        recommendations.append(
        "Offer loyalty welcome package"
        )
    
    if customer["network_quality_score"] < 7:
        recommendations.append(
        "Provide network quality support"
        )
    
    if len(recommendations) == 0:
        recommendations.append(
    "Customer currently appears stable"
    )

    # -------------------------------------------------
    # RISK DRIVERS
    # -------------------------------------------------

    drivers = []

    if customer["complaint_count"] >= 3:
        drivers.append(
    "High Complaint Count"
    )

    if customer["payment_delay_days"] >= 10:
        drivers.append(
    "Payment Delays"
    )

    if customer["contract"] == "Month-to-month":
        drivers.append(
    "Month-to-month Contract"
    )

    if customer["customer_health_score"] < 85:
        drivers.append(
    "Low Health Score"
    )

    if customer["tenure_months"] < 12:
        drivers.append(
    "Short Customer Tenure"
    )

    col1, col2 = st.columns(2)

    # -------------------------------------------------
    # AI RECOMMENDATIONS CARD
    # -------------------------------------------------

    with col1:

        st.markdown("""
            <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            border:1px solid #E5E7EB;
            ">
            <h3>🤖 AI Recommendation Engine</h3>
            <p style="color:#64748B;">
            Recommended retention actions
            </p>
        """,
        unsafe_allow_html=True)

    for item in recommendations:
        st.success(item)

        st.markdown("</div>", unsafe_allow_html=True)




    # -------------------------------------------------
    # RISK DRIVERS CARD
    # -------------------------------------------------

    with col2:

        st.markdown("""
            <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            border:1px solid #E5E7EB;
            ">
            <h3>⚠ Key Risk Drivers</h3>
            <p style="color:#64748B;">
            Factors contributing to churn risk
            </p>
        """,
        unsafe_allow_html=True)

    for item in drivers:
        st.warning(item)

        st.markdown("</div>", unsafe_allow_html=True)





    # -------------------------------------------------
    # REVENUE PROTECTION
    # -------------------------------------------------

    st.markdown("---")

    st.subheader(
    "Revenue Protection"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(f"""
            <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            border-left:6px solid #2563EB;
            box-shadow:0px 4px 12px rgba(0,0,0,0.05);
            ">

            <p style="color:#64748B;">
            Customer CLTV
            </p>

            <h2>
            ${cltv:,.0f}
            </h2>

            </div>
    """,
    unsafe_allow_html=True)

    with r2:

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:18px;
        border-left:6px solid #EF4444;
        box-shadow:0px 4px 12px rgba(0,0,0,0.05);
        ">

        <p style="color:#64748B;">
        Revenue At Risk
        </p>

        <h2>
        ${revenue_at_risk:,.0f}
        </h2>

        </div>
    """,
    unsafe_allow_html=True)

    with r3:

        retention_gain = round(
        revenue_at_risk * 0.60,
        2
    )

        st.markdown(f"""
        <div style="
        background:white;
        padding:25px;
        border-radius:18px;
        border-left:6px solid #22C55E;
        box-shadow:0px 4px 12px rgba(0,0,0,0.05);
        ">

        <p style="color:#64748B;">
        Potential Revenue Saved
        </p>

        <h2>
        ${retention_gain:,.0f}
        </h2>

        </div>
    """,
    unsafe_allow_html=True)

    if selected == "Customer Insights":
    
        st.title("👥 Customer Insights")
    
        st.info(
            "Customer Insights page coming next."
        )
    
    if selected == "Risk Segmentation":
    
        st.title("⚠ Risk Segmentation")
    
        st.info(
            "Risk Segmentation page coming next."
        )
    
    if selected == "Revenue Protection":
    
        st.title("💰 Revenue Protection")
    
        st.info(
            "Revenue Protection page coming next."
        )




# -------------------------------------------------
# BUSINESS INSIGHT
# -------------------------------------------------

if predicted_risk == "High Risk":
    st.error(
        f"""
        This customer represents approximately
        ${revenue_at_risk:,.0f}
        of revenue exposure and should be prioritized
        for retention efforts.
        """
    )

elif predicted_risk == "Medium Risk":

    st.warning(
        """
        Customer should be monitored closely
        and targeted with engagement campaigns.
        """
    )

else:

    st.success(
        """
        Customer currently appears stable
        and suitable for upsell opportunities.
        """
    )



# -------------------------------------------------
# EXECUTIVE ANALYTICS DASHBOARD
# -------------------------------------------------

import plotly.express as px
import plotly.graph_objects as go

st.markdown("---")

st.subheader(
    "📊 Executive Analytics Dashboard"
)

tab1, tab2, tab3 = st.tabs(
    [
        "Risk Analytics",
        "Customer Analytics",
        "Revenue Analytics"
    ]
)

# =================================================
# TAB 1
# =================================================

with tab1:

    left,right = st.columns(2)

    with left:

        st.markdown(
            "### Risk Distribution"
        )

        risk_counts = (
            df["risk_segment"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Segment",
            "Customers"
        ]

        fig_risk = px.pie(
            risk_counts,
            names="Risk Segment",
            values="Customers",
            hole=0.60,
            color="Risk Segment",
            color_discrete_map={
                "High Risk":"#EF4444",
                "Medium Risk":"#F59E0B",
                "Low Risk":"#22C55E"
            }
        )

        fig_risk.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    with right:

        st.markdown(
            "### High Risk Customers"
        )

        high_risk_df = df[
            df["risk_segment"]
            ==
            "High Risk"
        ]

        risk_country = (
            high_risk_df["country"]
            .value_counts()
            .reset_index()
        )

        risk_country.columns = [
            "Country",
            "High Risk Customers"
        ]

        fig_country = px.bar(
            risk_country,
            x="Country",
            y="High Risk Customers",
            color="Country"
        )

        fig_country.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_country,
            use_container_width=True
        )

# =================================================
# TAB 2
# =================================================

with tab2:

    left,right = st.columns(2)

    with left:

        st.image(
            "assets/globe.png",
            width=60
        )

        st.markdown(
            "### Customers by Country"
        )

        country_counts = (
            df["country"]
            .value_counts()
            .reset_index()
        )

        country_counts.columns = [
            "Country",
            "Customers"
        ]

        fig_customer_country = px.bar(
            country_counts,
            x="Country",
            y="Customers",
            color="Country"
        )

        fig_customer_country.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_customer_country,
            use_container_width=True
        )

    with right:

        st.markdown(
            "### Customer Type Distribution"
        )

        customer_type = (
            df["customer_type"]
            .value_counts()
            .reset_index()
        )

        customer_type.columns = [
            "Customer Type",
            "Customers"
        ]

        fig_customer_type = px.pie(
            customer_type,
            names="Customer Type",
            values="Customers",
            hole=0.55
        )

        fig_customer_type.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_customer_type,
            use_container_width=True
        )

# =================================================
# TAB 3
# =================================================

with tab3:

    left,right = st.columns(2)

    with left:

        st.markdown(
            "### Revenue by Risk Segment"
        )

        revenue_risk = (
            df.groupby(
                "risk_segment"
            )["cltv"]
            .sum()
            .reset_index()
        )

        fig_rev = px.bar(
            revenue_risk,
            x="risk_segment",
            y="cltv",
            color="risk_segment"
        )

        fig_rev.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_rev,
            use_container_width=True
        )

    with right:

        st.markdown(
            "### Average CLTV by Customer Type"
        )

        cltv_type = (
            df.groupby(
                "customer_type"
            )["cltv"]
            .mean()
            .reset_index()
        )

        fig_cltv = px.bar(
            cltv_type,
            x="customer_type",
            y="cltv",
            color="customer_type"
        )

        fig_cltv.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_cltv,
            use_container_width=True
        )

# -------------------------------------------------
# MODEL PERFORMANCE CENTER
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "🧠 Model Performance Center"
)

m1,m2,m3,m4 = st.columns(4)

m1.metric(
    "Accuracy",
    "79.91%"
)

m2.metric(
    "Precision",
    "67.17%"
)

m3.metric(
    "Recall",
    "47.59%"
)

m4.metric(
    "F1 Score",
    "55.71%"
)

st.info(
    """
Algorithm: Logistic Regression

Dataset Size: 7,043 Customers

Features Used:

• tenure_months

• monthly_charge

• total_charges

• roaming_usage

• app_logins

• avg_monthly_data_usage_gb

Experiment Tracking: MLflow

Platform: Microsoft Fabric
"""
)

# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "🚀 GCC Telecom Intelligence Platform"
)

st.success(
    """
Built Using:

✓ Microsoft Fabric

✓ Lakehouse Architecture

✓ Bronze Layer

✓ Silver Layer

✓ Gold Layer

✓ Power BI

✓ MLflow

✓ Logistic Regression

✓ Churn Prediction

✓ Revenue Intelligence

✓ Customer Segmentation

✓ Streamlit Application
"""
)

st.caption(
    """
GCC Telecom Customer Intelligence Platform

End-to-End Data Engineering + Analytics +
Machine Learning + Business Intelligence Project
"""
)
    

