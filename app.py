import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

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

elif selected == "AI Recommendations":
    st.success("AI Recommendations Page")
    
elif selected == "Model Performance":
    st.success("Model Performance Page")
    
elif selected == "About":
    st.success("About Page")


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



# -------------------------------------------------
# CUSTOMER SELECTOR
# -------------------------------------------------

selected_customer_id = st.selectbox(
    "Select Customer",
    sorted(df["customer_id"].unique())
)

customer = df[
    df["customer_id"] == selected_customer_id
].iloc[0]

# ============================================
# GLOBAL CHURN PREDICTION
# ============================================

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

probability = model.predict_proba(
    prediction_input
)

churn_probability = round(
    probability[0][1] * 100,
    2
)


if selected == "Prediction":
    
    #CUSTOMER SELECTOR
    st.success(
    f"Selected Customer: {selected_customer_id}"
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
# CUSTOMER INSIGHTS PAGE
# -------------------------------------------------

if selected == "Customer Insights":

    st.title("👤 Customer 360 Insights")

    st.caption(
        "Comprehensive customer profile, value, health and risk intelligence"
    )

    st.markdown("---")

    # =====================================
    # CUSTOMER PROFILE
    # =====================================

    st.subheader("Customer Profile")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Country",
            customer["country"]
        )

    with c2:
        st.metric(
            "City",
            customer["city"]
        )

    with c3:
        st.metric(
            "Customer Type",
            customer["customer_type"]
        )

    with c4:
        st.metric(
            "Contract",
            customer["contract"]
        )

    st.markdown("---")

    # =====================================
    # CUSTOMER DEMOGRAPHICS
    # =====================================

    st.subheader("Customer Demographics")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Gender",
            customer["gender"]
        )

    with d2:
        st.metric(
            "Partner",
            customer["partner"]
        )

    with d3:
        st.metric(
            "Dependents",
            customer["dependents"]
        )

    with d4:
        st.metric(
            "Senior Citizen",
            customer["senior_citizen"]
        )

    st.markdown("---")

    # =====================================
    # TENURE SEGMENT
    # =====================================

    tenure = customer["tenure_months"]

    if tenure <= 12:
        tenure_segment = "New Customer"

    elif tenure <= 36:
        tenure_segment = "Established Customer"

    else:
        tenure_segment = "Loyal Customer"

    # =====================================
    # CUSTOMER VALUE INTELLIGENCE
    # =====================================

    st.subheader("Customer Value Intelligence")

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.metric(
            "CLTV",
            f"${float(customer['cltv']):,.0f}"
        )

    with v2:
        st.metric(
            "Value Segment",
            customer["customer_value_segment"]
        )

    with v3:
        st.metric(
            "CLTV Segment",
            customer["cltv_segment"]
        )

    with v4:
        st.metric(
            "Risk Segment",
            customer["risk_segment"]
        )

    st.markdown("---")

    # =====================================
    # CUSTOMER HEALTH
    # =====================================

    st.subheader("Customer Health Intelligence")

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "Health Score",
            customer["customer_health_score"]
        )

    with h2:
        st.metric(
            "Network Score",
            customer["network_quality_score"]
        )

    with h3:
        st.metric(
            "App Logins",
            customer["app_logins"]
        )

    with h4:
        st.metric(
            "Data Usage (GB)",
            customer["avg_monthly_data_usage_gb"]
        )

    st.markdown("---")

    # =====================================
    # BILLING & TENURE
    # =====================================

    st.subheader("Billing & Tenure")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.metric(
            "Tenure (Months)",
            customer["tenure_months"]
        )

    with b2:
        st.metric(
            "Tenure Segment",
            tenure_segment
        )

    with b3:
        st.metric(
            "Monthly Charge",
            f"${float(customer['monthly_charge']):,.2f}"
        )

    with b4:
        st.metric(
            "Total Charges",
            f"${float(customer['total_charges']):,.2f}"
        )

    st.markdown("---")

    # =====================================
    # CUSTOMER RISK SIGNALS
    # =====================================

    st.subheader("Customer Risk Signals")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Complaint Count",
            customer["complaint_count"]
        )

    with r2:
        st.metric(
            "Payment Delay Days",
            customer["payment_delay_days"]
        )

    with r3:
        st.metric(
            "Churn Label",
            customer["churn_label"]
        )

    with r4:
        st.metric(
            "Roaming Usage",
            customer["roaming_usage"]
        )



# -------------------------------------------------
# RISK SEGMENTATION PAGE
# -------------------------------------------------

if selected == "Risk Segmentation":

    st.title("⚠ Risk Segmentation Intelligence")

    st.caption(
        "Portfolio-wide customer risk analytics and churn exposure monitoring"
    )

    st.markdown("---")

    # ============================================
    # RISK KPI CARDS
    # ============================================

    high_risk_count = (
        df["risk_segment"] == "High Risk"
    ).sum()

    medium_risk_count = (
        df["risk_segment"] == "Medium Risk"
    ).sum()

    low_risk_count = (
        df["risk_segment"] == "Low Risk"
    ).sum()

    high_risk_pct = round(
        (high_risk_count / len(df)) * 100,
        1
    )

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.metric(
            "High Risk Customers",
            f"{high_risk_count:,}"
        )

    with k2:
        st.metric(
            "Medium Risk Customers",
            f"{medium_risk_count:,}"
        )

    with k3:
        st.metric(
            "Low Risk Customers",
            f"{low_risk_count:,}"
        )

    with k4:
        st.metric(
            "High Risk %",
            f"{high_risk_pct}%"
        )

    st.markdown("---")

    # ============================================
    # RISK DISTRIBUTION DONUT
    # ============================================

    st.subheader(
        "Risk Distribution"
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

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # COUNTRY + CUSTOMER TYPE
    # ============================================

    left,right = st.columns(2)

    with left:

        st.subheader(
            "High Risk Customers by Country"
        )

        high_country = df[
            df["risk_segment"] == "High Risk"
        ]

        high_country = (
            high_country["country"]
            .value_counts()
            .reset_index()
        )

        high_country.columns = [
            "Country",
            "Customers"
        ]

        fig_country = px.bar(
            high_country,
            x="Country",
            y="Customers",
            color="Country"
        )

        st.plotly_chart(
            fig_country,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Risk by Customer Type"
        )

        risk_type = (
            df.groupby(
                "customer_type"
            )["risk_segment"]
            .count()
            .reset_index()
        )

        risk_type.columns = [
            "Customer Type",
            "Customers"
        ]

        fig_type = px.bar(
            risk_type,
            x="Customer Type",
            y="Customers",
            color="Customer Type"
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )

    st.markdown("---")

    # ============================================
    # REVENUE EXPOSURE
    # ============================================

    st.subheader(
        "Revenue Exposure by Risk Segment"
    )

    revenue_risk = (
        df.groupby(
            "risk_segment"
        )["cltv"]
        .sum()
        .reset_index()
    )

    fig_revenue = px.bar(
        revenue_risk,
        x="risk_segment",
        y="cltv",
        color="risk_segment",
        color_discrete_map={
            "High Risk":"#EF4444",
            "Medium Risk":"#F59E0B",
            "Low Risk":"#22C55E"
        }
    )

    st.plotly_chart(
        fig_revenue,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # AI RISK INTELLIGENCE
    # ============================================

    st.subheader(
        "🤖 AI Risk Intelligence"
    )

    top_country = (
        df[df["risk_segment"] == "High Risk"]
        ["country"]
        .mode()[0]
    )

    top_customer_type = (
        df[df["risk_segment"] == "High Risk"]
        ["customer_type"]
        .mode()[0]
    )

    st.info(
        f"""
        High Risk customers account for {high_risk_pct}% of the customer base.

        The largest concentration of High Risk customers is in {top_country}.

        {top_customer_type} customers contribute the highest volume of churn exposure.

        Retention campaigns should prioritize High Risk customers with low health scores and payment delays.
        """
    )


# -------------------------------------------------
# REVENUE PROTECTION PAGE
# -------------------------------------------------

if selected == "Revenue Protection":

    st.title("💰 Revenue Protection Intelligence")

    st.caption(
        "Revenue exposure monitoring and retention opportunity analytics"
    )

    st.markdown("---")

    # ============================================
    # KPI CALCULATIONS
    # ============================================

    total_cltv = df["cltv"].astype(float).sum()

    high_risk_df = df[
        df["risk_segment"] == "High Risk"
    ]

    high_risk_cltv = (
        high_risk_df["cltv"]
        .astype(float)
        .sum()
    )

    medium_risk_df = df[
        df["risk_segment"] == "Medium Risk"
    ]

    medium_risk_cltv = (
        medium_risk_df["cltv"]
        .astype(float)
        .sum()
    )

    low_risk_df = df[
        df["risk_segment"] == "Low Risk"
    ]

    low_risk_cltv = (
        low_risk_df["cltv"]
        .astype(float)
        .sum()
    )

    total_revenue_exposure = (
        high_risk_cltv * 0.70
    )

    potential_revenue_saved = (
        total_revenue_exposure * 0.60
    )

    # ============================================
    # KPI CARDS
    # ============================================

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.metric(
            "Revenue At Risk",
            f"${total_revenue_exposure:,.0f}"
        )

    with k2:
        st.metric(
            "Portfolio CLTV",
            f"${total_cltv:,.0f}"
        )

    with k3:
        st.metric(
            "Potential Revenue Saved",
            f"${potential_revenue_saved:,.0f}"
        )

    with k4:
        st.metric(
            "High Risk Exposure",
            f"${high_risk_cltv:,.0f}"
        )

    st.markdown("---")

    # ============================================
    # REVENUE BY RISK SEGMENT
    # ============================================

    st.subheader(
        "Revenue Exposure by Risk Segment"
    )

    revenue_risk = pd.DataFrame({
        "Risk Segment":[
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        "Revenue":[
            high_risk_cltv,
            medium_risk_cltv,
            low_risk_cltv
        ]
    })

    fig_risk_revenue = px.bar(
        revenue_risk,
        x="Risk Segment",
        y="Revenue",
        color="Risk Segment",
        color_discrete_map={
            "High Risk":"#EF4444",
            "Medium Risk":"#F59E0B",
            "Low Risk":"#22C55E"
        }
    )

    st.plotly_chart(
        fig_risk_revenue,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # COUNTRY + CUSTOMER TYPE
    # ============================================

    left,right = st.columns(2)

    with left:

        st.subheader(
            "Revenue Exposure by Country"
        )

        country_revenue = (
            df.groupby("country")["cltv"]
            .sum()
            .reset_index()
        )

        fig_country = px.bar(
            country_revenue,
            x="country",
            y="cltv",
            color="country"
        )

        st.plotly_chart(
            fig_country,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Revenue Exposure by Customer Type"
        )

        customer_type_revenue = (
            df.groupby("customer_type")["cltv"]
            .sum()
            .reset_index()
        )

        fig_customer_type = px.bar(
            customer_type_revenue,
            x="customer_type",
            y="cltv",
            color="customer_type"
        )

        st.plotly_chart(
            fig_customer_type,
            use_container_width=True
        )

    st.markdown("---")

    # ============================================
    # TOP 10 HIGH VALUE AT-RISK CUSTOMERS
    # ============================================

    st.subheader(
        "Top 10 Revenue At-Risk Customers"
    )

    top_risk_customers = (
        df[df["risk_segment"] == "High Risk"]
        .sort_values(
            "cltv",
            ascending=False
        )
        [
            [
                "customer_id",
                "country",
                "customer_type",
                "cltv",
                "customer_health_score"
            ]
        ]
        .head(10)
    )

    st.dataframe(
        top_risk_customers,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # AI REVENUE INTELLIGENCE
    # ============================================

    st.subheader(
        "🤖 Revenue Protection Intelligence"
    )

    top_country = (
        df.groupby("country")["cltv"]
        .sum()
        .idxmax()
    )

    top_customer_type = (
        df.groupby("customer_type")["cltv"]
        .sum()
        .idxmax()
    )

    st.info(
        f"""
        Portfolio CLTV currently stands at ${total_cltv:,.0f}.

        Estimated revenue exposure is ${total_revenue_exposure:,.0f}.

        {top_customer_type} customers represent the highest revenue concentration.

        {top_country} contributes the largest share of portfolio value.

        A focused retention campaign targeting High Risk customers could potentially recover approximately ${potential_revenue_saved:,.0f} in future revenue.
        """
    )

# ============================================
# AI RECOMMENDATIONS
# ============================================

if selected == "AI Recommendations":

    st.title("🤖 GCC Telecom AI Copilot")

    st.caption(
        "AI-powered churn analysis, retention strategy and revenue protection assistant"
    )

    st.markdown("---")

    st.subheader("Selected Customer")

    st.json({
        "Customer ID": customer["customer_id"],
        "Country": customer["country"],
        "Customer Type": customer["customer_type"],
        "Risk Segment": customer["risk_segment"],
        "Health Score": customer["customer_health_score"],
        "CLTV": customer["cltv"],
        "Churn Probability": f"{churn_probability:.1f}%"
    })

    st.markdown("---")

    question = st.selectbox(
        "Ask AI Copilot",
        [
            "Why is this customer high risk?",
            "Generate retention strategy",
            "How can we reduce churn?",
            "How much revenue is at risk?",
            "Summarize this customer"
        ]
    )

    
    if st.button("🚀 Generate AI Analysis"):

        try:

            prompt = f"""
                You are a senior telecom retention consultant.
                
                Customer Information:
                
                Customer ID: {customer['customer_id']}
                Country: {customer['country']}
                Customer Type: {customer['customer_type']}
                Tenure: {customer['tenure_months']}
                Contract: {customer['contract']}
                CLTV: {customer['cltv']}
                Health Score: {customer['customer_health_score']}
                Risk Segment: {customer['risk_segment']}
                Complaints: {customer['complaint_count']}
                Payment Delay: {customer['payment_delay_days']}
                App Logins: {customer['app_logins']}
                Network Quality: {customer['network_quality_score']}
                Average Data Usage: {customer['avg_monthly_data_usage_gb']}
                
                Churn Probability: {churn_probability:.1f}%
                
                Question:
                {question}
                
                Provide:
                
                1. Executive Summary
                2. Key Risk Drivers
                3. Retention Strategy
                4. Revenue Protection Actions
                5. Next Best Action
                
                Use professional telecom business language.
            """

            with st.spinner("🤖 Gemini AI is analyzing customer..."):

                model = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )

                response = model.generate_content(
                    prompt
                )

            st.markdown("### 🤖 AI Analysis")

            st.markdown(
                response.text
            )

        except Exception as e:

            st.error(
                f"AI Service Error: {str(e)}"
            )

if selected == "Model Performance":

    st.title("📈 Model Performance")

    st.info(
        "Model Performance Page"
    )

if selected == "About":

    st.title("ℹ About")

    st.info(
        "About Page"
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

