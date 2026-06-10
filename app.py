import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
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
            "Home",
            "Customer Insights",
            "Risk Segmentation",
            "Revenue Protection",
            "Model Performance",
            "About"
        ],
        icons=[
            "house-fill",
            "people-fill",
            "shield-fill-check",
            "cash-stack",
            "graph-up-arrow",
            "info-circle-fill"
        ],
        default_index=0
    )


if selected == "Home":
    st.success("Home Page")

elif selected == "Customer Insights":
    st.success("Customer Insights Page")

elif selected == "Risk Segmentation":
    st.success("Risk Segmentation Page")

elif selected == "Revenue Protection":
    st.success("Revenue Protection Page")
    
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
    st.image("assets/business.png", width=250)



if selected == "Home":

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



    # ============================================
    # EXECUTIVE DASHBOARD
    # ============================================


    st.title("📊 Executive Analytics Dashboard")
    st.caption(
        "Portfolio-level telecom intelligence across customer, risk and revenue analytics"
    )

    st.markdown("---")

    # -------------------------------------------------
    # EXECUTIVE KPI CALCULATIONS
    # -------------------------------------------------
    total_customers = len(df)

    high_risk_count = (df["risk_segment"] == "High Risk").sum()
    medium_risk_count = (df["risk_segment"] == "Medium Risk").sum()
    low_risk_count = (df["risk_segment"] == "Low Risk").sum()

    high_risk_pct = round((high_risk_count / total_customers) * 100, 1)

    avg_cltv = pd.to_numeric(df["cltv"], errors="coerce").mean()
    avg_health = pd.to_numeric(df["customer_health_score"], errors="coerce").mean()

    risk_weight_map = {
        "High Risk": 0.60,
        "Medium Risk": 0.30,
        "Low Risk": 0.10
    }

    est_revenue_at_risk = (
        pd.to_numeric(df["cltv"], errors="coerce")
        * df["risk_segment"].map(risk_weight_map).fillna(0.10)
    ).sum()

    # -------------------------------------------------
    # KPI CARDS
    # -------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("Total Customers", f"{total_customers:,}")

    with k2:
        st.metric("High Risk Customers", f"{high_risk_count:,}")

    with k3:
        st.metric("High Risk %", f"{high_risk_pct}%")

    with k4:
        st.metric("Average CLTV", f"${avg_cltv:,.0f}")

    with k5:
        st.metric("Estimated Revenue at Risk", f"${est_revenue_at_risk:,.0f}")

    st.markdown("---")

    # -------------------------------------------------
    # TABS
    # -------------------------------------------------
    tab_risk, tab_customer, tab_revenue = st.tabs([
        "Risk Analytics",
        "Customer Analytics",
        "Revenue Analytics"
    ])

    # =================================================
    # RISK ANALYTICS TAB
    # =================================================
    with tab_risk:

        st.subheader("Risk Distribution")

        col1, col2 = st.columns(2)

        with col1:
            risk_counts = (
                df["risk_segment"]
                .value_counts()
                .reset_index()
            )
            risk_counts.columns = ["Risk Segment", "Customers"]

            fig_risk = px.pie(
                risk_counts,
                names="Risk Segment",
                values="Customers",
                hole=0.60,
                color="Risk Segment",
                color_discrete_map={
                    "High Risk": "#EF4444",
                    "Medium Risk": "#F59E0B",
                    "Low Risk": "#22C55E"
                }
            )
            fig_risk.update_traces(textinfo="percent+label")
            fig_risk.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            st.subheader("High Risk Rate by Country (%)")

            country_risk_rate = (
                df.groupby("country")
                .apply(
                    lambda x:
                        (
                        x["risk_segment"] == "High Risk"
                        ).mean() * 100
                        )
                .reset_index(
                    name="Risk Rate (%)"
                    )
                )

            fig_country_risk = px.bar(
                    country_risk_rate,
                    x="country",
                    y="Risk Rate (%)",
                    color="Risk Rate (%)"
                )

            st.plotly_chart(
                fig_country_risk,
                use_container_width=True
            )

            st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Risk by Customer Type")
            type_risk_rate = (
                    df.groupby("customer_type")
                    .apply(
                        lambda x:
                        (
                            x["risk_segment"] == "High Risk"
                        ).mean() * 100
                    )
                    .reset_index(
                        name="Risk Rate (%)"
                    )
                )
                
            fig_type_risk = px.bar(
                    type_risk_rate,
                    x="customer_type",
                    y="Risk Rate (%)",
                    color="Risk Rate (%)"
                )
                
            st.plotly_chart(
                    fig_type_risk,
                    use_container_width=True
                )

        with col4:

            st.subheader("🤖 AI Risk Intelligence")
        
            highest_risk_country = (
                country_risk_rate
                .sort_values(
                    by="risk_rate",
                    ascending=False
                )
                .iloc[0]
            )
        
            highest_risk_type = (
                type_risk_rate
                .sort_values(
                    by="risk_rate",
                    ascending=False
                )
                .iloc[0]
            )
        
            st.info(f"""
        ### Risk Summary
        
        • {high_risk_pct}% of customers are currently High Risk.
        
        • Highest Risk Country:
        {highest_risk_country['country']}
        ({highest_risk_country['risk_rate']:.1f}%)
        
        • Highest Risk Customer Type:
        {highest_risk_type['customer_type']}
        ({highest_risk_type['risk_rate']:.1f}%)
        
        ### Recommended Actions
        
        1. Prioritize High Risk customers.
        
        2. Reduce payment delays.
        
        3. Improve customer health scores.
        
        4. Increase engagement among risky


        
    # =================================================
    # CUSTOMER ANALYTICS TAB
    # =================================================
    with tab_customer:

        st.subheader("Customer Analytics")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Customers by Country")

            country_df = (
                df["country"].value_counts().reset_index()
            )
            country_df.columns = ["Country", "Customers"]

            fig_country = px.bar(
                country_df,
                x="Country",
                y="Customers",
                color="Country"
            )
            fig_country.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_country, use_container_width=True)

        with c2:
            st.subheader("Customer Type Distribution")

            type_df = (
                df["customer_type"].value_counts().reset_index()
            )
            type_df.columns = ["Customer Type", "Customers"]

            fig_type = px.pie(
                type_df,
                names="Customer Type",
                values="Customers",
                hole=0.55
            )
            fig_type.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("---")

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Average Health Score by Customer Type")

            health_type = (
                df.groupby("customer_type")["customer_health_score"]
                .mean()
                .reset_index()
            )

            fig_health = px.bar(
                health_type,
                x="customer_type",
                y="customer_health_score",
                color="customer_health_score",
                color_continuous_scale="Blues"
            )
            fig_health.update_layout(
                height=380,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_health, use_container_width=True)

        with c4:
            st.subheader("Customer Intelligence Summary")

            top_country_customer = df["country"].value_counts().index[0]
            top_type_customer = df["customer_type"].value_counts().index[0]

            st.success(
                f"The portfolio is concentrated in **{top_country_customer}**.\n\n"
                f"**{top_type_customer}** customers represent the largest customer group.\n\n"
                f"Average customer health score is **{avg_health:,.1f}**."
            )

    # =================================================
    # REVENUE ANALYTICS TAB
    # =================================================
    with tab_revenue:

        st.subheader("Revenue Analytics")

        r1, r2 = st.columns(2)

        with r1:
            st.subheader("Revenue Exposure by Risk Segment")

            revenue_df = (
                df.groupby("risk_segment")["cltv"]
                .sum()
                .reset_index()
            )

            fig_revenue = px.bar(
                revenue_df,
                x="risk_segment",
                y="cltv",
                color="risk_segment",
                color_discrete_map={
                    "High Risk": "#EF4444",
                    "Medium Risk": "#F59E0B",
                    "Low Risk": "#22C55E"
                }
            )
            fig_revenue.update_layout(
                height=420,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_revenue, use_container_width=True)

        with r2:
            st.subheader("Revenue Exposure by Country")

            revenue_country = (
                df.groupby("country")["cltv"]
                .sum()
                .reset_index()
                .sort_values("cltv", ascending=False)
            )

            fig_rev_country = px.bar(
                revenue_country,
                x="country",
                y="cltv",
                color="cltv",
                color_continuous_scale="Blues"
            )
            fig_rev_country.update_layout(
                height=420,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig_rev_country, use_container_width=True)

        st.markdown("---")

        r3, r4 = st.columns(2)

        with r3:
            st.subheader("Top 10 High Risk Customers by CLTV")

            top_risk_customers = (
                df[df["risk_segment"] == "High Risk"]
                .sort_values(by="cltv", ascending=False)
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

        with r4:
            st.subheader("Revenue Intelligence Summary")

            top_rev_country = revenue_country.iloc[0]["country"]
            top_rev_country_value = revenue_country.iloc[0]["cltv"]

            st.info(
                f"Total portfolio CLTV is **${avg_cltv * total_customers:,.0f}**.\n\n"
                f"Estimated revenue at risk is **${est_revenue_at_risk:,.0f}**.\n\n"
                f"**{top_rev_country}** contributes the highest revenue exposure at **${top_rev_country_value:,.0f}**.\n\n"
                "The strongest monetization opportunity is in customers with High Risk and low health scores."
            )






if selected == "Customer Insights":

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
    
    
    

  
    # -------------------------------------------------
    # CUSTOMER PROFILE
    # -------------------------------------------------
    
    st.markdown("## 👤 Customer Profile")
    
    p1,p2,p3,p4 = st.columns(4)
    
    with p1:
        card.html=f"""
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
        st.markdown(card_html, unsafe_allow_html=True)
    
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
        st.markdown(
        f"""
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
        unsafe_allow_html=True
    )

    with p4:
        st.markdown(
        f"""
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
        unsafe_allow_html=True
    )
    



    
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
            "Summarize this customer",
            "What is the next best action?",
            "How should the retention team handle this customer?"
        ]
    )

    if st.button("🚀 Generate AI Analysis"):

        try:

            prompt = f"""
You are a Senior Telecom Retention Consultant.

Analyze the customer and provide a business-focused response.

CUSTOMER INFORMATION

Customer ID: {customer['customer_id']}
Country: {customer['country']}
Customer Type: {customer['customer_type']}
Contract: {customer['contract']}
Tenure: {customer['tenure_months']} months

CLTV: {customer['cltv']}
Risk Segment: {customer['risk_segment']}
Health Score: {customer['customer_health_score']}
Network Quality Score: {customer['network_quality_score']}

Complaint Count: {customer['complaint_count']}
Payment Delay Days: {customer['payment_delay_days']}
App Logins: {customer['app_logins']}
Roaming Usage: {customer['roaming_usage']}
Monthly Charge: {customer['monthly_charge']}
Data Usage: {customer['avg_monthly_data_usage_gb']}

Predicted Churn Probability:
{churn_probability:.1f}%

USER QUESTION:
{question}

Provide:

1. Executive Summary
2. Key Risk Drivers
3. Retention Strategy
4. Revenue Protection Actions
5. Next Best Action

Keep the answer concise and suitable for telecom executives.
"""

            with st.spinner(
                "🤖 AI Copilot is analyzing customer..."
            ):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content":
                            "You are an expert telecom churn and revenue retention consultant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

            st.markdown("---")

            st.markdown(
                "## 🤖 AI Copilot Analysis"
            )

            st.markdown(
                response.choices[0].message.content
            )

        except Exception as e:

            st.error(
                f"AI Service Error: {str(e)}"
            )


# ============================================
# MODEL PERFORMANCE
# ============================================

if selected == "Model Performance":

    st.title("📊 Model Performance")

    st.caption(
        "Evaluate churn prediction model accuracy and business effectiveness"
    )

    st.markdown("---")

    # ============================================
    # LOAD METRICS
    # ============================================

    metrics_df = pd.read_csv(
        "model_metrics.csv"
    )

    accuracy = metrics_df["accuracy"][0]
    precision = metrics_df["precision"][0]
    recall = metrics_df["recall"][0]
    f1 = metrics_df["f1"][0]

    # ============================================
    # KPI CARDS
    # ============================================

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            f"{accuracy*100:.1f}%"
        )

    with c2:
        st.metric(
            "Precision",
            f"{precision*100:.1f}%"
        )

    with c3:
        st.metric(
            "Recall",
            f"{recall*100:.1f}%"
        )

    with c4:
        st.metric(
            "F1 Score",
            f"{f1*100:.1f}%"
        )

    st.markdown("---")

    # ============================================
    # MODEL OVERVIEW
    # ============================================

    st.subheader(
        "Model Overview"
    )

    ov1,ov2,ov3,ov4 = st.columns(4)

    ov1.info("Model\n\nLogistic Regression")
    ov2.info("Training Samples\n\n5,634")
    ov3.info("Testing Samples\n\n1,409")
    ov4.info("Features\n\n6")

    st.markdown("---")

    # ============================================
    # MODEL COMPARISON
    # ============================================

    st.subheader(
        "Model Comparison"
    )

    comparison_df = pd.DataFrame({
        "Model":[
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Accuracy":[
            79.91,
            79.77,
            79.06
        ]
    })

    fig_compare = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Accuracy Comparison Across Models"
    )

    fig_compare.update_layout(
        height=450,
        showlegend=False
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # FEATURE IMPORTANCE
    # ============================================

    st.subheader(
        "Feature Importance"
    )

    try:

        feature_df = pd.read_csv(
            "feature_importance.csv"
        )

        fig_features = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance Ranking"
        )

        fig_features.update_layout(
            height=500
        )

        st.plotly_chart(
            fig_features,
            use_container_width=True
        )

    except:

        st.warning(
            "feature_importance.csv not found"
        )

    st.markdown("---")

    # ============================================
    # BUSINESS INTERPRETATION
    # ============================================

    st.subheader(
        "Business Interpretation"
    )

    st.markdown(
        f"""
### Key Findings

✅ Model Accuracy: **{accuracy*100:.1f}%**

✅ Precision: **{precision*100:.1f}%**

✅ Recall: **{recall*100:.1f}%**

✅ F1 Score: **{f1*100:.1f}%**

### Executive Summary

The Logistic Regression model was selected as the production model after evaluating multiple algorithms.

The model achieves approximately **80% accuracy** on unseen customer data and provides reliable churn risk predictions.

Customer tenure, monthly charges, service usage patterns and engagement metrics are the strongest drivers of churn behaviour.

This model can be used to:

- Identify high-risk customers
- Prioritize retention campaigns
- Protect recurring revenue
- Support customer success teams
- Improve customer lifetime value (CLTV)
"""
    )

    st.success(
        "Production Model Status: Active ✅"
    )


# ============================================
# ABOUT
# ============================================

if selected == "About":

    st.title("ℹ️ About GCC Telecom AI")

    st.caption(
        "AI-Powered Customer Churn Prediction and Revenue Protection Platform"
    )

    st.markdown("---")

    st.markdown("""
    ### Platform Overview

    GCC Telecom AI is an end-to-end customer intelligence platform designed to help telecom providers proactively identify churn risk, protect revenue, and improve customer retention.

    The platform combines Data Engineering, Machine Learning, Business Intelligence, and Generative AI to deliver actionable insights for business stakeholders.
    """)

    st.markdown("---")

    st.subheader("🚀 Core Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
        🎯 Customer Churn Prediction

        Predict churn probability using a machine learning model trained on telecom customer behavior.
        """)

        st.success("""
        📊 Customer Insights

        Analyze customer profiles, health scores, engagement levels, and service usage.
        """)

        st.success("""
        🛡️ Risk Segmentation

        Categorize customers into Low, Medium, and High Risk groups for targeted interventions.
        """)

    with col2:

        st.success("""
        💰 Revenue Protection

        Estimate revenue at risk and identify retention opportunities.
        """)

        st.success("""
        🤖 AI Copilot

        Generate AI-powered retention recommendations and executive summaries.
        """)

        st.success("""
        📈 Model Performance

        Monitor predictive model effectiveness and business impact.
        """)

    st.markdown("---")

    st.subheader("🏗️ Solution Architecture")

    st.markdown("""
    **Data Source**
    - Telecom Customer Dataset

    **Data Engineering**
    - Bronze Layer
    - Silver Layer
    - Gold Layer

    **Machine Learning**
    - Logistic Regression
    - Churn Prediction Engine

    **Business Intelligence**
    - Customer Segmentation
    - Revenue Risk Analysis

    **Generative AI**
    - Telecom AI Copilot
    """)

    st.markdown("---")

    st.subheader("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.info("""
        Data Engineering

        • Microsoft Fabric
        • OneLake
        • Lakehouse
        • Data Pipelines
        """)

    with tech2:

        st.info("""
        Analytics

        • Python
        • Pandas
        • Scikit-Learn
        • Plotly
        """)

    with tech3:

        st.info("""
        Application Layer

        • Streamlit
        • AI Copilot
        • SaaS Dashboard
        • Cloud Deployment
        """)

    st.markdown("---")

    st.subheader("📊 Business Impact")

    st.markdown("""
    The platform enables telecom operators to:

    - Reduce customer churn
    - Improve customer retention
    - Increase customer lifetime value (CLTV)
    - Prioritize high-risk customers
    - Protect recurring revenue streams
    - Support data-driven decision making
    """)

    st.markdown("---")

    st.subheader("👨‍💻 Project Information")

    st.info("""
    GCC Telecom Customer Intelligence Platform

    End-to-End Data Engineering, Machine Learning,
    Business Intelligence and AI Solution.

    Developed as a telecom analytics and customer retention platform.
    """)

    st.success(
        "Platform Status: Production Ready ✅"
    )


st.caption(
    """
GCC Telecom Customer Intelligence Platform

End-to-End Data Engineering + Analytics +
Machine Learning + Business Intelligence Project
"""
)

