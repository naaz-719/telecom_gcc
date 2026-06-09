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

.main {
    background-color: #F5F7FA;
}

.metric-card {
    background-color:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.05);
}

.sidebar-title{
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.image(tower_icon, width=80)

    st.markdown("## GCC Telecom AI")

    st.markdown("---")

    st.markdown("### Navigation")

    st.markdown("📊 Dashboard")
    st.markdown("👤 Customer Insights")
    st.markdown("⚠ Risk Intelligence")
    st.markdown("💰 Revenue Protection")
    st.markdown("🤖 AI Recommendations")
    st.markdown("📈 Model Performance")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

col1, col2 = st.columns([8,1])

with col1:

    st.title("GCC Telecom Customer Intelligence Platform")

    st.caption(
        "Predict churn probability and identify revenue risk using machine learning"
    )

with col2:
    st.image(tower_icon,width=70)

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

c1,c2,c3,c4 = st.columns(4)

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
        "Avg Health Score",
        f"{avg_health:.1f}"
    )

with c4:
    st.metric(
        "High Risk Customers",
        f"{high_risk:,}"
    )

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

st.subheader("Customer Profile")

p1,p2,p3,p4 = st.columns(4)

with p1:
    st.write("Country")
    st.info(customer["country"])

with p2:
    st.write("City")
    st.info(customer["city"])

with p3:
    st.write("Customer Type")
    st.info(customer["customer_type"])

with p4:
    st.write("Contract")
    st.info(customer["contract"])

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
# PREDICTION SECTION
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Prediction Result"
)

left,right = st.columns(2)

with left:

    st.metric(
        "Churn Probability",
        f"{churn_probability}%"
    )

    if predicted_risk == "High Risk":

        st.error(
            "Customer is likely to CHURN"
        )

    elif predicted_risk == "Medium Risk":

        st.warning(
            "Customer requires attention"
        )

    else:

        st.success(
            "Customer appears stable"
        )

with right:

    st.metric(
        "Risk Level",
        predicted_risk
    )

    st.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

# -------------------------------------------------
# CUSTOMER SUMMARY
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Customer Summary"
)

s1,s2,s3,s4 = st.columns(4)

with s1:

    st.metric(
        "Tenure",
        f"{int(customer['tenure_months'])} Months"
    )

with s2:

    st.metric(
        "Monthly Charge",
        f"${customer['monthly_charge']}"
    )

with s3:

    st.metric(
        "CLTV",
        f"${cltv:,.0f}"
    )

with s4:

    st.metric(
        "Health Score",
        customer["customer_health_score"]
    )

# -------------------------------------------------
# AI RECOMMENDATIONS
# -------------------------------------------------

st.markdown("---")

col1,col2 = st.columns(2)

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
        "Increase digital engagement through app promotions"
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

with col1:

    st.subheader(
        "🤖 AI Recommendation Engine"
    )

    st.success(
        "Recommended Action Plan"
    )

    for item in recommendations:

        st.write(
            "• " + item
        )

# -------------------------------------------------
# RISK DRIVERS
# -------------------------------------------------

with col2:

    st.subheader(
        "⚠ Key Factors Driving Churn"
    )

    if customer["complaint_count"] >= 3:

        st.error(
            "High Complaint Count"
        )

    if customer["payment_delay_days"] >= 10:

        st.error(
            "Payment Delays"
        )

    if customer["contract"] == "Month-to-month":

        st.warning(
            "Month-to-month Contract"
        )

    if customer["customer_health_score"] < 85:

        st.warning(
            "Low Health Score"
        )

    if customer["tenure_months"] < 12:

        st.info(
            "Short Customer Tenure"
        )

# -------------------------------------------------
# CUSTOMER VALUE INTELLIGENCE
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Customer Value Intelligence"
)

v1,v2,v3,v4 = st.columns(4)

with v1:

    st.metric(
        "Customer Value Segment",
        customer["customer_value_segment"]
    )

with v2:

    st.metric(
        "CLTV Segment",
        customer["cltv_segment"]
    )

with v3:

    st.metric(
        "Risk Segment",
        customer["risk_segment"]
    )

with v4:

    st.metric(
        "Tenure Segment",
        tenure_segment
    )

# -------------------------------------------------
# REVENUE PROTECTION
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Revenue Protection"
)

r1,r2,r3 = st.columns(3)

with r1:

    st.metric(
        "Customer CLTV",
        f"${cltv:,.0f}"
    )

with r2:

    st.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

with r3:

    retention_gain = round(
        revenue_at_risk * 0.60,
        2
    )

    st.metric(
        "Potential Revenue Saved",
        f"${retention_gain:,.0f}"
    )


# -------------------------------------------------
# AI RECOMMENDATIONS
# -------------------------------------------------

st.markdown("---")

col1,col2 = st.columns(2)

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
        "Increase digital engagement through app promotions"
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

with col1:

    st.subheader(
        "🤖 AI Recommendation Engine"
    )

    st.success(
        "Recommended Action Plan"
    )

    for item in recommendations:

        st.write(
            "• " + item
        )

# -------------------------------------------------
# RISK DRIVERS
# -------------------------------------------------

with col2:

    st.subheader(
        "⚠ Key Factors Driving Churn"
    )

    if customer["complaint_count"] >= 3:

        st.error(
            "High Complaint Count"
        )

    if customer["payment_delay_days"] >= 10:

        st.error(
            "Payment Delays"
        )

    if customer["contract"] == "Month-to-month":

        st.warning(
            "Month-to-month Contract"
        )

    if customer["customer_health_score"] < 85:

        st.warning(
            "Low Health Score"
        )

    if customer["tenure_months"] < 12:

        st.info(
            "Short Customer Tenure"
        )

# -------------------------------------------------
# CUSTOMER VALUE INTELLIGENCE
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Customer Value Intelligence"
)

v1,v2,v3,v4 = st.columns(4)

with v1:

    st.metric(
        "Customer Value Segment",
        customer["customer_value_segment"]
    )

with v2:

    st.metric(
        "CLTV Segment",
        customer["cltv_segment"]
    )

with v3:

    st.metric(
        "Risk Segment",
        customer["risk_segment"]
    )

with v4:

    st.metric(
        "Tenure Segment",
        tenure_segment
    )

# -------------------------------------------------
# REVENUE PROTECTION
# -------------------------------------------------

st.markdown("---")

st.subheader(
    "Revenue Protection"
)

r1,r2,r3 = st.columns(3)

with r1:

    st.metric(
        "Customer CLTV",
        f"${cltv:,.0f}"
    )

with r2:

    st.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

with r3:

    retention_gain = round(
        revenue_at_risk * 0.60,
        2
    )

    st.metric(
        "Potential Revenue Saved",
        f"${retention_gain:,.0f}"
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
    

