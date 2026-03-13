import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title = "ChurnLens . AI Retention Dashboard",
    page_icon = "🔮",
    layout = "wide",
    initial_sidebar_state = "expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; }

.section-title {
    font-size: 13px;
    font-weight: 500;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px 0 14px 0;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)
@st.cache_data
def generate_data(n=2000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "customer_id":           [f"CUST-{i:04d}" for i in range(n)],
        "days_since_last_order": np.random.exponential(45, n).astype(int).clip(1, 365),
        "total_orders":          np.random.poisson(6, n).clip(1, 50),
        "avg_order_value":       np.random.lognormal(4.2, 0.7, n).round(2),
        "support_tickets":       np.random.poisson(0.8, n).clip(0, 10),
        "returns_count":         np.random.poisson(0.4, n).clip(0, 8),
        "email_open_rate":       np.random.beta(2, 5, n).round(3),
        "days_since_signup":     np.random.randint(30, 1200, n),
        "promo_used_count":      np.random.poisson(2, n).clip(0, 15),
        "country": np.random.choice(
            ["US","UK","CA","AU","DE","FR","Other"], n,
            p=[0.35,0.18,0.12,0.08,0.07,0.07,0.13]
        ),
        "acquisition_channel": np.random.choice(
            ["Organic","Paid Social","Email","Referral","Influencer"], n,
            p=[0.30,0.25,0.20,0.15,0.10]
        ),
    })
    df["total_spend"] = (df["total_orders"] * df["avg_order_value"]).round(2)

    churn_score = (
        0.25 * (df["days_since_last_order"] / 365) +
        0.20 * (1 - df["email_open_rate"]) +
        0.15 * (df["support_tickets"] / 10) +
        0.15 * (df["returns_count"] / 8) +
        0.10 * (1 / (df["total_orders"] + 1)) +
        0.15 * np.random.uniform(0, 1, n)
    )
    df["churned"] = (churn_score > 0.42).astype(int)
    return df
@st.cache_data
def train_model(df):
    features = [
        "days_since_last_order", "total_orders", "avg_order_value",
        "total_spend", "support_tickets", "returns_count",
        "email_open_rate", "days_since_signup", "promo_used_count"
    ]
    le_country = LabelEncoder()
    le_channel = LabelEncoder()
    df = df.copy()
    df["country_enc"] = le_country.fit_transform(df["country"])
    df["channel_enc"] = le_channel.fit_transform(df["acquisition_channel"])
    features += ["country_enc", "channel_enc"]

    X = df[features]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=120, max_depth=4,
        learning_rate=0.08, random_state=42
    )
    model.fit(X_train, y_train)

    proba     = model.predict_proba(X_test)[:, 1]
    auc       = roc_auc_score(y_test, proba)

    df["churn_probability"] = model.predict_proba(df[features])[:, 1]
    df["risk_tier"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return df, model, auc, importances, features
# Run both functions — cached after first run
df_raw               = generate_data()
df, model, auc_score, feature_importances, features = train_model(df_raw)

with st.sidebar:
    st.markdown("## 🔮 ChurnLens")
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-top:-8px;'>AI Retention Intelligence</p>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**Filter customers**")

    risk_filter = st.multiselect(
        "Risk tier",
        ["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"]
    )
    country_filter = st.multiselect(
        "Country",
        sorted(df["country"].unique()),
        default=sorted(df["country"].unique())
    )
    channel_filter = st.multiselect(
        "Acquisition channel",
        sorted(df["acquisition_channel"].unique()),
        default=sorted(df["acquisition_channel"].unique())
    )
    min_spend = st.slider("Min total spend ($)", 0, 2000, 0, step=50)

    st.divider()
    st.markdown(f"<p style='color:#6b7280;font-size:12px;'>Model AUC-ROC: "
                f"<strong style='color:#10b981'>{auc_score:.3f}</strong></p>",
                unsafe_allow_html=True)
    
    mask = (
    df["risk_tier"].isin(risk_filter) &
    df["country"].isin(country_filter) &
    df["acquisition_channel"].isin(channel_filter) &
    (df["total_spend"] >= min_spend)
)
df_filt = df[mask]

st.markdown("## Customer Churn Intelligence")
st.markdown(
    f"<p style='color:#6b7280;margin-top:-10px;'>Showing {len(df_filt):,} of "
    f"{len(df):,} customers</p>",
    unsafe_allow_html=True
)

# Five KPI cards across the top
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Churn rate",
          f"{df_filt['churned'].mean()*100:.1f}%")

k2.metric("High-risk customers",
          f"{(df_filt['risk_tier']=='High Risk').sum():,}")

k3.metric("Avg customer LTV",
          f"${df_filt['total_spend'].mean():,.0f}")

k4.metric("Revenue at risk",
          f"${df_filt.loc[df_filt['risk_tier']=='High Risk','total_spend'].sum():,.0f}")

k5.metric("Avg churn probability",
          f"{df_filt['churn_probability'].mean()*100:.1f}%")

st.markdown("---")

color_map = {
    "High Risk": "#ef4444",
    "Medium Risk": "#f59e0b",
    "Low Risk": "#10b981"
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>Risk tier distribution</div>",
                unsafe_allow_html=True)
    risk_counts = df_filt["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["tier", "count"]
    fig = px.bar(
        risk_counts, x="tier", y="count",
        color="tier", color_discrete_map=color_map,
        template="plotly_dark"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=260
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div class='section-title'>Churn rate by acquisition channel</div>",
                unsafe_allow_html=True)
    ch_churn = (df_filt.groupby("acquisition_channel")["churn_probability"]
                .mean().sort_values(ascending=True).reset_index())
    fig2 = px.bar(
        ch_churn, x="churn_probability", y="acquisition_channel",
        orientation="h", template="plotly_dark",
        color="churn_probability",
        color_continuous_scale=["#10b981","#f59e0b","#ef4444"]
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=260
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
st.markdown("<div class='section-title'>Top churn drivers — what is causing customers to leave</div>",
            unsafe_allow_html=True)

label_map = {
    "days_since_last_order": "Days since last order",
    "email_open_rate":       "Email open rate",
    "support_tickets":       "Support tickets",
    "returns_count":         "Returns count",
    "total_orders":          "Total orders",
    "avg_order_value":       "Avg order value",
    "total_spend":           "Total spend",
    "promo_used_count":      "Promos used",
    "days_since_signup":     "Account age",
    "country_enc":           "Country",
    "channel_enc":           "Acq. channel"
}

top_fi = feature_importances.head(7).reset_index()
top_fi.columns = ["feature", "importance"]
top_fi["feature"] = top_fi["feature"].map(label_map).fillna(top_fi["feature"])

fig3 = px.bar(
    top_fi.sort_values("importance"), x="importance", y="feature",
    orientation="h", template="plotly_dark",
    color="importance",
    color_continuous_scale=["#1e3a5f","#378ADD","#7EC8E3"]
)
fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=10, b=0),
    height=300
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("<div class='section-title'>High-risk customers — prioritise for retention</div>",
            unsafe_allow_html=True)

hr_table = (df_filt[df_filt["risk_tier"] == "High Risk"][[
    "customer_id", "churn_probability", "total_spend",
    "days_since_last_order", "email_open_rate",
    "support_tickets", "acquisition_channel", "country"
]].sort_values("churn_probability", ascending=False).head(50).copy())

hr_table["churn_probability"] = (hr_table["churn_probability"]*100).round(1).astype(str) + "%"
hr_table["email_open_rate"]   = (hr_table["email_open_rate"]*100).round(1).astype(str) + "%"
hr_table["total_spend"]       = "$" + hr_table["total_spend"].round(0).astype(int).astype(str)
hr_table.columns = ["Customer ID","Churn %","LTV","Days Inactive","Email Open %","Tickets","Channel","Country"]

st.dataframe(hr_table, use_container_width=True, height=320, hide_index=True)

st.markdown("---")
st.markdown("### Predict churn for a single customer")

pc1, pc2, pc3 = st.columns(3)

with pc1:
    new_days    = st.number_input("Days since last order", 1, 365, 45)
    new_orders  = st.number_input("Total orders", 1, 100, 5)
    new_aov     = st.number_input("Avg order value ($)", 10.0, 2000.0, 85.0, step=5.0)

with pc2:
    new_tickets = st.number_input("Support tickets", 0, 20, 1)
    new_returns = st.number_input("Returns count", 0, 10, 0)
    new_open    = st.slider("Email open rate", 0.0, 1.0, 0.25, step=0.01)

with pc3:
    new_signup  = st.number_input("Days since signup", 30, 2000, 300)
    new_promos  = st.number_input("Promos used", 0, 20, 2)
    new_country = st.selectbox("Country", ["US","UK","CA","AU","DE","FR","Other"])
    new_channel = st.selectbox("Channel", ["Organic","Paid Social","Email","Referral","Influencer"])

if st.button("Predict churn probability →", type="primary"):
    country_map = {"AU":0,"CA":1,"DE":2,"FR":3,"Other":4,"UK":5,"US":6}
    channel_map = {"Email":0,"Influencer":1,"Organic":2,"Paid Social":3,"Referral":4}

    input_data = [[
        new_days, new_orders, new_aov,
        new_orders * new_aov,
        new_tickets, new_returns, new_open,
        new_signup, new_promos,
        country_map.get(new_country, 4),
        channel_map.get(new_channel, 2)
    ]]

    prob = model.predict_proba(input_data)[0][1]
    tier = "High Risk" if prob > 0.65 else ("Medium Risk" if prob > 0.35 else "Low Risk")
    color = "#ef4444" if tier == "High Risk" else ("#f59e0b" if tier == "Medium Risk" else "#10b981")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:#1a1d27;border:2px solid {color};border-radius:12px;
                    padding:24px;text-align:center;margin-top:12px;'>
            <div style='font-size:48px;font-weight:600;color:{color};'>{prob*100:.1f}%</div>
            <div style='font-size:14px;color:{color};margin-top:4px;'>{tier}</div>
            <div style='font-size:12px;color:#6b7280;margin-top:8px;'>churn probability</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        if tier == "High Risk":
            recs = [
                "Send personalised win-back email within 24 hrs",
                "Offer 15–20% discount on next purchase",
                "Assign to VIP support queue",
                "Trigger SMS with exclusive offer"
            ]
        elif tier == "Medium Risk":
            recs = [
                "Add to re-engagement email sequence",
                "Show targeted product recommendations",
                "Offer loyalty points incentive"
            ]
        else:
            recs = [
                "Continue standard nurture flow",
                "Great candidate for referral program",
                "Consider upsell campaign"
            ]
        st.markdown(f"**Recommended actions — {tier}:**")
        for rec in recs:
            st.markdown(f"- {rec}")

st.markdown("---")
st.markdown("<p style='color:#374151;font-size:12px;text-align:center;'>"
            "ChurnLens · Gradient Boosting · Demo data · "
            "Replace generate_data() with your real CSV</p>",
            unsafe_allow_html=True)