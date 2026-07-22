import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Enterprise ATS Control Center", layout="wide", page_icon="🛡️")
BACKEND_URL = "http://127.0.0.1:8000/api/v1/match"

st.title("🛡️ Enterprise Next-Gen ATS Screening & Analytics Dashboard")
st.markdown("Stateless Multi-Layered Sourcing Engine powered by **Async Parallel NLP Architecture**.")

st.sidebar.header("🎛️ Hybrid Matching Calibration")
w_embed = st.sidebar.slider("Embedding (Semantic Weight):", 0.0, 1.0, 0.6, step=0.1)

jd_input = st.text_area("💼 Core Corporate Requirements (Job Description):", height=150)
uploaded_files = st.file_uploader("📥 Intake Pipeline Channels (PDF, DOCX):", accept_multiple_files=True, type=["pdf", "docx"])

if st.button("🚀 Execute Enterprise Batch Extraction") and uploaded_files and jd_input:
    form_data = {"job_description": jd_input, "weight_embedding": str(w_embed)}
    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
    
    with st.spinner("Invoking background concurrency processing..."):
        try:
            res = requests.post(BACKEND_URL, data=form_data, files=files_payload)
            response_data = res.json()
        except Exception:
            st.error("❌ Infrastructure Ping Failure: Sourcing Services Offline.")
            st.stop()
            
    if res.status_code == 200:
        leaderboard = response_data["leaderboard"]
        perf = response_data["performance_summary"]
        
        records = []
        missing_skills_pool = []
        for item in leaderboard:
            p = item["profile"]
            e = item["explainability"]
            records.append({
                "Candidate": p["name"],
                "Final Score (%)": item["final_hybrid_score"],
                "Semantic Score": item["embedding_score"],
                "Keyword Score": item["tfidf_score"],
                "Experience": p["experience_years"],
                "Education": p["education_tier"],
                "Recommendation": item["recommendation"],
                "Email": p["email"],
                "LinkedIn": p["linkedin"],
                "Matched": e["matched_skills"],
                "Missing": e["missing_skills"]
            })
            missing_skills_pool.extend(e["missing_skills"])
            
        df = pd.DataFrame(records)
        
        # --- 1. System Performance & Sourcing Metrics ---
        st.markdown("### 📊 Operational & Infrastructure Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unique Profiles Evaluated", response_data["total_processed"])
        m2.metric("Duplicate Files Isolated", response_data["duplicates_detected"])
        m3.metric("Pipeline Batch Latency", f"{perf['processing_time_seconds']}s")
        throughput = round(response_data["total_processed"] / perf['processing_time_seconds'], 2) if perf['processing_time_seconds'] else 0.0
        m4.metric("Throughput (Files/Sec)", f"{throughput} f/s")
        
        st.markdown("---")
        
        # --- 2. Advanced Analytics Visualization Grid ---
        g1, g2, g3 = st.columns(3)
        with g1:
            st.subheader("🎯 Match Score Distribution Histogram")
            fig_hist = px.histogram(df, x="Final Score (%)", nbins=10, color="Recommendation", color_discrete_map={'🟢 Highly Recommended':'green', '🟡 Consider for Review':'orange', '🔴 Rejected':'red'})
            st.plotly_chart(fig_hist, use_container_width=True)
        with g2:
            st.subheader("🎓 Educational Distribution Tiers")
            fig_pie = px.pie(df, names="Education", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with g3:
            st.subheader("📉 Missing Target Skills Analytics")
            if missing_skills_pool:
                miss_df = pd.Series(missing_skills_pool).value_counts().reset_index()
                miss_df.columns = ["Skill", "Frequency"]
                fig_miss = px.bar(miss_df, x="Frequency", y="Skill", orientation='h', color="Frequency", color_continuous_scale="Reds")
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.info("Zero missing skills across the pool.")
                
        st.markdown("---")
        
        # --- 3. Individual Candidate Audit Explorer ---
        st.subheader("🔍 Local Profile Audit & Explainability Explorer")
        chosen_cand = st.selectbox("Select Candidate to Audit:", df["Candidate"].unique())
        c_row = df[df["Candidate"] == chosen_cand].iloc[0]
        
        aud1, aud2 = st.columns(2)
        with aud1:
            st.markdown(f"""
            - **Email:** `{c_row['Email']}`
            - **LinkedIn:** `{c_row['LinkedIn']}`
            - **Experience:** `{c_row['Experience']} Years`
            - **Education:** `{c_row['Education']}`
            """)
        with aud2:
            st.markdown(f"""
            - **Semantic Vector Match Score:** `{c_row['Semantic Score']}`
            - **TF-IDF Keyword Match Score:** `{c_row['Keyword Score']}`
            - **Verified Matched Skills:** `{', '.join(c_row['Matched'])}`
            - **Critical Missing Skills:** `{', '.join(c_row['Missing'])}`
            """)
            
        st.markdown("---")
        st.subheader("📋 Core Enterprise ATS Talent Ledger")
        st.dataframe(df.drop(columns=["Matched", "Missing", "Email", "LinkedIn"]), use_container_width=True)
        
        # ميزة التصدير للـ Export Layer
        st.download_button("📥 Export Results Ledger to CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="ATS_Sourcing_Report.csv", mime="text/csv")