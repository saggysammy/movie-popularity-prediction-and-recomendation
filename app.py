import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🎬 Movie AI Pro", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.title {
    font-size: 40px;
    text-align: center;
    color: #00FFC6;
    font-weight: bold;
}
.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,255,198,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 AI Movie Dashboard</div>', unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Choose Section", ["Prediction", "Recommendation", "Analytics"])

# ================= PREDICTION =================
if mode == "Prediction":

    st.subheader("🔮 Predict Movie Popularity")

    col1, col2 = st.columns(2)

    with col1:
        vote_avg = st.slider("⭐ Vote Average", 0.0, 10.0, 5.0)

    with col2:
        vote_count = st.slider("🗳 Vote Count", 0, 50000, 1000)

    if st.button("🚀 Predict Now"):

        score = vote_avg * vote_count
        data = np.array([[vote_avg, vote_count, score]])
        data = scaler.transform(data)

        pred = model.predict(data)[0]

        # ---------- METRIC DISPLAY ----------
        col1, col2, col3 = st.columns(3)

        col1.metric("⭐ Rating", vote_avg)
        col2.metric("🗳 Votes", vote_count)
        col3.metric("🔥 Popularity", round(pred, 2))

        st.success("Prediction completed successfully!")

        # ---------- DOWNLOAD ----------
        report = pd.DataFrame({
            "Vote Average": [vote_avg],
            "Vote Count": [vote_count],
            "Predicted Popularity": [pred]
        })

        st.download_button("📥 Download Report", report.to_csv(index=False))

# ================= RECOMMENDATION =================
elif mode == "Recommendation":

    st.subheader("🎯 Movie Recommendation Engine")

    movie_list = movies["title"].values
    selected_movie = st.selectbox("🔍 Search Movie", movie_list)

    def recommend(movie):
        index = movies[movies["title"] == movie].index[0]
        distances = similarity[index]

        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        return [movies.iloc[i[0]].title for i in movie_list]

    if st.button("🎬 Get Recommendations"):

        results = recommend(selected_movie)

        st.subheader("🎥 Top 5 Movies")

        cols = st.columns(5)

        for i in range(5):
            with cols[i]:
                st.markdown(f"""
                <div class="card">
                    🎬 <br><br> {results[i]}
                </div>
                """, unsafe_allow_html=True)

        # Download
        df_rec = pd.DataFrame({"Recommended Movies": results})
        st.download_button("📥 Download List", df_rec.to_csv(index=False))

# ================= ANALYTICS =================
elif mode == "Analytics":

    st.subheader("📊 Data Insights")

    df = pd.read_csv("tmdb_top_10k_movies_2026.csv.csv")

    col1, col2 = st.columns(2)

    col1.metric("Total Movies", len(df))
    col2.metric("Average Popularity", round(df["popularity"].mean(), 2))

    st.subheader("📈 Popularity Trend")
    st.line_chart(df["popularity"].head(200))

    st.subheader("📊 Vote vs Popularity")
    st.scatter_chart(df[["vote_count", "popularity"]])

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>🚀 Built with ML + NLP + Streamlit</center>", unsafe_allow_html=True)