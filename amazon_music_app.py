import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import base64


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>

    /* Reduce top and side padding of main page */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CSS FOR METRIC
# ---------------------------------------------------------
st.markdown("""
<style>

[data-testid="stMetricLabel"] {
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    font-size: 22px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER IMAGE
# ---------------------------------------------------------
from pathlib import Path
header_path = Path("images/music_header.png")

if header_path.exists():
    st.image(
        str(header_path),
        width="stretch"
    )
else:
    st.error("Header image not found!")

# ============================================================
# BACKGROUND IMAGE
# ============================================================
import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, 
                #d4d8dc 0%,      /* smoky gray top */
                #c9ccd4 25%,     
                #d8ccd0 50%,     
                #e0b8bc 75%,     /* transitioning to pink */
                #d99a9e 100%     /* rose/pink bottom */
            );
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        .block-container {
            padding-top: 2rem;
        }

        div[data-testid="stMetric"], 
        div[data-testid="stDataFrame"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)
# ============================================================
# CUSTOM CSS
# ============================================================

# ---------------------------------------------------------
# SIDEBAR STYLING
# ---------------------------------------------------------
st.markdown("""
<style>

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #090f1f 0%,
            #111936 55%,
            #1b123d 100%
        );
    }

    /* Sidebar overall spacing */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Music Explorer title */
    .sidebar-title {
        font-size: 25px;
        font-weight: 750;
        color: white;
        margin-bottom: 2px;
    }

    /* Subtitle */
    .sidebar-subtitle {
        font-size: 11px;
        color: #a9b0c8;
        letter-spacing: 1.2px;
        margin-left: 4px;
        margin-bottom: 22px;
    }

    /* Divider */
    .sidebar-line {
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            #4b4f75,
            transparent
        );
        margin-bottom: 22px;
    }

    /* Navigation heading */
    [data-testid="stSidebar"] .stRadio > label {
        color: #c8cce0 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin-bottom: 8px;
    }

    /* Navigation options */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 7px;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.035);
        border-radius: 12px;
        padding: 9px 10px;
        transition: all 0.2s ease;
    }

    /* Hover */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(151, 82, 255, 0.16);
        border-color: rgba(151, 82, 255, 0.35);
        transform: translateX(3px);
    }

    /* Navigation text */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: #eef0fa !important;
        font-size: 14px !important;
    }

    /* Footer */
    .sidebar-footer {
        position: fixed;
        bottom: 18px;
        left: 20px;
        color: #858ca8;
        font-size: 11px;
        line-height: 1.6;
    }

    .sidebar-footer span {
        color: #b66cff;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/Amazon_Music_Clustered_Final.csv")

    return df


df = load_data()


# ============================================================
# FEATURE LIST
# ============================================================

key_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

available_features = [
    col for col in key_features
    if col in df.columns
]


# ============================================================
# IDENTIFY SONG / ARTIST / POPULARITY COLUMNS
# ============================================================

song_column = next(
    (
        col for col in
        ["name_song", "track_name", "song_name"]
        if col in df.columns
    ),
    None
)

artist_column = next(
    (
        col for col in
        ["name_artists", "artist_name", "artist"]
        if col in df.columns
    ),
    None
)

popularity_column = next(
    (
        col for col in
        [
            "popularity_songs",
            "popularity",
            "track_popularity"
        ]
        if col in df.columns
    ),
    None
)


# ============================================================
# CHECK REQUIRED CLUSTER COLUMN
# ============================================================

if "cluster" not in df.columns:

    st.error(
        "❌ 'cluster' column is missing from the final CSV."
    )

    st.stop()


# ============================================================
# CLUSTER NAME MAPPING
# ============================================================

if "Cluster_Name" in df.columns:

    cluster_name_map = (
        df[["cluster", "Cluster_Name"]]
        .drop_duplicates()
        .sort_values("cluster")
        .set_index("cluster")["Cluster_Name"]
        .to_dict()
    )

else:

    cluster_name_map = {
        cluster: f"Cluster {cluster}"
        for cluster in sorted(df["cluster"].unique())
    }

st.markdown("<br>", unsafe_allow_html=True)
# ============================================================
# HEADER
# ============================================================

# st.markdown(
#     '<div class="dashboard-title">'
#     '🎵 Amazon Music Clustering '
#     '</div>',
#     unsafe_allow_html=True
# )


# st.markdown(
#     '<div class="dashboard-subtitle">'
#     'Discovering music profiles using K-Means clustering'
#     '</div>',
#     unsafe_allow_html=True
# )

# st.markdown("---")
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">🎵 Music Explorer</div>

        <div class="sidebar-subtitle">
            AMAZON MUSIC • ML DASHBOARD
        </div>

        <div class="sidebar-line"></div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🎯 Cluster Profiles",
            "📊 Visualizations",
            "🎶 Top Tracks",
            "🔎 Song Explorer"
        ],
        label_visibility="visible"
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            <span>●</span> K-Means Clustering<br>
            <span>♪</span> Music Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
    "<h2 style='font-size:28px;'>📌 Dataset Overview</h2>",
    unsafe_allow_html=True
    )

    total_songs = len(df)

    total_clusters = df["cluster"].nunique()

    avg_popularity = (
        df[popularity_column].mean()
        if popularity_column
        else 0
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎵 Total Songs",
        f"{total_songs:,}"
    )

    c2.metric(
        "🎯 Clusters",
        total_clusters
    )

    if popularity_column:

        c3.metric(
            "⭐ Avg Popularity",
            f"{avg_popularity:.2f}"
        )

    if "danceability" in df.columns:

        c4.metric(
            "💃 Avg Danceability",
            f"{df['danceability'].mean():.2f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
    "<h2 style='font-size:28px;'>🎯 Cluster Distribution</h2>",
    unsafe_allow_html=True
    )

    cluster_counts = (
        df.groupby("cluster")
        .size()
        .reset_index(name="Song_Count")
    )

    cluster_counts["Cluster_Name"] = (
        cluster_counts["cluster"]
        .map(cluster_name_map)
    )

    cluster_counts["Percentage"] = (
        cluster_counts["Song_Count"]
        / len(df) * 100
    ).round(2)


    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            data=cluster_counts,
            x="Cluster_Name",
            y="Song_Count",
            ax=ax,
            palette={
            "Chill Acoustic & Instrumental": "#7DCD9E",
            "Energetic Party & Pop Tracks": "#7DB5E0",
            "Speech-Heavy / Spoken Tracks": "#D379A4"
    },
        )

        ax.set_title(
            "Number of Songs per Cluster"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Number of Songs")

        plt.xticks(rotation=20)

        plt.tight_layout()

        st.pyplot(fig)


    with col2:

        st.dataframe(
            cluster_counts[
                [
                    "cluster",
                    "Cluster_Name",
                    "Song_Count",
                    "Percentage"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CLUSTER DESCRIPTION
    # --------------------------------------------------------

    st.subheader("🎼 Cluster Profiles")

    descriptions = {
        "Chill Acoustic & Instrumental":
            "High acousticness with lower energy and danceability."
            "Cluster 0 is characterized by low energy, lower danceability, "
            "lower loudness and high acousticness. The cluster has an average "
            "acousticness of approximately 0.74 and energy of 0.32, suggesting "
            "relatively calm and acoustic-oriented tracks",

        "Energetic Party & Pop Tracks":
            "High energy, loudness, valence and danceability."
            "Cluster 1 contains tracks with high energy, higher loudness, "
            "danceability and valence. It also has the highest average tempo among "
            "the three clusters. With energy around 0.70 and valence around 0.67, "
            "this cluster represents energetic and upbeat tracks.",

        "Speech-Heavy / Spoken Tracks":
            "Very high speechiness and liveness with shorter duration."
            "Cluster 2 is strongly distinguished by its very high speechiness, "
            "with an average around 0.67, compared with approximately 0.05 and 0.07 "
            "in Clusters 0 and 1. It also has higher liveness and considerably "
            "shorter average duration, making it the most distinctive "
            "speech-heavy/spoken-content cluster."
    }


    for cluster, name in cluster_name_map.items():

        description = descriptions.get(
            name,
            "Audio profile discovered by K-Means clustering."
        )

        st.markdown(
            f"""
            <div class="cluster-card">
                <h5>Cluster {cluster}: {name}</h5>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 2 — CLUSTER PROFILES
# ============================================================

elif page == "🎯 Cluster Profiles":

    st.header("🎯 Cluster Profiles")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["cluster"].unique()),
        format_func=lambda x:
            f"Cluster {x} — {cluster_name_map[x]}"
    )


    cluster_df = df[
        df["cluster"] == selected_cluster
    ].copy()


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Cluster",
        selected_cluster
    )

    c2.metric(
        "Music Profile",
        cluster_name_map[selected_cluster]
    )

    c3.metric(
        "Number of Songs",
        f"{len(cluster_df):,}"
    )


    st.markdown("---")


    # --------------------------------------------------------
    # ACTUAL FEATURE PROFILE
    # --------------------------------------------------------

    st.subheader("📋 Actual Audio Feature Profile")

    profile = (
        cluster_df[available_features]
        .mean()
        .round(3)
        .reset_index()
    )

    profile.columns = [
        "Feature",
        "Average Value"
    ]

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    st.subheader("📊 Average Audio Features")


    # Features with values roughly between 0 and 1
    small_scale_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence"
    ]

    small_scale_features = [
        col
        for col in small_scale_features
        if col in cluster_df.columns
    ]


    profile_small = (
        cluster_df[small_scale_features]
        .mean()
        .reset_index()
    )

    profile_small.columns = [
        "Feature",
        "Average Value"
    ]


    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    sns.barplot(
        data=profile_small,
        x="Feature",
        y="Average Value",
        ax=ax,
        color='#2E6F7E',
    )

    ax.set_title(
        f"Average Audio Features — "
        f"{cluster_name_map[selected_cluster]}"
    )

    ax.set_xlabel("Audio Feature")
    ax.set_ylabel("Average Value")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# PAGE 3 — VISUALIZATIONS
# ============================================================

elif page == "📊 Visualizations":

    st.header("📊 Cluster Visualizations")

    st.write(
        "The following visualizations were generated during "
        "Exploratory Data Analysis (EDA) and are presented "
        "here for cluster interpretation."
    )

    # ========================================================
    # 1. AVERAGE AUDIO FEATURES
    # ========================================================

    st.subheader("📊 Average Audio Features")

    st.image(
        "images/average_audio_features.png",
        use_container_width=True
    )

    st.caption(
        "Average audio feature values across the identified clusters."
    )


    # ========================================================
    # 2. STANDARDIZED HEATMAP
    # ========================================================

    st.subheader("🔥 Standardized Audio Feature Profiles")

    st.image(
        "images/standardized_heatmap.png",
        use_container_width=True
    )

    st.caption(
        "Standardized feature values allow comparison of "
        "audio characteristics across clusters."
    )


    # ========================================================
    # 3. PCA CLUSTER VISUALIZATION
    # ========================================================

    st.subheader("🔵 PCA 2D Cluster Visualization")

    st.image(
        "images/pca_clusters.png",
        use_container_width=True
    )

    st.caption(
        "Two-dimensional PCA representation of the music clusters."
    )

    # ========================================================
    # 4. DANCEABILITY DISTRIBUTION
    # ========================================================

    st.subheader("💃 Danceability Distribution by Cluster")

    st.image(
        "images/danceability_distribution.png",
        use_container_width=True
    )

    st.caption(
        "Distribution of danceability values across the clusters."
    )



    # ========================================================
    # 5. OPTIONAL OTHER EDA PLOTS
    # ========================================================

    st.subheader("📈 Other Feature Distributions")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "images/energy_distribution.png",
            caption="Energy Distribution by Cluster",
            use_container_width=True
        )

    with col2:

        st.image(
            "images/acousticness_distribution.png",
            caption="Acousticness Distribution by Cluster",
            use_container_width=True
        )

# ============================================================
# PAGE 4 — TOP TRACKS
# ============================================================

elif page == "🎶 Top Tracks":

    st.header("🎶 Top Tracks by Cluster")


    if song_column is None:

        st.warning(
            "Song name column is not available in your final CSV."
        )

        st.info(
            "Add the original song-name column to your "
            "final dataset if you want to display track names."
        )

    else:

        selected_cluster = st.selectbox(
            "Select Cluster",
            sorted(df["cluster"].unique()),
            format_func=lambda x:
                f"Cluster {x} — {cluster_name_map[x]}"
        )


        top_n = st.slider(
            "Number of tracks",
            5,
            50,
            10
        )


        cluster_df = df[
            df["cluster"] == selected_cluster
        ].copy()


        # Sort by popularity
        if popularity_column:

            top_tracks = (
                cluster_df
                .sort_values(
                    popularity_column,
                    ascending=False
                )
                .head(top_n)
            )

        else:

            top_tracks = cluster_df.head(top_n)


        # Columns to display
        display_columns = []

        if song_column:
            display_columns.append(song_column)

        if artist_column:
            display_columns.append(artist_column)

        if popularity_column:
            display_columns.append(popularity_column)

        display_columns.append("Cluster_Name")


        display_columns += [
            col
            for col in [
                "danceability",
                "energy",
                "speechiness",
                "acousticness",
                "valence"
            ]
            if col in df.columns
        ]


        st.dataframe(
            top_tracks[display_columns],
            use_container_width=True,
            hide_index=True
        )


        # Download
        csv = top_tracks.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "📥 Download Top Tracks",
            data=csv,
            file_name="top_tracks_by_cluster.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 5 — SONG EXPLORER
# ============================================================

elif page == "🔎 Song Explorer":

    st.header("🔎 Explore Songs")


    filtered_df = df.copy()


    # --------------------------------------------------------
    # CLUSTER FILTER
    # --------------------------------------------------------

    cluster_options = [
        "All Clusters"
    ] + sorted(df["cluster"].unique().tolist())


    selected_cluster = st.selectbox(
        "Filter by Cluster",
        cluster_options,
        format_func=lambda x:
            (
                x
                if x == "All Clusters"
                else
                f"Cluster {x} — {cluster_name_map[x]}"
            )
    )


    if selected_cluster != "All Clusters":

        filtered_df = filtered_df[
            filtered_df["cluster"] == selected_cluster
        ]


    # --------------------------------------------------------
    # SEARCH SONG
    # --------------------------------------------------------

    if song_column:

        search_text = st.text_input(
            "🔎 Search Song"
        )

        if search_text:

            filtered_df = filtered_df[
                filtered_df[song_column]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]


    # --------------------------------------------------------
    # POPULARITY FILTER
    # --------------------------------------------------------

    if popularity_column:

        min_popularity = st.slider(
            "Minimum Popularity",
            int(df[popularity_column].min()),
            int(df[popularity_column].max()),
            int(df[popularity_column].min())
        )

        filtered_df = filtered_df[
            filtered_df[popularity_column]
            >= min_popularity
        ]


    st.write(
        f"Showing **{len(filtered_df):,} songs**"
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_columns = []

    for col in [
        song_column,
        artist_column,
        popularity_column,
        "cluster",
        "Cluster_Name",
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "valence"
    ]:

        if col and col in filtered_df.columns:

            display_columns.append(col)


    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_music_clusters.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎵 Amazon Music Clustering | "
    "K-Means based Music Profiling"
)