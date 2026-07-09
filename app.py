"""
Painel de Sobreposição de Disciplinas USP
Analisa disciplinas oferecidas em inglês e mapeia sobreposição temática.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Disciplinas USP | LABDAPS",
    page_icon="🎓",
    layout="wide",
)

# ── estilos ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #223886;
    line-height: 1.1;
}
.kpi-label {
    font-size: .8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: 4px;
}
.section-tag {
    display: inline-block;
    background: #eef1fb;
    color: #223886;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
}
.disc-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #223886;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.disc-name { font-weight: 600; color: #0f1730; font-size: .95rem; }
.disc-meta { font-size: .78rem; color: #6b7280; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── dados ─────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "disciplinas.json"

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["School"].str.match(r"^[A-Z]{2,8}$", na=False)].copy()
    df["credits_num"] = pd.to_numeric(df["Number of credits"], errors="coerce")
    df["texto"] = (
        df["Name"].fillna("") + " " +
        df["Objectives"].fillna("") + " " +
        df["Content"].fillna("")
    )
    return df.reset_index(drop=True)

@st.cache_data
def calcular_similaridade(df: pd.DataFrame, n_clusters: int = 8):
    vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(df["texto"])
    X_norm = normalize(X)
    sim = cosine_similarity(X_norm)

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_norm)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)

    # nome do cluster = termo mais frequente
    tfidf_dense = X_norm.toarray()
    feature_names = vec.get_feature_names_out()
    cluster_names = {}
    for c in range(n_clusters):
        mask = labels == c
        mean_tfidf = tfidf_dense[mask].mean(axis=0)
        top_terms = [feature_names[i] for i in mean_tfidf.argsort()[-3:][::-1]]
        cluster_names[c] = " / ".join(top_terms)

    return sim, coords, labels, cluster_names, vec

df = carregar_dados()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("🎓 **Disciplinas USP**")
    st.markdown("---")
    st.markdown('<span class="section-tag">Filtros</span>', unsafe_allow_html=True)

    escolas_disp = sorted(df["School"].unique())
    escolas_sel = st.multiselect("Escola", escolas_disp, placeholder="Todas")

    n_clusters = st.slider("Número de clusters temáticos", 3, 15, 8)
    top_n = st.slider("Top disciplinas similares", 3, 10, 5)

    st.markdown("---")
    st.markdown('<span class="section-tag">Sobre</span>', unsafe_allow_html=True)
    st.caption(
        "Disciplinas ofertadas em inglês pela USP (Janus). "
        "Sobreposição calculada por TF-IDF + similaridade de cosseno."
    )

df_filt = df[df["School"].isin(escolas_sel)] if escolas_sel else df

sim_full, coords_full, labels_full, cluster_names_full, vec = calcular_similaridade(df, n_clusters)

# índices do df_filt dentro do df completo
idx_filt = df_filt.index.tolist()
sim = sim_full[np.ix_(idx_filt, idx_filt)]
coords = coords_full[idx_filt]
labels = labels_full[idx_filt]
cluster_names = cluster_names_full

# ── cabeçalho ─────────────────────────────────────────────────────────────────

st.markdown("## 🎓 Sobreposição de Disciplinas USP")
st.caption("Disciplinas de pós-graduação ofertadas em inglês. Análise de similaridade temática por TF-IDF.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_filt)}</div><div class="kpi-label">Disciplinas</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df_filt["School"].nunique()}</div><div class="kpi-label">Escolas</div></div>', unsafe_allow_html=True)
with c3:
    media_sim = sim[np.triu_indices_from(sim, k=1)].mean()
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{media_sim:.2f}</div><div class="kpi-label">Sim. média</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{n_clusters}</div><div class="kpi-label">Clusters</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── abas ──────────────────────────────────────────────────────────────────────

tab_mapa, tab_heatmap, tab_busca, tab_dados = st.tabs(
    ["🗺 Mapa Temático", "🔥 Heatmap de Sobreposição", "🔍 Buscar Similares", "📋 Dados"]
)

# ── aba 1: mapa temático ──────────────────────────────────────────────────────

with tab_mapa:
    st.markdown('<span class="section-tag">Mapa 2D por tópico</span>', unsafe_allow_html=True)
    st.markdown("Cada ponto é uma disciplina. Posição calculada por SVD sobre TF-IDF. Cores = clusters temáticos.")

    plot_df = df_filt.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["cluster"] = [cluster_names.get(l, str(l)) for l in labels]
    plot_df["hover"] = plot_df["Name"] + "<br>" + plot_df["School"] + " | " + plot_df["credits_num"].fillna(0).astype(int).astype(str) + " cr."

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="cluster",
        hover_name="hover",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"x": "Dimensão 1", "y": "Dimensão 2", "cluster": "Cluster"},
        height=520,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", color="#0f1730"),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # distribuicao por escola
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<span class="section-tag">Por escola</span>', unsafe_allow_html=True)
        escola_counts = df_filt["School"].value_counts().reset_index()
        escola_counts.columns = ["Escola", "Disciplinas"]
        fig2 = px.bar(
            escola_counts, x="Disciplinas", y="Escola", orientation="h",
            color="Disciplinas", color_continuous_scale=["#eef1fb", "#223886"],
            height=380,
        )
        fig2.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter", color="#0f1730"),
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<span class="section-tag">Por cluster</span>', unsafe_allow_html=True)
        cluster_df = pd.DataFrame({"cluster": plot_df["cluster"]}).value_counts().reset_index()
        cluster_df.columns = ["Cluster", "Disciplinas"]
        fig3 = px.pie(
            cluster_df, names="Cluster", values="Disciplinas",
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=380,
        )
        fig3.update_traces(textposition="inside", textinfo="percent+label")
        fig3.update_layout(
            font=dict(family="Inter", color="#0f1730"),
            paper_bgcolor="#ffffff",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── aba 2: heatmap ────────────────────────────────────────────────────────────

with tab_heatmap:
    st.markdown('<span class="section-tag">Heatmap de similaridade de cosseno</span>', unsafe_allow_html=True)
    max_disp = 40
    if len(df_filt) > max_disp:
        st.info(f"Exibindo as {max_disp} disciplinas mais representativas (todas: {len(df_filt)}). Filtre por escola para ver mais.")
        # pega as com maior soma de similaridade (mais conectadas)
        soma_sim = sim.sum(axis=1)
        top_idx_rel = np.argsort(soma_sim)[-max_disp:]
        sim_disp = sim[np.ix_(top_idx_rel, top_idx_rel)]
        nomes_disp = df_filt.iloc[top_idx_rel]["Name"].str.slice(0, 35).tolist()
    else:
        sim_disp = sim
        nomes_disp = df_filt["Name"].str.slice(0, 35).tolist()

    fig_heat = go.Figure(go.Heatmap(
        z=sim_disp,
        x=nomes_disp,
        y=nomes_disp,
        colorscale=[[0, "#f0f3ff"], [0.5, "#7b93d3"], [1, "#223886"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Sim.", thickness=14),
    ))
    fig_heat.update_layout(
        height=600,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=10, color="#0f1730"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # pares com maior sobreposição
    st.markdown('<span class="section-tag">Pares com maior sobreposição</span>', unsafe_allow_html=True)
    names_all = df_filt["Name"].tolist()
    schools_all = df_filt["School"].tolist()
    pares = []
    n = len(df_filt)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > 0.05:
                pares.append({
                    "Disciplina A": names_all[i],
                    "Escola A": schools_all[i],
                    "Disciplina B": names_all[j],
                    "Escola B": schools_all[j],
                    "Similaridade": round(float(sim[i, j]), 3),
                })
    pares_df = pd.DataFrame(pares).sort_values("Similaridade", ascending=False).head(20)
    if pares_df.empty:
        st.info("Nenhum par com similaridade > 5% no filtro atual.")
    else:
        st.dataframe(
            pares_df,
            use_container_width=True,
            column_config={"Similaridade": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
        )

# ── aba 3: buscar similares ───────────────────────────────────────────────────

with tab_busca:
    st.markdown('<span class="section-tag">Encontre disciplinas sobrepostas</span>', unsafe_allow_html=True)

    nomes_opcao = df["Name"].tolist()
    disc_sel = st.selectbox("Selecione uma disciplina", nomes_opcao)

    if disc_sel:
        idx_sel_global = df[df["Name"] == disc_sel].index[0]
        sims_col = sim_full[idx_sel_global]
        top_idx = np.argsort(sims_col)[::-1][1:top_n + 1]

        disc_info = df.iloc[idx_sel_global]
        st.markdown(f"""
<div class="disc-card">
  <div class="disc-name">📖 {disc_info['Name']}</div>
  <div class="disc-meta">{disc_info['School name'] or disc_info['School']} &nbsp;|&nbsp; {int(disc_info['credits_num'] or 0)} créditos</div>
  <div style="margin-top:8px;font-size:.85rem;color:#374151;">{str(disc_info['Objectives'])[:300]}...</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"**Disciplinas mais similares** (top {top_n})")
        for rank, i in enumerate(top_idx, 1):
            sim_val = sims_col[i]
            row = df.iloc[i]
            pct = int(sim_val * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            st.markdown(f"""
<div class="disc-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="disc-name">#{rank} {row['Name']}</div>
      <div class="disc-meta">{row['School name'] or row['School']} &nbsp;|&nbsp; {int(row['credits_num'] or 0)} cr.</div>
    </div>
    <div style="text-align:right;font-size:.85rem;">
      <span style="color:#223886;font-weight:700;">{sim_val:.1%}</span>
      <div style="font-family:monospace;font-size:.65rem;color:#9ca3af;">{bar}</div>
    </div>
  </div>
  <div style="margin-top:8px;font-size:.8rem;color:#6b7280;">{str(row['Objectives'])[:200]}...</div>
</div>
""", unsafe_allow_html=True)

# ── aba 4: dados brutos ───────────────────────────────────────────────────────

with tab_dados:
    st.markdown('<span class="section-tag">Dataset completo</span>', unsafe_allow_html=True)
    busca = st.text_input("Filtrar por nome ou escola", placeholder="ex: machine learning, EACH...")
    df_show = df_filt.copy()
    if busca:
        mask = (
            df_show["Name"].str.contains(busca, case=False, na=False) |
            df_show["School"].str.contains(busca, case=False, na=False) |
            df_show["Objectives"].str.contains(busca, case=False, na=False)
        )
        df_show = df_show[mask]

    cols_show = ["School", "Code", "Name", "credits_num", "Professors", "Start date", "End date"]
    st.dataframe(
        df_show[cols_show].rename(columns={"credits_num": "Créditos", "School": "Escola", "Code": "Código", "Name": "Nome", "Professors": "Professores", "Start date": "Início", "End date": "Fim"}),
        use_container_width=True,
        height=450,
    )
    st.caption(f"{len(df_show)} disciplinas exibidas")
