import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import random
import math

st.set_page_config(page_title="CampaignSpread", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
section[data-testid="stSidebar"] { background: #0f0f1a !important; }
[data-testid="metric-container"] { background: #12121f; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #7c6af7; font-size: 1.8rem; font-weight: 700; }
.stButton > button { background: linear-gradient(135deg, #7c6af7, #a855f7); color: white; border: none; border-radius: 8px; font-weight: 600; }
.card { background: #12121f; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Core Functions ────────────────────────────────────────────────

def build_network(n_nodes, topology, avg_degree=4):
    random.seed(42); np.random.seed(42)
    m = max(1, avg_degree // 2)
    if topology == "Scale-Free (Barabasi-Albert)":
        G = nx.barabasi_albert_graph(n_nodes, m=m, seed=42)
    elif topology == "Small-World (Watts-Strogatz)":
        G = nx.watts_strogatz_graph(n_nodes, k=avg_degree, p=0.1, seed=42)
    elif topology == "Random (Erdos-Renyi)":
        G = nx.erdos_renyi_graph(n_nodes, p=avg_degree/(n_nodes-1), seed=42)
    else:
        G = nx.barabasi_albert_graph(n_nodes, m=2, seed=42)

    segments  = ['Enterprise', 'SMB', 'Consumer', 'Influencer']
    platforms = ['LinkedIn', 'Twitter/X', 'Instagram', 'Internal']
    for node in G.nodes():
        G.nodes[node]['segment']      = random.choice(segments)
        G.nodes[node]['platform']     = random.choice(platforms)
        G.nodes[node]['adoption_prob']= round(random.uniform(0.05, 0.95), 2)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 2)
    return G


def select_seeds(G, k, strategy):
    if strategy == "Degree Centrality":
        scores = nx.degree_centrality(G)
    elif strategy == "Betweenness Centrality":
        scores = nx.betweenness_centrality(G)
    elif strategy == "PageRank":
        scores = nx.pagerank(G, alpha=0.85)
    elif strategy == "Eigenvector Centrality":
        try:    scores = nx.eigenvector_centrality(G, max_iter=500)
        except: scores = nx.degree_centrality(G)
    else:  # Random
        nodes = list(G.nodes()); random.shuffle(nodes)
        return nodes[:k]
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def simulate_ic(G, seeds, prob, max_rounds=20):
    adopted = set(seeds); wave_data = {n: 0 for n in seeds}
    timeline = [len(seeds)]; current = list(seeds)
    for rnd in range(1, max_rounds + 1):
        next_wave = []
        for node in current:
            for nb in G.neighbors(node):
                if nb not in adopted:
                    w = G[node][nb].get('weight', 1.0)
                    p = prob * w * G.nodes[nb]['adoption_prob']
                    if random.random() < p:
                        adopted.add(nb); next_wave.append(nb); wave_data[nb] = rnd
        timeline.append(len(next_wave)); current = next_wave
        if not next_wave: break
    return adopted, timeline, wave_data


def simulate_lt(G, seeds, threshold=0.5, max_rounds=20):
    thresholds = {n: random.uniform(0.2, threshold) for n in G.nodes()}
    adopted = set(seeds); wave_data = {n: 0 for n in seeds}
    timeline = [len(seeds)]
    for rnd in range(1, max_rounds + 1):
        new_wave = []
        for node in G.nodes():
            if node in adopted: continue
            nbrs = list(G.neighbors(node))
            if not nbrs: continue
            inf_sum   = sum(G[node][nb].get('weight', 1.0) for nb in nbrs if nb in adopted)
            total_w   = sum(G[node][nb].get('weight', 1.0) for nb in nbrs)
            if total_w > 0 and (inf_sum / total_w) >= thresholds[node]:
                new_wave.append(node)
        for node in new_wave:
            adopted.add(node); wave_data[node] = rnd
        timeline.append(len(new_wave))
        if not new_wave: break
    return adopted, timeline, wave_data


def network_effect(rate, model):
    if model == "Metcalfe (n²)":   return rate ** 2
    elif model == "Reed (2^n)":    return min(2 ** (rate * 10), 1024) / 1024
    else:                          return rate


def draw_network(G, seeds, adopted, wave_data, max_show=250):
    seeds_set = set(seeds)
    nodes_show = list(seeds_set) + random.sample(
        [n for n in G.nodes() if n not in seeds_set],
        min(max_show - len(seeds_set), max(0, len(G) - len(seeds_set)))
    )
    H   = G.subgraph(nodes_show)
    pos = nx.spring_layout(H, seed=42, k=1.5/math.sqrt(max(len(H),1)))

    ex, ey = [], []
    for u, v in H.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]

    nx_list, ny_list, colors, sizes, texts = [], [], [], [], []
    for node in H.nodes():
        x, y = pos[node]
        nx_list.append(x); ny_list.append(y)
        if node in seeds_set:
            colors.append('#7c6af7'); sizes.append(18)
        elif node in adopted:
            colors.append('#22c55e'); sizes.append(10)
        else:
            colors.append('#2a2a4a'); sizes.append(6)
        texts.append(
            f"Node {node} | {G.nodes[node]['segment']} | {G.nodes[node]['platform']}"
            f"<br>Degree: {G.degree(node)} | "
            f"{'Seed' if node in seeds_set else 'Adopted' if node in adopted else 'Unaware'}"
        )

    fig = go.Figure([
        go.Scatter(x=ex, y=ey, mode='lines',
                   line=dict(width=0.4, color='rgba(100,100,180,0.2)'), hoverinfo='none'),
        go.Scatter(x=nx_list, y=ny_list, mode='markers',
                   marker=dict(size=sizes, color=colors,
                               line=dict(width=0.5, color='rgba(255,255,255,0.1)')),
                   text=texts, hoverinfo='text'),
    ])
    fig.update_layout(
        showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 Campaign Settings")
    n_nodes    = st.slider("Audience Nodes",          50, 1000, 300, 50)
    topology   = st.selectbox("Network Topology", [
        "Scale-Free (Barabasi-Albert)",
        "Small-World (Watts-Strogatz)",
        "Random (Erdos-Renyi)",
        "Corporate Hierarchical",
    ])
    avg_degree = st.slider("Avg Connections / Node",  2, 12, 4)
    st.markdown("---")
    n_seeds    = st.slider("Seed Influencers (k)",    1, 30, 5)
    strategy   = st.selectbox("Influencer Selection", [
        "Degree Centrality",
        "Betweenness Centrality",
        "PageRank",
        "Eigenvector Centrality",
        "Random Baseline",
    ])
    st.markdown("---")
    model      = st.selectbox("Diffusion Model", [
        "Independent Cascade (IC)",
        "Linear Threshold (LT)",
    ])
    prop_prob  = st.slider("Propagation Probability", 0.01, 0.50, 0.12, 0.01)
    max_rounds = st.slider("Max Rounds",              5, 50, 20)
    st.markdown("---")
    ne_model   = st.selectbox("Network Effect Law", [
        "Metcalfe (n²)", "Reed (2^n)", "Linear"
    ])
    run_btn    = st.button("🚀 Run Simulation", use_container_width=True)


# ─── Header ────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#0f0f1a,#1a1040,#0f0f1a);
            border:1px solid #1e1e3a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">
  <h1 style="margin:0;background:linear-gradient(90deg,#e8e8f0,#7c6af7);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    📡 CampaignSpread
  </h1>
  <p style="color:#6060a0;margin:0.4rem 0 0">
    Corporate Product Awareness · Influence Maximization · Network Effects
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Info Cards ────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
for col, icon, title, body in [
    (c1, "🌱", "Seed Users",            "High-influence early adopters who trigger the awareness cascade."),
    (c2, "📣", "Influence Maximization","Pick optimal k seeds to maximise downstream product adoption."),
    (c3, "🔗", "Network Effects",       "Each new adopter raises product value for all (Metcalfe/Reed)."),
    (c4, "🎯", "Audience Segments",     "Enterprise · SMB · Consumer · Influencer across 4 platforms."),
]:
    col.markdown(f"""
    <div class="card">
      <b style="color:#7c6af7">{icon} {title}</b>
      <p style="color:#9090a8;font-size:0.85rem;margin:0.4rem 0 0">{body}</p>
    </div>""", unsafe_allow_html=True)


# ─── Run ───────────────────────────────────────────────────────────

if run_btn or 'res' not in st.session_state:
    with st.spinner("Simulating…"):
        G     = build_network(n_nodes, topology, avg_degree)
        seeds = select_seeds(G, n_seeds, strategy)

        if model == "Independent Cascade (IC)":
            adopted, timeline, wave_data = simulate_ic(G, seeds, prop_prob, max_rounds)
        else:
            adopted, timeline, wave_data = simulate_lt(G, seeds, prop_prob, max_rounds)

        adoption_rate = len(adopted) / n_nodes
        ne_val        = network_effect(adoption_rate, ne_model)

        # Influencer table
        dc = nx.degree_centrality(G)
        seed_df = pd.DataFrame([{
            "Node": n, "Degree": G.degree(n),
            "Segment": G.nodes[n]['segment'], "Platform": G.nodes[n]['platform'],
            "Adopt Prob": G.nodes[n]['adoption_prob'],
            "Centrality": round(dc[n], 4),
        } for n in seeds]).sort_values("Degree", ascending=False)

        # Timeline
        tdf = pd.DataFrame({
            "Round": range(len(timeline)),
            "New Adoptions": timeline,
            "Cumulative": np.cumsum(timeline),
        })

        # Segment / Platform
        seg_cnt = defaultdict(int); plt_cnt = defaultdict(int)
        for node in adopted:
            seg_cnt[G.nodes[node]['segment']] += 1
            plt_cnt[G.nodes[node]['platform']] += 1

        st.session_state['res'] = dict(
            G=G, seeds=seeds, adopted=adopted, wave_data=wave_data,
            adoption_rate=adoption_rate, ne_val=ne_val,
            seed_df=seed_df, tdf=tdf,
            seg_df=pd.DataFrame(list(seg_cnt.items()), columns=["Segment","Adopted"]),
            plt_df=pd.DataFrame(list(plt_cnt.items()), columns=["Platform","Adopted"]),
        )

R = st.session_state['res']

# ─── KPIs ──────────────────────────────────────────────────────────

st.markdown("---")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Nodes",    f"{n_nodes:,}")
k2.metric("Seed Influencers", f"{len(R['seeds'])}")
k3.metric("Nodes Reached",  f"{len(R['adopted']):,}")
k4.metric("Adoption Rate",  f"{R['adoption_rate']*100:.1f}%")
k5.metric("Network Effect", f"{R['ne_val']:.3f}")

# ─── Tabs ──────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    "🕸 Network", "📈 Timeline", "🏆 Influencers", "🔬 Segments", "⚔ Compare Strategies"
])

LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#9090a8'), margin=dict(l=0,r=0,t=40,b=0),
    xaxis=dict(gridcolor='#1e1e3a'), yaxis=dict(gridcolor='#1e1e3a'),
)

# Tab 1 – Network Graph
with t1:
    l, r = st.columns([3,1])
    with r:
        st.markdown("""
        <div class="card">
        <b style="color:#7c6af7">Legend</b><br><br>
        🟣 Seed Influencers<br>
        🟢 Adopted Nodes<br>
        ⬛ Unaware Nodes
        </div>""", unsafe_allow_html=True)
        st.progress(min(R['adoption_rate'], 1.0))
        st.caption(f"{R['adoption_rate']*100:.1f}% reached | {len(R['tdf'])-1} rounds")
    with l:
        st.plotly_chart(
            draw_network(R['G'], R['seeds'], R['adopted'], R['wave_data']),
            use_container_width=True, config={"displayModeBar": False}
        )

# Tab 2 – Timeline
with t2:
    c_a, c_b = st.columns(2)
    with c_a:
        fig = go.Figure(go.Bar(
            x=R['tdf']['Round'], y=R['tdf']['New Adoptions'],
            marker_color='rgba(124,106,247,0.8)'
        ))
        fig.update_layout(title="New Adoptions per Round", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        fig = go.Figure(go.Scatter(
            x=R['tdf']['Round'], y=R['tdf']['Cumulative'],
            mode='lines+markers', line=dict(color='#a855f7', width=3),
            fill='tozeroy', fillcolor='rgba(124,106,247,0.1)'
        ))
        fig.add_hline(y=n_nodes, line_dash='dot', line_color='rgba(255,255,255,0.2)',
                      annotation_text='Total Network')
        fig.update_layout(title="Cumulative Adoption", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    rates  = np.linspace(0, 1, 100)
    ne_vals = [network_effect(r, ne_model) for r in rates]
    fig = go.Figure(go.Scatter(
        x=rates*100, y=ne_vals, mode='lines',
        line=dict(color='#22c55e', width=2.5),
        fill='tozeroy', fillcolor='rgba(34,197,94,0.08)'
    ))
    fig.add_vline(x=R['adoption_rate']*100, line_dash='dash', line_color='#7c6af7',
                  annotation_text=f"Current: {R['adoption_rate']*100:.1f}%",
                  annotation_font_color='#7c6af7')
    fig.update_layout(title=f"Network Effect Curve — {ne_model}", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3 – Influencer Table
with t3:
    st.dataframe(
        R['seed_df'].style
            .background_gradient(subset=['Degree','Centrality'], cmap='Purples')
            .format({'Adopt Prob': '{:.0%}', 'Centrality': '{:.4f}'}),
        use_container_width=True
    )
    fig = go.Figure(go.Bar(
        x=R['seed_df']['Centrality'],
        y=[f"Node {n}" for n in R['seed_df']['Node']],
        orientation='h',
        marker=dict(color=R['seed_df']['Centrality'], colorscale='Purples', showscale=True)
    ))
    fig.update_layout(title="Influencer Centrality Scores", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4 – Segment Analysis
with t4:
    s1, s2 = st.columns(2)
    with s1:
        fig = px.pie(R['seg_df'], names='Segment', values='Adopted',
                     color_discrete_sequence=['#7c6af7','#a855f7','#22c55e','#f59e0b'], hole=0.55)
        fig.update_layout(title="Adoptions by Segment",
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#9090a8'),
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with s2:
        fig = px.bar(R['plt_df'], x='Platform', y='Adopted', color='Platform',
                     color_discrete_sequence=['#7c6af7','#a855f7','#22c55e','#f59e0b'])
        fig.update_layout(title="Adoptions by Platform", showlegend=False, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# Tab 5 – Strategy Comparison
with t5:
    st.info("Runs all 5 strategies on the same network for a fair comparison.")
    if st.button("▶ Compare All Strategies"):
        strategies = [
            "Degree Centrality", "Betweenness Centrality",
            "PageRank", "Eigenvector Centrality", "Random Baseline"
        ]
        rows = []
        prog = st.progress(0)
        for i, strat in enumerate(strategies):
            s = select_seeds(R['G'], n_seeds, strat)
            if model == "Independent Cascade (IC)":
                ad, tl, _ = simulate_ic(R['G'], s, prop_prob, max_rounds)
            else:
                ad, tl, _ = simulate_lt(R['G'], s, prop_prob, max_rounds)
            rows.append({
                "Strategy": strat,
                "Nodes Adopted": len(ad),
                "Adoption %": round(len(ad)/n_nodes*100, 2),
                "Rounds to Peak": len(tl)-1,
                "Network Effect": round(network_effect(len(ad)/n_nodes, ne_model), 4),
            })
            prog.progress((i+1)/len(strategies))

        cdf = pd.DataFrame(rows).sort_values("Nodes Adopted", ascending=False)
        st.dataframe(
            cdf.style.background_gradient(subset=["Adoption %","Network Effect"], cmap="Purples"),
            use_container_width=True
        )
        fig = go.Figure(go.Bar(
            x=cdf['Strategy'], y=cdf['Nodes Adopted'],
            marker=dict(color=cdf['Nodes Adopted'], colorscale='Purples', showscale=True)
        ))
        fig.update_layout(title="Strategy Comparison — Nodes Adopted", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
