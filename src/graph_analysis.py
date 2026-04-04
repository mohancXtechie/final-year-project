import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend - saves file instead of showing window
import matplotlib.pyplot as plt
import os

def run(base_dir):
    print("=" * 50)
    print("STEP 5: Graph Analysis")
    print("=" * 50)

    print("Loading log data...")
    logon  = pd.read_csv(os.path.join(base_dir, "dataset", "logon.csv"))
    device = pd.read_csv(os.path.join(base_dir, "dataset", "device.csv"))
    http   = pd.read_csv(
        os.path.join(base_dir, "dataset", "http.csv"),
        header=None,
        names=["id", "date", "user", "pc", "url"]
    )

    # Limit to 5000 events for performance
    logon  = logon.head(5000)
    device = device.head(5000)
    http   = http.head(5000)
    print("Using first 5000 events per log for visualization")

    # Load UEBA scores to highlight suspicious users
    try:
        ueba_path = os.path.join(base_dir, "dataset", "processed", "ueba_scores.csv")
        ueba = pd.read_csv(ueba_path)
        threshold_val = ueba["ueba_threshold"].iloc[0]
        suspicious_users = set(ueba[ueba["ueba_score_weighted"] > threshold_val]["user"])
        print(f"Suspicious users loaded: {len(suspicious_users)}")
    except Exception:
        suspicious_users = set()
        print("No UEBA scores found, skipping suspicious user highlighting")

    print("Building user activity graph...")
    G = nx.Graph()

    for _, row in logon.iterrows():
        G.add_edge(row["user"], row["pc"])

    for _, row in http.iterrows():
        G.add_edge(row["user"], row["url"])

    for _, row in device.iterrows():
        G.add_edge(row["user"], row["pc"])

    print(f"Graph nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

    print("Calculating degree centrality...")
    centrality = nx.degree_centrality(G)
    centrality_df = pd.DataFrame(
        list(centrality.items()),
        columns=["node", "centrality"]
    ).sort_values("centrality", ascending=False)

    out_path = os.path.join(base_dir, "dataset", "processed", "graph_scores.csv")
    centrality_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    print("\nTop 10 most central nodes:")
    print(centrality_df.head(10))

    # Visualization - saves to file instead of showing a window
    print("\nGenerating network visualization...")
    node_colors = []
    for node in G.nodes():
        if node in suspicious_users:
            node_colors.append("red")
        elif "DTAA" in str(node):
            node_colors.append("blue")
        elif "PC" in str(node):
            node_colors.append("gray")
        else:
            node_colors.append("green")

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=0.4, iterations=30, seed=42)
    nx.draw(
        G, pos,
        node_size=40,
        node_color=node_colors,
        edge_color="lightgray",
        with_labels=False
    )
    plt.title(
        "Insider Threat Behavior Network (5000 Events)\n"
        "Red=Suspicious  Blue=Normal Users  Gray=PCs  Green=Websites"
    )

    plot_path = os.path.join(base_dir, "dataset", "processed", "network_graph.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Network graph saved -> {plot_path}")