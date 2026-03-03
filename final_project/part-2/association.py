import pandas as pd
import emoji
from mlxtend.frequent_patterns import apriori, association_rules
import math
import networkx as nx
import matplotlib, mplcairo

matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ---------- 1) Load preprocessed CSV ----------
df = pd.read_csv("../data/reddit-amazon-emojis-clean.csv")


# ---------- 2) Extract unique emojis from each review ----------
def extract_emojis(s):
    if not isinstance(s, str):
        return []
    return list(dict.fromkeys(ch for ch in s if ch in emoji.EMOJI_DATA))


df["emojis"] = df["text"].apply(extract_emojis)

# ---------- 3) Keep reviews with ≥2 distinct emojis ----------
df = df[df["emojis"].apply(len) >= 2]

# ---------- 4) One-hot encode ----------
exploded = df.explode("emojis")[["emojis"]]
exploded["val"] = 1
exploded = exploded.reset_index(names="review_id").drop_duplicates()
basket = (
    exploded.pivot(index="review_id", columns="emojis", values="val")
    .fillna(0)
    .astype("uint8")
)

# drop very rare emojis to speed Apriori
min_item_support = 0.002
col_support = basket.mean(axis=0)
basket = basket.loc[:, col_support[col_support >= min_item_support].index]
basket = basket.drop(
    columns=["©", "®", "🏿", "🏾", "🏻", "🏼", "🏽", "💀", "♂", "♀"], errors="ignore"
)

# ---------- 5) Apriori frequent itemsets ----------
freq_itemsets = apriori(
    basket, min_support=0.001, use_colnames=True, max_len=4, low_memory=True
)
freq_itemsets = freq_itemsets.sort_values(["itemsets", "support"]).reset_index(
    drop=True
)

# ---------- 6) Association rules ----------
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.05)
rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(
    drop=True
)

# ---------- 7) Save results ----------
freq_itemsets.to_csv("frequent_emoji_itemsets.csv", index=False)
rules.to_csv("emoji_association_rules.csv", index=False)

# ---------- 8) Example ----------
A = {"😭"}
B = {"💔"}
q = rules[
    (rules["antecedents"] == frozenset(A)) & (rules["consequents"] == frozenset(B))
]
if not q.empty:
    print(q[["support", "confidence", "lift"]])
else:
    p_A = basket[list(A)].mean().item()
    p_B = basket[list(B)].mean().item()
    p_AuB = (basket[list(A | B)].sum(axis=1).eq(len(A | B))).mean().item()
    conf = p_AuB / p_A if p_A > 0 else float("nan")
    lift = conf / p_B if p_B > 0 else float("nan")
    print({"support_A": p_A, "support_AuB": p_AuB, "confidence": conf, "lift": lift})


def build_emoji_network(basket, freq_itemsets, top_n=50, min_pair_support=0.0005):
    """
    Undirected co-occurrence network.
    Nodes: emojis, size ∝ support.
    Edges: frequent 2-item sets, width ∝ support(A∪B), color by lift.
    """
    # Node supports from 1-item itemsets; fall back to column means if needed
    one_item = freq_itemsets[
        freq_itemsets["itemsets"].apply(lambda s: len(s) == 1)
    ].copy()
    if one_item.empty:
        supp = basket.mean(axis=0).to_dict()
    else:
        supp = {
            next(iter(s)): v for s, v in zip(one_item["itemsets"], one_item["support"])
        }

    # Select candidate nodes: top by support to reduce clutter
    top_nodes = sorted(supp.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    keep = {k for k, _ in top_nodes}

    # Edge list from 2-item itemsets
    two_item = freq_itemsets[
        freq_itemsets["itemsets"].apply(lambda s: len(s) == 2)
    ].copy()
    edges = []
    for s, sup in zip(two_item["itemsets"], two_item["support"]):
        a, b = tuple(s)
        if a in keep and b in keep and sup >= min_pair_support:
            p_a, p_b = supp.get(a, 0.0), supp.get(b, 0.0)
            lift = sup / (p_a * p_b) if p_a > 0 and p_b > 0 else float("nan")
            edges.append((a, b, {"support": sup, "lift": lift}))

    # Build graph
    G = nx.Graph()
    for n, p in top_nodes:
        G.add_node(n, support=p)
    G.add_edges_from(edges)
    return G


def draw_emoji_network(G, k=None, seed=42):
    """
    Draw with spring layout. Edge width ∝ support. Edge color by lift:
    <1 = gray, ~1 = lightgray, >1 = black (stronger dark as lift↑).
    """
    prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

    if k is None:
        k = 1 / math.sqrt(max(G.number_of_nodes(), 1))

    # three graph display options
    # pos = nx.spring_layout(G, k=k, seed=seed)
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)

    # Node sizes
    node_supports = nx.get_node_attributes(G, "support")
    sizes = [max(100, 20000 * node_supports[n]) for n in G.nodes()]  # scale

    # Edge widths and colors
    esupp = nx.get_edge_attributes(G, "support")
    elift = nx.get_edge_attributes(G, "lift")
    svals = list(esupp.values())
    min_s, max_s = min(svals), max(svals)
    widths = [
        0.5 + ((esupp[e] - min_s) / (max_s - min_s)) * (8.0 - 0.5) for e in G.edges()
    ]

    def lift_to_gray(l):
        if not math.isfinite(l):
            return (0.8, 0.8, 0.8)
        l_clamped = max(0.5, min(2.0, l))
        shade = 0.8 - (l_clamped - 1.0) * 0.6
        shade = max(0.1, min(1.0, shade))
        return (shade, shade, shade)

    colors = [lift_to_gray(elift[e]) for e in G.edges()]

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, linewidths=0.8, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors)
    ax = plt.gca()
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node,
            fontproperties=prop,
            fontsize=20,
            ha="center",
            va="center",
            clip_on=True,
        )

    legend_text = (
        "Thickness: Support(A)\n"
        "Darkness: Association beyond chance (Lift)\n"
        "Opacity: Confidence(A→B)"
    )

    plt.text(
        0.01,
        0.01,
        legend_text,
        transform=plt.gca().transAxes,
        fontsize=20,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )

    plt.axis("off")
    plt.title("Emoji Co-occurrence Network")
    plt.tight_layout()
    # plt.show()
    plt.savefig("movie_emoji_associations", dpi=300)


# ---------- Build and draw ----------
G = build_emoji_network(basket, freq_itemsets, top_n=30, min_pair_support=0.0008)
draw_emoji_network(G, 1.5)
