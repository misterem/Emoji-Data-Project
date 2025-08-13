# pip install pandas mlxtend emoji networkx matplotlib
import os, math
import pandas as pd
import numpy as np
import emoji
import matplotlib, mplcairo
matplotlib.use("module://mplcairo.macosx")
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from matplotlib.font_manager import FontProperties

# ---------- CONFIG ----------
CSV_PATH = "../movie_data/reddit-amazon-emoji-only.csv"      # your preprocessed file (must have a 'text' column)
MIN_EMOJI_PER_REVIEW = 2                   # keep only reviews with ≥ this many distinct emojis
MIN_ITEM_SUPPORT = 0.002                   # drop rare single emojis before Apriori (speed)
MIN_PAIR_SUPPORT = 0.001                   # Apriori min_support for itemsets (pairs+)
MAX_ITEMSET_LEN   = 3                      # search up to 3-emoji itemsets
TOP_N_NODES       = 30                     # show only top-N emojis by support in the graph
WIDTH_MIN, WIDTH_MAX = 0.5, 9.0            # visual edge width range
ALPHA_MIN, ALPHA_MAX = 0.25, 1.0           # opacity range for edges (confidence→alpha)
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH)

# ---------- EMOJI EXTRACTION (using emoji package) ----------
def extract_emojis(s):
    if not isinstance(s, str): return []
    # iterate over codepoints; keep unique per review (order-preserving)
    return list(dict.fromkeys(ch for ch in s if ch in emoji.EMOJI_DATA))

df["emojis"] = df["text"].apply(extract_emojis)
df = df[df["emojis"].apply(len) >= MIN_EMOJI_PER_REVIEW].reset_index(drop=True)

# ---------- ONE-HOT ----------
exploded = df.explode("emojis")[["emojis"]]
exploded["val"] = 1
exploded = exploded.reset_index(names="review_id").drop_duplicates()
basket = exploded.pivot(index="review_id", columns="emojis", values="val").fillna(0).astype("uint8")

# drop ultra-rare single items
col_support = basket.mean(axis=0)
basket = basket.loc[:, col_support[col_support >= MIN_ITEM_SUPPORT].index]
basket = basket.drop(columns=['©', '®', "🏿", "🏾", "🏻", "🏼", "🏽", '💀', '♂', '♀'], errors="ignore")

# ---------- APRIORI + RULES ----------
freq_itemsets = apriori(
    basket, min_support=MIN_PAIR_SUPPORT, use_colnames=True,
    max_len=MAX_ITEMSET_LEN, low_memory=True
).sort_values(["itemsets"]).reset_index(drop=True)

rules = association_rules(
    freq_itemsets, metric="confidence", min_threshold=0.0
).sort_values(["lift","confidence"], ascending=False).reset_index(drop=True)

# supports dict for singles
supp_single = {}
for s, sup in zip(freq_itemsets["itemsets"], freq_itemsets["support"]):
    if len(s) == 1:
        supp_single[next(iter(s))] = sup
# fallback for singles missing from freq table
for c in basket.columns:
    supp_single.setdefault(c, basket[c].mean())

# supports for pairs (unordered frozenset of size 2)
supp_pair = {frozenset(s): sup for s, sup in zip(freq_itemsets["itemsets"], freq_itemsets["support"]) if len(s) == 2}

# confidence map for ordered pairs from rules; if missing, we’ll compute
conf_map = {}
for a, b, conf in zip(rules["antecedents"], rules["consequents"], rules["confidence"]):
    if len(a)==1 and len(b)==1:
        a1, b1 = next(iter(a)), next(iter(b))
        conf_map[(a1, b1)] = max(conf_map.get((a1,b1), 0.0), float(conf))

# ---------- BUILD GRAPH ----------
# choose top-N nodes by single support
top_nodes = sorted(supp_single.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_NODES]
keep = {k for k,_ in top_nodes}

G = nx.Graph()
for n, p in top_nodes:
    G.add_node(n, support=p)

edges = []
for pair, sup_ab in supp_pair.items():
    if len(pair) != 2: continue
    a, b = tuple(pair)
    if a not in keep or b not in keep: continue
    p_a, p_b = supp_single.get(a,0), supp_single.get(b,0)
    if p_a==0 or p_b==0: continue
    # lift
    lift = sup_ab / (p_a * p_b)
    # confidence (use the stronger direction)
    c_ab = conf_map.get((a,b), sup_ab / p_a)
    c_ba = conf_map.get((b,a), sup_ab / p_b)
    conf = max(c_ab, c_ba)
    edges.append((a, b, {"support": sup_ab, "lift": lift, "confidence": conf}))

G.add_edges_from(edges)

# ---------- LAYOUT ----------
# cleaner layout than raw spring for many graphs
# pos = nx.kamada_kawai_layout(G)
pos = nx.circular_layout(G)

# ---------- VISUAL ENCODING HELPERS ----------
# def lift_to_gray(l):
#     # clamp lift to [0.5, 2.5], map >1 to darker
#     l = max(0.5, min(2.5, float(l)))
#     shade = 0.85 - (l - 1.0) * 0.5    # lift>1 → darker
#     shade = max(0.1, min(1.0, shade)) # safety
#     return (shade, shade, shade)

def lift_to_gray(l):
    if not math.isfinite(l):
        return (0.8, 0.8, 0.8)
    l_clamped = max(0.5, min(2.0, l))
    shade = 0.8 - (l_clamped - 1.0) * 0.6
    shade = max(0.1, min(1.0, shade))  # clamp into valid range
    return (shade, shade, shade)

def normalize(values, out_min, out_max):
    v = np.asarray(values, float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or vmin==vmax:
        return [0.5*(out_min+out_max)] * len(v)
    return list(out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin))

# node sizes by support
node_sup = nx.get_node_attributes(G, "support")
node_sizes = normalize([node_sup[n] for n in G.nodes()], 200, 9000)

# edge widths by pair support
e_sup  = [G.edges[e]["support"] for e in G.edges()]
edge_w = normalize(e_sup, WIDTH_MIN, WIDTH_MAX)

# edge colors by lift (darkness)
e_lift = [G.edges[e]["lift"] for e in G.edges()]
edge_c = [lift_to_gray(l) for l in e_lift]

# edge alpha by confidence
e_conf = [G.edges[e]["confidence"] for e in G.edges()]
edge_a = normalize(e_conf, ALPHA_MIN, ALPHA_MAX)

# ---------- DRAW ----------
plt.figure(figsize=(13, 10))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, linewidths=0.8, edgecolors="black")
# draw edges in batches to apply per-edge alpha
ax = plt.gca()
for (u,v), w, c, a in zip(G.edges(), edge_w, edge_c, edge_a):
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], width=w, edge_color=[c], alpha=a)

# labels with emoji font
for node, (x, y) in pos.items():
    ax.text(x, y, node, fontproperties=prop, ha='center', va='center', clip_on=True)

# text legend (requested phrasing)
legend_text = "Thickness: Support\nDarkness: Association (Lift)\nOpacity: Confidence"
plt.text(
        0.01, 0.01, legend_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va='center',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
    )

plt.title("Emoji Co-occurrence Network (Apriori-derived)", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.savefig('movie_emoji_associations_apriori', dpi=300)
# plt.show()
