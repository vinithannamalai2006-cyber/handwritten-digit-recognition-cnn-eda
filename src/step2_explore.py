"""Step 2 - filter and mine: data-quality audit and exploratory analysis of the digit images.

Run:  python src/step2_explore.py      (writes figures/d*.png and results/eda.json)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

import common as C

C.style()
tra, tes = C.load_tables()
Xtr, Xte = C.to_images(tra), C.to_images(tes)
ytr, yte = tra["digit"].to_numpy(), tes["digit"].to_numpy()
out = {"n_train": len(tra), "n_test": len(tes), "n_fields": tra.shape[1]}

# ------------------------------------------------------------------ audit
out["missing_cells"] = int(tra.isna().sum().sum() + tes.isna().sum().sum())
out["out_of_range"] = int(((tra[C.PIXELS] < 0) | (tra[C.PIXELS] > 16)).sum().sum()
                          + ((tes[C.PIXELS] < 0) | (tes[C.PIXELS] > 16)).sum().sum())
out["dup_rows_train"] = int(tra.duplicated().sum())
out["dup_rows_test"] = int(tes.duplicated().sum())
cross = pd.merge(tra[C.PIXELS].drop_duplicates(), tes[C.PIXELS].drop_duplicates(), how="inner")
out["identical_images_train_and_test"] = int(len(cross))
dup_feats = tra[C.PIXELS].duplicated()
conflicts = tra[tra[C.PIXELS].duplicated(keep=False)].groupby(C.PIXELS)["digit"].nunique()
out["dup_images_with_conflicting_labels"] = int((conflicts > 1).sum())
var = tra[C.PIXELS].var()
out["zero_variance_pixels"] = [p for p in C.PIXELS if var[p] == 0]
out["near_constant_pixels"] = [p for p in C.PIXELS if (tra[p] == 0).mean() > 0.99]
both = pd.concat([tra, tes])
out["pixels_zero_pct"] = round(100 * float((both[C.PIXELS] == 0).to_numpy().mean()), 2)
out["pixels_sixteen_pct"] = round(100 * float((both[C.PIXELS] == 16).to_numpy().mean()), 2)

# parse check: the 8 x 8 test table equals 4 x 4 block counts of the original 32 x 32 bitmaps
B, yb = C.load_bitmaps("optdigits-orig.windep.Z")
blocks = B.reshape(-1, 8, 4, 8, 4).sum(axis=(2, 4))
out["bitmap_to_table_exact_match_pct"] = round(100 * float(np.mean(np.all(blocks == Xte, axis=(1, 2)) & (yb == yte))), 2)

fig = plt.figure(figsize=(10, 4.6))
picks = [int(np.where(yte == d)[0][3]) for d in (0, 2, 4, 7, 9)]
for k, i in enumerate(picks):
    a = fig.add_subplot(2, 5, k + 1)
    a.imshow(B[i], cmap="gray_r", interpolation="nearest")
    for g in range(1, 8):
        a.axhline(g * 4 - 0.5, color=C.PINK, lw=0.6); a.axvline(g * 4 - 0.5, color=C.PINK, lw=0.6)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_title(f"32 x 32 bitmap, label {yte[i]}", fontsize=9)
    b = fig.add_subplot(2, 5, k + 6)
    b.imshow(Xte[i], cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
    for r in range(8):
        for c in range(8):
            v = int(Xte[i, r, c])
            b.text(c, r, v, ha="center", va="center", fontsize=6.5, color="white" if v > 9 else C.INK2)
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    b.set_title("8 x 8 block counts", fontsize=9)
fig.tight_layout()
C.save(fig, "d01_bitmap_to_features")

# ------------------------------------------------------------------ class balance
ct, ce = np.bincount(ytr, minlength=10), np.bincount(yte, minlength=10)
chi_tr = stats.chisquare(ct)
chi_te = stats.chisquare(ce)
homog = stats.chi2_contingency(np.vstack([ct, ce]))
out["class_counts_train"], out["class_counts_test"] = ct.tolist(), ce.tolist()
out["chisq_uniform_train"] = [round(chi_tr.statistic, 2), round(chi_tr.pvalue, 4)]
out["chisq_uniform_test"] = [round(chi_te.statistic, 2), round(chi_te.pvalue, 4)]
out["chisq_train_vs_test"] = [round(homog.statistic, 2), round(homog.pvalue, 4)]
fig, ax = plt.subplots(figsize=(9, 3.3))
x = np.arange(10)
ax.bar(x - 0.2, 100 * ct / ct.sum(), 0.36, color=C.GREEN, label=f"Training writers (n = {ct.sum():,})")
ax.bar(x + 0.2, 100 * ce / ce.sum(), 0.36, color=C.PINK, label=f"Test writers (n = {ce.sum():,})")
ax.axhline(10, color=C.INK2, lw=1, ls=":")
ax.text(9.6, 10.15, "uniform 10%", ha="right", fontsize=8, color=C.INK2)
ax.set_xticks(x); ax.set_xlabel("Digit class"); ax.set_ylabel("Share of images (%)")
ax.set_ylim(8.5, 11)
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
C.save(fig, "d02_class_balance")

# ------------------------------------------------------------------ mean / spread images
fig, axes = plt.subplots(2, 11, figsize=(11, 2.6), gridspec_kw={"width_ratios": [1] * 10 + [0.08]})
for d in range(10):
    m, s = Xtr[ytr == d].mean(0), Xtr[ytr == d].std(0)
    im1 = axes[0, d].imshow(m, cmap="gray_r", vmin=0, vmax=16)
    im2 = axes[1, d].imshow(s, cmap=C.GREENS, vmin=0, vmax=8)
    axes[0, d].set_title(str(d), loc="center", fontsize=10)
    for a in axes[:, d]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
axes[0, 0].set_ylabel("mean", fontsize=9); axes[1, 0].set_ylabel("std. dev.", fontsize=9)
fig.colorbar(im1, cax=axes[0, 10]); fig.colorbar(im2, cax=axes[1, 10])
fig.tight_layout(w_pad=0.3)
C.save(fig, "d03_class_mean_std")

# ------------------------------------------------------------------ discrete distribution of pixel counts
vals = both[C.PIXELS].to_numpy().ravel().astype(int)
obs = np.bincount(vals, minlength=17)
n = 16
p1 = vals.mean() / n
ll_bin = stats.binom.logpmf(vals, n, p1).sum()


def binom_mix_em(v, k=3, iters=300):
    """EM for a k-component binomial mixture on counts 0..16 (ink regimes: background, edge, stroke core)."""
    cnt = np.bincount(v, minlength=n + 1)
    support = np.arange(n + 1)
    w = np.full(k, 1 / k)
    p = np.linspace(0.02, 0.95, k)
    for _ in range(iters):
        like = np.array([w[j] * stats.binom.pmf(support, n, p[j]) for j in range(k)])     # k x 17
        resp = like / like.sum(0, keepdims=True)
        wk = (resp * cnt).sum(1)
        w = wk / wk.sum()
        p = (resp * cnt * support).sum(1) / (n * wk)
    ll = float((cnt * np.log(like.sum(0) + 1e-300)).sum())
    return w, p, ll


w3, pk3, ll3 = binom_mix_em(vals, 3)
order = np.argsort(pk3)
w3, pk3 = w3[order], pk3[order]
mix_pmf = sum(w3[j] * stats.binom.pmf(np.arange(17), n, pk3[j]) for j in range(3))
out["pixel_value_counts"] = obs.tolist()
out["binomial_p"] = round(float(p1), 4)
out["aic_binomial"] = round(2 * 1 - 2 * float(ll_bin), 1)
out["aic_mixture3"] = round(2 * 5 - 2 * ll3, 1)
out["mixture3_weights"] = [round(float(v), 4) for v in w3]
out["mixture3_p"] = [round(float(v), 4) for v in pk3]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
N = len(vals)
ax[0].bar(np.arange(17), obs / N, color="#c9c9c9", width=0.75, label="Observed")
ax[0].plot(np.arange(17), stats.binom.pmf(np.arange(17), n, p1), "o-", color=C.PINK, ms=4, lw=1.6,
           label=f"Binomial(16, {p1:.2f})")
ax[0].plot(np.arange(17), mix_pmf, "s-", color=C.GREEN, ms=4, lw=1.8, label="3-component binomial mixture")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 0.8); C.plain_log(ax[0])
ax[0].set_xlabel("Pixel value (on-pixels in a 4 x 4 block)"); ax[0].set_ylabel("Probability (log scale)")
ax[0].set_title("a  Distribution of all 359,680 pixel values")
ax[0].legend(loc="upper center")
ink_p = (both[C.PIXELS] > 0).mean().to_numpy().reshape(8, 8)
im = ax[1].imshow(ink_p, cmap=C.GREENS, vmin=0, vmax=1)
for r in range(8):
    for c in range(8):
        ax[1].text(c, r, f"{ink_p[r, c]:.2f}", ha="center", va="center", fontsize=6.5,
                   color="white" if ink_p[r, c] > 0.6 else C.INK)
ax[1].set_xticks(range(8)); ax[1].set_yticks(range(8)); ax[1].grid(False)
ax[1].set_title("b  P(pixel > 0) at each position")
ax[1].set_xlabel("column"); ax[1].set_ylabel("row")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
fig.tight_layout()
C.save(fig, "d04_pixel_distribution")

# ------------------------------------------------------------------ continuous: total ink per image
ink = both[C.PIXELS].sum(axis=1).to_numpy()
digit_all = both["digit"].to_numpy()
mu, sd = ink.mean(), ink.std(ddof=1)
sw = stats.shapiro(np.random.default_rng(C.SEED).choice(ink, 1000, replace=False))
skn = stats.skewnorm.fit(ink)
ll_norm = stats.norm.logpdf(ink, mu, sd).sum()
ll_skn = stats.skewnorm.logpdf(ink, *skn).sum()
kw = stats.kruskal(*[ink[digit_all == d] for d in range(10)])
eps2 = kw.statistic / ((len(ink) ** 2 - 1) / (len(ink) + 1))
med = {d: float(np.median(ink[digit_all == d])) for d in range(10)}
out["ink"] = {"mean": round(float(mu), 1), "sd": round(float(sd), 1), "min": int(ink.min()), "max": int(ink.max()),
              "skew": round(float(stats.skew(ink)), 3), "kurtosis": round(float(stats.kurtosis(ink)), 3),
              "shapiro_W": round(float(sw.statistic), 4), "shapiro_p": float(sw.pvalue),
              "aic_normal": round(4 - 2 * float(ll_norm), 1), "aic_skewnormal": round(6 - 2 * float(ll_skn), 1),
              "skewnorm_a": round(float(skn[0]), 3), "kruskal_H": round(float(kw.statistic), 1),
              "kruskal_p": float(kw.pvalue), "epsilon2": round(float(eps2), 3),
              "median_by_digit": {str(k): v for k, v in med.items()}}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.1, 0.8, 1.4]})
ax[0].hist(ink, bins=45, density=True, color="#c9c9c9", edgecolor="white", lw=0.3, label="Observed")
g = np.linspace(ink.min(), ink.max(), 300)
ax[0].plot(g, stats.norm.pdf(g, mu, sd), color=C.PINK, lw=1.8, label="Normal fit")
ax[0].plot(g, stats.skewnorm.pdf(g, *skn), color=C.GREEN, lw=1.8, label="Skew-normal fit")
ax[0].set_xlabel("Total ink (sum of 64 pixel values)"); ax[0].set_ylabel("Density")
ax[0].set_title("a  Ink per image"); ax[0].legend(loc="upper left", fontsize=8)
(osm, osr), (sl, ic, rq) = stats.probplot(ink, dist="norm")
ax[1].plot(osm[::4], osr[::4], "o", ms=2.5, color=C.GREEN, alpha=0.6, mec="none")
ax[1].plot(osm, sl * osm + ic, color=C.INK, lw=1)
ax[1].set_xlabel("Normal quantiles"); ax[1].set_ylabel("Ordered ink")
ax[1].set_title(f"b  Q-Q plot (r = {rq:.3f})")
order_d = sorted(range(10), key=lambda d: med[d])
bp = ax[2].boxplot([ink[digit_all == d] for d in order_d], widths=0.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color=C.INK, lw=1.4))
for box in bp["boxes"]:
    box.set(facecolor="#dff0df", edgecolor=C.GREEN)
ax[2].set_xticks(range(1, 11), [str(d) for d in order_d])
ax[2].set_xlabel("Digit (sorted by median ink)"); ax[2].set_ylabel("Total ink")
ax[2].set_title(f"c  Ink by digit (Kruskal-Wallis H = {kw.statistic:.0f})")
fig.tight_layout()
C.save(fig, "d05_ink_distribution")
out["ink_qq_r"] = round(float(rq), 4)

# ------------------------------------------------------------------ pixel-wise variability and correlation
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1, 1.45]})
sdmap = tra[C.PIXELS].std().to_numpy().reshape(8, 8)
im = ax[0].imshow(sdmap, cmap=C.GREENS)
for r in range(8):
    for c in range(8):
        ax[0].text(c, r, f"{sdmap[r, c]:.1f}", ha="center", va="center", fontsize=6.5,
                   color="white" if sdmap[r, c] > 5 else C.INK)
ax[0].set_xticks(range(8)); ax[0].set_yticks(range(8)); ax[0].grid(False)
ax[0].set_title("a  Standard deviation of each pixel")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
keep = [p for p in C.PIXELS if var[p] > 0]
rho = tra[keep].corr(method="spearman")
im = ax[1].imshow(rho.values, cmap=C.PIYG, vmin=-1, vmax=1)
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].grid(False)
ax[1].set_title(f"b  Spearman correlation among {len(keep)} non-constant pixels")
ax[1].set_xlabel("pixel index (row-major order)"); ax[1].set_ylabel("pixel index")
cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03); cb.set_label("rho (pink negative, green positive)")
fig.tight_layout()
C.save(fig, "d06_variability_correlation")
r = rho.where(~np.eye(len(rho), dtype=bool))
pos_pair = r.stack().idxmax(); neg_pair = r.stack().idxmin()
nb = []
for p in keep:
    rr, cc = int(p[1]), int(p[2])
    for q in (f"p{rr}{cc + 1}", f"p{rr + 1}{cc}"):
        if q in keep:
            nb.append(rho.loc[p, q])
far = [rho.loc[p, q] for i, p in enumerate(keep) for q in keep[i + 1:]
       if abs(int(p[1]) - int(q[1])) + abs(int(p[2]) - int(q[2])) >= 6]
out["corr"] = {"max_pair": list(pos_pair), "max_rho": round(float(r.stack().max()), 3),
               "min_pair": list(neg_pair), "min_rho": round(float(r.stack().min()), 3),
               "mean_abs_neighbour": round(float(np.mean(np.abs(nb))), 3),
               "mean_abs_far": round(float(np.mean(np.abs(far))), 3),
               "share_abs_gt_0_5": round(float((r.abs() > 0.5).stack().mean()), 4)}

# ------------------------------------------------------------------ bivariate: shape features for confusable digits
cols_idx = np.arange(8)
tot = np.maximum(Xtr.sum((1, 2)), 1)
cx = (Xtr.sum(1) * cols_idx).sum(1) / tot                                 # ink-weighted column centre
top = Xtr[:, :4, :].sum((1, 2)) / tot                                    # share of ink in the top half
lr = Xtr[:, :, :4].sum((1, 2)) / tot                                     # share of ink in the left half
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
for d, col, mk in [(1, C.GREEN, "o"), (7, C.PINK, "s"), (4, C.INK, "^")]:
    m = ytr == d
    ax[0].scatter(cx[m], top[m], s=9, color=col, alpha=0.45, marker=mk, lw=0, label=f"digit {d}")
ax[0].set_xlabel("Horizontal centre of ink (column 0-7)"); ax[0].set_ylabel("Share of ink in the top half")
ax[0].set_title("a  Where the ink sits: digits 1, 7 and 4"); ax[0].legend(markerscale=2)
for d, col in [(3, C.GREEN), (5, C.PINK), (8, C.INK)]:
    v = lr[ytr == d]
    kde = stats.gaussian_kde(v)
    gg = np.linspace(0, 1, 200)
    ax[1].plot(gg, kde(gg), color=col, lw=2, label=f"digit {d} (median {np.median(v):.2f})")
    ax[1].fill_between(gg, kde(gg), color=col, alpha=0.08)
ax[1].axvline(0.5, color=C.MUTED, ls=":", lw=1)
ax[1].set_xlabel("Share of ink in the left half of the image"); ax[1].set_ylabel("Density")
ax[1].set_title("b  Left-right balance of 3, 5 and 8"); ax[1].legend()
fig.tight_layout()
C.save(fig, "d07_shape_features")
out["top_share_median"] = {str(d): round(float(np.median(top[ytr == d])), 3) for d in range(10)}
out["centre_x_median"] = {str(d): round(float(np.median(cx[ytr == d])), 3) for d in range(10)}
out["left_share_median"] = {str(d): round(float(np.median(lr[ytr == d])), 3) for d in range(10)}
out["centre_top_spearman_147"] = round(float(stats.spearmanr(cx[np.isin(ytr, [1, 4, 7])], top[np.isin(ytr, [1, 4, 7])]).statistic), 3)

# ------------------------------------------------------------------ multivariate: PCA and t-SNE
Z = tra[C.PIXELS].to_numpy(float) / 16
pca = PCA(random_state=C.SEED).fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
P2 = pca.transform(Z)[:, :2]
out["pca"] = {"pc1": round(float(evr[0]), 4), "pc2": round(float(evr[1]), 4),
              "n80": int(np.searchsorted(cum, 0.80) + 1), "n90": int(np.searchsorted(cum, 0.90) + 1),
              "n95": int(np.searchsorted(cum, 0.95) + 1)}
emb = TSNE(n_components=2, perplexity=35, init="pca", random_state=C.SEED).fit_transform(Z)
fig = plt.figure(figsize=(11, 4.6))
a0 = fig.add_subplot(1, 3, 1)
a0.plot(range(1, 65), 100 * cum, color=C.GREEN, lw=2)
for th in (80, 90, 95):
    k = int(np.searchsorted(cum, th / 100) + 1)
    a0.plot([k], [th], "o", color=C.PINK, ms=6, mec="white")
    a0.annotate(f"{th}% at {k} PCs", (k, th), xytext=(8, -10), textcoords="offset points", fontsize=8)
a0.set_xlabel("Number of components"); a0.set_ylabel("Cumulative variance (%)")
a0.set_title("a  PCA: variance retained")
for j, (E, ttl) in enumerate([(P2, "b  First two principal components"), (emb, "c  t-SNE embedding")]):
    a = fig.add_subplot(1, 3, j + 2)
    idx = np.random.default_rng(C.SEED).choice(len(E), 1500, replace=False)
    for i in idx:
        a.text(E[i, 0], E[i, 1], str(ytr[i]), fontsize=6, color="#9a9a9a", ha="center", va="center")
    for d in range(10):
        cx, cy = np.median(E[ytr == d], axis=0)
        a.text(cx, cy, str(d), fontsize=13, fontweight="bold", color=C.GREEN, ha="center", va="center",
               bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=C.GREEN, lw=1))
    a.set_xlim(E[:, 0].min(), E[:, 0].max()); a.set_ylim(E[:, 1].min(), E[:, 1].max())
    a.set_xticks([]); a.set_yticks([]); a.set_title(ttl)
fig.tight_layout()
C.save(fig, "d08_pca_tsne")
# cluster separation in t-SNE space: share of points whose 10 nearest neighbours mostly share their label
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
knn_emb = KNeighborsClassifier(10).fit(emb, ytr)
out["tsne_neighbour_purity"] = round(float((knn_emb.predict(emb) == ytr).mean()), 4)

# ------------------------------------------------------------------ outliers: atypical handwriting
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=C.SEED).fit(Z)
score = -iso.score_samples(Z)                      # higher = more unusual
q1, q3 = np.percentile(score, [25, 75])
fence = q3 + 1.5 * (q3 - q1)
out["isoforest"] = {"fence": round(float(fence), 4), "n_above_fence": int((score > fence).sum()),
                    "pct_above_fence": round(100 * float((score > fence).mean()), 2),
                    "top_digits": np.bincount(ytr[np.argsort(-score)[:100]], minlength=10).tolist()}
fig = plt.figure(figsize=(11, 3.7))
a = fig.add_subplot(1, 3, 1)
a.hist(score, bins=50, color="#c9c9c9", edgecolor="white", lw=0.3)
a.axvline(fence, color=C.PINK, lw=1.6)
a.text(fence, a.get_ylim()[1] * 0.92, f" Tukey fence\n {(score > fence).sum()} images", fontsize=8, color=C.INK2)
a.set_xlabel("Isolation-forest anomaly score"); a.set_ylabel("Images"); a.set_title("a  Anomaly scores")
top = np.argsort(-score)[:12]
for k, i in enumerate(top):
    b = fig.add_subplot(2, 9, (k // 6) * 9 + 4 + k % 6)
    b.imshow(Xtr[i], cmap="gray_r", vmin=0, vmax=16)
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    b.set_title(f"label {ytr[i]}", fontsize=8, loc="center")
fig.text(0.68, 0.98, "b  The 12 most unusual training images (kept, not deleted)", fontsize=11, ha="center", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.93))
C.save(fig, "d09_outliers")

# ------------------------------------------------------------------ writer shift between training and test writers
diff = Xte.mean(0) - Xtr.mean(0)
t_stat = stats.ttest_ind(tes[C.PIXELS], tra[C.PIXELS], equal_var=False, nan_policy="omit")
pvals = np.nan_to_num(t_stat.pvalue, nan=1.0)
sig = (stats.false_discovery_control(pvals) < 0.05).reshape(8, 8)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5))
for a, img, ttl in [(ax[0], Xtr.mean(0), "a  Mean training image"), (ax[1], Xte.mean(0), "b  Mean test image")]:
    a.imshow(img, cmap="gray_r", vmin=0, vmax=10); a.set_title(ttl); a.set_xticks([]); a.set_yticks([]); a.grid(False)
lim = np.abs(diff).max()
im = ax[2].imshow(diff, cmap=C.PIYG, vmin=-lim, vmax=lim)
for rr in range(8):
    for cc in range(8):
        if sig[rr, cc]:
            ax[2].text(cc, rr, "*", ha="center", va="center", fontsize=10, color=C.INK)
ax[2].set_title("c  Test minus training (* FDR < 0.05)"); ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].grid(False)
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
fig.tight_layout()
C.save(fig, "d10_writer_shift")
out["writer_shift"] = {"max_abs_diff": round(float(lim), 3), "n_sig_pixels": int(sig.sum()),
                       "mean_ink_train": round(float(Xtr.sum((1, 2)).mean()), 1),
                       "mean_ink_test": round(float(Xte.sum((1, 2)).mean()), 1)}

# ------------------------------------------------------------------ univariate summary table (selected pixels)
sel = ["p02", "p13", "p20", "p33", "p34", "p36", "p43", "p52", "p63", "p66"]
summ = tra[sel].describe().T
summ["skew"] = tra[sel].skew()
summ["zeros_pct"] = (tra[sel] == 0).mean() * 100
summ.round(3).to_csv(C.RES / "pixel_summary.csv")
out["pixel_summary"] = {p: {k: round(float(v), 3) for k, v in summ.loc[p].items()} for p in sel}
C.dump("eda", out)
print({k: v for k, v in out.items() if k not in ("pixel_summary", "pixel_value_counts")})
