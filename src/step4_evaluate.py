"""Step 4 - represent and refine: writer-independent evaluation of the CNN on the 1,797 test images.

Run after step 3:  python src/step4_evaluate.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, cohen_kappa_score)
from sklearn.preprocessing import label_binarize

import common as C

C.style()
P = np.load(C.RES / "predictions.npz")
p, y = P["p_cnn"], P["y"]
hist = json.loads((C.RES / "model.json").read_text(encoding="utf-8"))["history"]
pred = p.argmax(1)
conf = p.max(1)
tes = C.load_tables()[1]
Xte = C.to_images(tes)
out = {}

# ------------------------------------------------------------------ learning curves
ep = np.arange(1, len(hist["loss"]) + 1)
be = int(np.argmin(hist["val_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
for a, key, lab in [(ax[0], "loss", "Cross-entropy loss"), (ax[1], "accuracy", "Accuracy")]:
    a.plot(ep, hist[key], color=C.GREEN, label="Training")
    a.plot(ep, hist["val_" + key], color=C.PINK, label="Validation")
    a.axvline(be, color=C.MUTED, ls=":", lw=1)
    a.set_xlabel("Epoch"); a.set_ylabel(lab); a.legend()
ax[0].set_yscale("log"); C.plain_log(ax[0]); ax[0].set_title("a  Loss versus epoch (log scale)")
ax[1].set_ylim(0.9, 1.002); ax[1].set_title("b  Accuracy versus epoch")
ax[0].text(be, ax[0].get_ylim()[1] * 0.7, f" restored epoch {be}", fontsize=8, color=C.INK2)
fig.tight_layout()
C.save(fig, "m03_learning_curves")
out["curves"] = {"epochs": len(ep), "best_epoch": be, "train_loss": hist["loss"][be - 1], "val_loss": hist["val_loss"][be - 1],
                 "train_acc": hist["accuracy"][be - 1], "val_acc": hist["val_accuracy"][be - 1],
                 "first_epoch_val_acc": hist["val_accuracy"][0]}

# ------------------------------------------------------------------ headline metrics and baselines
def summary(prob):
    pr = prob.argmax(1)
    return {"accuracy": round(accuracy_score(y, pr), 4), "macro_f1": round(f1_score(y, pr, average="macro"), 4),
            "kappa": round(cohen_kappa_score(y, pr), 4), "log_loss": round(log_loss(y, np.clip(prob, 1e-7, 1)), 4),
            "macro_auc": round(roc_auc_score(y, prob, multi_class="ovr", average="macro"), 5),
            "errors": int((pr != y).sum())}


out["cnn"], out["knn"], out["logreg"] = summary(p), summary(P["p_knn"]), summary(P["p_lr"])
# McNemar test: CNN versus k-NN on the same test images
a_ok, b_ok = pred == y, P["p_knn"].argmax(1) == y
n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
mc = stats.binomtest(min(n01, n10), n01 + n10, 0.5) if n01 + n10 else None
out["mcnemar_cnn_vs_knn"] = {"cnn_right_knn_wrong": n01, "cnn_wrong_knn_right": n10,
                             "p_value": float(mc.pvalue) if mc else 1.0}
acc_ci = stats.binomtest(int((pred == y).sum()), len(y)).proportion_ci(method="wilson")
out["cnn"]["acc_ci95"] = [round(acc_ci.low, 4), round(acc_ci.high, 4)]

# ------------------------------------------------------------------ confusion matrix
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
rown = cm / cm.sum(1, keepdims=True)
shade = np.where(np.eye(10, dtype=bool), 0, rown)
im = ax.imshow(np.where(np.eye(10, dtype=bool), rown, np.nan), cmap=C.GREENS, vmin=0.8, vmax=1)
ax.imshow(np.where(np.eye(10, dtype=bool), np.nan, shade), cmap=C.PIYG.reversed(), vmin=-0.06, vmax=0.06)
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if i == j else C.INK, fontweight="bold" if i != j else "normal")
ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False)
ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
ax.set_title("Counts; green = correct, pink = confusions")
fig.tight_layout()
C.save(fig, "m04_confusion_matrix")
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)
out["confusion_top"] = [{"true": i, "pred": j, "n": n} for n, i, j in pairs[:6]]

# ------------------------------------------------------------------ per-class precision / recall / F1
pr_, rc_, f1_, sup = precision_recall_fscore_support(y, pred)
fig, ax = plt.subplots(figsize=(9, 3.4))
xs = np.arange(10)
ax.plot(xs - 0.15, 100 * pr_, "o", color=C.GREEN, ms=7, label="Precision")
ax.plot(xs, 100 * rc_, "s", color=C.PINK, ms=7, label="Recall")
ax.plot(xs + 0.15, 100 * f1_, "D", color=C.INK, ms=5, label="F1 score")
for x_ in xs:
    ax.plot([x_ - 0.25, x_ + 0.25], [100, 100], color=C.RULE, lw=0.8)
ax.set_xticks(xs); ax.set_xlabel("Digit class"); ax.set_ylabel("Score (%)")
ax.set_ylim(min(95, 100 * min(pr_.min(), rc_.min()) - 1), 100.6)
ax.legend(loc="lower left", ncol=3)
fig.tight_layout()
C.save(fig, "m05_per_class_scores")
out["per_class"] = {str(d): {"precision": round(float(pr_[d]), 4), "recall": round(float(rc_[d]), 4),
                             "f1": round(float(f1_[d]), 4), "support": int(sup[d])} for d in range(10)}

# ------------------------------------------------------------------ ROC curves (one-vs-rest)
Y = label_binarize(y, classes=range(10))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
aucs = {}
for d in range(10):
    fpr, tpr, _ = roc_curve(Y[:, d], p[:, d])
    aucs[d] = roc_auc_score(Y[:, d], p[:, d])
    for a in ax:
        a.plot(fpr, tpr, color="#bdbdbd", lw=1)
fpr_m, tpr_m, _ = roc_curve(Y.ravel(), p.ravel())
auc_micro = roc_auc_score(Y, p, average="micro")
fk, tk, _ = roc_curve(Y.ravel(), P["p_knn"].ravel())
for a in ax:
    a.plot(fpr_m, tpr_m, color=C.GREEN, lw=2.2, label=f"CNN micro-average (AUC = {auc_micro:.4f})")
    a.plot(fk, tk, color=C.PINK, lw=1.8, ls="--", label=f"k-NN micro-average (AUC = {roc_auc_score(Y, P['p_knn'], average='micro'):.4f})")
ax[0].plot([0, 1], [0, 1], color=C.MUTED, ls=":", lw=1)
ax[0].set_title("a  Full ROC space (grey: the ten digits)")
ax[1].set_xscale("log"); ax[1].set_xlim(1e-4, 1); C.plain_log(ax[1], "x"); ax[1].set_ylim(0.8, 1.005)
ax[1].set_title("b  Zoom: log false-positive rate")
for a in ax:
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
ax[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
C.save(fig, "m06_roc_curves")
out["auc_per_class"] = {str(d): round(float(v), 5) for d, v in aucs.items()}
out["auc_micro"] = round(float(auc_micro), 5)

# ------------------------------------------------------------------ misclassified images
wrong = np.where(pred != y)[0]
wrong = wrong[np.argsort(-conf[wrong])]
k = min(len(wrong), 24)
cols = 8
rows_ = int(np.ceil(k / cols))
fig, axes = plt.subplots(rows_, cols, figsize=(10, 1.55 * rows_ + 0.2))
axes = np.atleast_2d(axes)
for a in axes.ravel():
    a.axis("off")
for n_, i in enumerate(wrong[:k]):
    a = axes.ravel()[n_]
    a.imshow(Xte[i], cmap="gray_r", vmin=0, vmax=16)
    a.set_title(f"true {y[i]} → {pred[i]}\n{100 * conf[i]:.0f}% sure", fontsize=8, loc="center",
                color=C.INK if conf[i] < 0.9 else "#b0336a")
fig.tight_layout()
C.save(fig, "m07_misclassified")
out["n_errors"] = int(len(wrong))
out["errors_over_90pct_conf"] = int((conf[wrong] > 0.9).sum())

# ------------------------------------------------------------------ confidence, calibration and error distribution
bins = np.linspace(0, 1, 11)
ids = np.clip(np.digitize(conf, bins) - 1, 0, 9)
acc_b = np.array([np.mean(pred[ids == b] == y[ids == b]) if (ids == b).any() else np.nan for b in range(10)])
conf_b = np.array([conf[ids == b].mean() if (ids == b).any() else np.nan for b in range(10)])
cnt_b = np.bincount(ids, minlength=10)
ece = float(np.nansum(cnt_b / len(y) * np.abs(acc_b - conf_b)))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
ax[0].plot([0, 1], [0, 1], color=C.MUTED, ls=":", lw=1, label="Perfect calibration")
m = (~np.isnan(acc_b)) & (cnt_b >= 5)          # bins with fewer than 5 images are not plotted
ax[0].plot(conf_b[m], acc_b[m], "o-", color=C.GREEN, ms=6, label=f"CNN (ECE = {100 * ece:.2f}%)")
for cb_, ab_, n_ in zip(conf_b[m], acc_b[m], cnt_b[m]):
    ax[0].annotate(f"n={n_}", (cb_, ab_), xytext=(4, -10), textcoords="offset points", fontsize=7, color=C.INK2)
ax[0].set_xlabel("Mean predicted confidence in bin"); ax[0].set_ylabel("Observed accuracy in bin")
ax[0].set_title("a  Reliability diagram"); ax[0].legend(loc="upper left")
hb = np.linspace(0.3, 1, 36)
ax[1].hist(conf[pred == y], bins=hb, color=C.GREEN, alpha=0.75, label=f"Correct ({(pred == y).sum():,})")
ax[1].hist(conf[pred != y], bins=hb, color=C.PINK, alpha=0.95, label=f"Wrong ({(pred != y).sum()})")
ax[1].set_yscale("log"); C.plain_log(ax[1]); ax[1].set_xlabel("Confidence of the predicted class"); ax[1].set_ylabel("Images (log scale)")
ax[1].set_title("b  Confidence of right and wrong answers"); ax[1].legend(loc="upper left")
fig.tight_layout()
C.save(fig, "m08_calibration")
out["ece"] = round(ece, 4)
out["median_conf_correct"] = round(float(np.median(conf[pred == y])), 4)
out["median_conf_wrong"] = round(float(np.median(conf[pred != y])), 4) if len(wrong) else None
thr = 0.9
keep = conf >= thr
out["selective_0_9"] = {"coverage": round(float(keep.mean()), 4), "accuracy": round(float((pred[keep] == y[keep]).mean()), 5)}

# ------------------------------------------------------------------ what the network learned: embedding map
emb = TSNE(n_components=2, perplexity=35, init="pca", random_state=C.SEED).fit_transform(P["emb"])
fig, ax = plt.subplots(figsize=(6.6, 5.2))
for i in range(len(y)):
    ax.text(emb[i, 0], emb[i, 1], str(y[i]), fontsize=6.5, color="#8f8f8f", ha="center", va="center")
for i in wrong:
    ax.text(emb[i, 0], emb[i, 1], str(y[i]), fontsize=9, color="#b0336a", fontweight="bold", ha="center", va="center")
for d in range(10):
    cx, cy = np.median(emb[y == d], axis=0)
    ax.text(cx, cy, str(d), fontsize=13, fontweight="bold", color=C.GREEN, ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=C.GREEN, lw=1))
ax.set_xlim(emb[:, 0].min() - 2, emb[:, 0].max() + 2); ax.set_ylim(emb[:, 1].min() - 2, emb[:, 1].max() + 2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Test images in the network's 64-d embedding (pink: misclassified)")
fig.tight_layout()
C.save(fig, "m09_embedding_tsne")

# ------------------------------------------------------------------ model comparison
fig, ax = plt.subplots(figsize=(8.2, 3.1))
names = ["Logistic regression", f"k-NN (k = {json.loads((C.RES / 'model.json').read_text())['knn_k']})", "CNN (this work)"]
vals = [out["logreg"]["accuracy"], out["knn"]["accuracy"], out["cnn"]["accuracy"]]
errs = [out["logreg"]["errors"], out["knn"]["errors"], out["cnn"]["errors"]]
ax.barh(names, [100 * v for v in vals], color=["#cfcfcf", "#cfcfcf", C.GREEN], height=0.5)
for k_, (v, e) in enumerate(zip(vals, errs)):
    ax.text(100 * v + 0.05, k_, f"{100 * v:.2f}%  ({e} errors)", va="center", fontsize=9)
ax.set_xlim(min(vals) * 100 - 2, 100.8); ax.set_xlabel("Writer-independent test accuracy (%)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
C.save(fig, "m10_model_comparison")

C.dump("evaluation", out)
print(json.dumps({k: out[k] for k in ("cnn", "knn", "logreg", "mcnemar_cnn_vs_knn", "ece", "n_errors", "confusion_top")}, indent=1))
