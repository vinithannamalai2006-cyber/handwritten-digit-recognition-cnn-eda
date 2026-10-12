"""Step 3 - mine: train a convolutional neural network, tune it and analyse what it looks at.

Data use follows the original authors: the 3,823 training images (30 writers) are split into
training and validation sets; the 1,797 images from 13 *different* writers stay untouched
for the final writer-independent test in step 4.

Run:  python src/step3_train_cnn.py
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import common as C

C.style()
keras.utils.set_random_seed(C.SEED)
tf.config.experimental.enable_op_determinism()
FAST = os.environ.get("FAST_RUN") == "1"

tra, tes = C.load_tables()
X = C.to_images(tra)[..., None] / 16.0            # scale counts 0-16 to 0-1, add channel axis
y = tra["digit"].to_numpy()
Xte = C.to_images(tes)[..., None] / 16.0
yte = tes["digit"].to_numpy()
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=C.SEED)
C.dump("model", {"split": {"train": int(len(Xtr)), "validation": int(len(Xva)), "test": int(len(Xte))}})


def build(filters=(32, 64), dropout=0.25, augment=False, lr=1e-3):
    inp = keras.Input((8, 8, 1), name="image_8x8")
    x = inp
    if augment:                                   # shift by up to one pixel (1/8 of the image)
        x = layers.RandomTranslation(0.125, 0.125, fill_mode="constant", name="shift_augment")(x)
    for k in range(2):
        x = layers.Conv2D(filters[0], 3, padding="same", use_bias=False, name=f"conv1_{k + 1}")(x)
        x = layers.BatchNormalization(name=f"bn1_{k + 1}")(x)
        x = layers.ReLU(name=f"relu1_{k + 1}")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)                      # 8x8 -> 4x4
    x = layers.Dropout(dropout / 2, name="drop1")(x)
    x = layers.Conv2D(filters[1], 3, padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)                      # 4x4 -> 2x2
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="embedding")(x)
    x = layers.Dropout(dropout, name="drop2")(x)
    out = layers.Dense(10, activation="softmax", name="digit_probabilities")(x)
    m = keras.Model(inp, out, name="digit_cnn")
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit(model, epochs):
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
    return model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, callbacks=cb, verbose=0)


# ------------------------------------------------------------------ basic hyper-parameter search
grid = [dict(filters=f, dropout=d, augment=a) for f in [(16, 32), (32, 64)] for d in (0.25, 0.5) for a in (False, True)]
if FAST:
    grid = grid[:2]
EPOCHS = 15 if FAST else 120
rows, best = [], None
t0 = time.time()
for i, cfg in enumerate(grid, 1):
    keras.utils.set_random_seed(C.SEED)
    m = build(**cfg)
    h = fit(m, EPOCHS)
    va_loss, va_acc = m.evaluate(Xva, yva, verbose=0)
    row = {"config": i, "filters": f"{cfg['filters'][0]}-{cfg['filters'][1]}", "dropout": cfg["dropout"],
           "augment": cfg["augment"], "epochs": len(h.history["loss"]),
           "best_epoch": int(np.argmin(h.history["val_loss"]) + 1), "val_acc": round(float(va_acc), 4),
           "val_loss": round(float(va_loss), 4), "params": int(m.count_params())}
    rows.append(row)
    print(row)
    key = (row["val_acc"], -row["val_loss"])
    if best is None or key > best[0]:
        best = (key, row, cfg, m, h.history)
_, brow, bcfg, model, hist = best
model.save(C.RES / "cnn_best.keras")
C.dump("model", {"tuning": rows, "tuning_seconds": round(time.time() - t0, 1), "best": brow,
                 "history": {k: [round(float(v), 5) for v in vals] for k, vals in hist.items()}})
model.summary(print_fn=lambda s, **kw: open(C.RES / "cnn_summary.txt", "a", encoding="utf-8").write(s + "\n")
              if s.strip() else None)

fig, ax = plt.subplots(figsize=(9, 3.4))
labels = [f"{r['filters']}\nd={r['dropout']}\n{'shift' if r['augment'] else 'no aug.'}" for r in rows]
cols = [C.GREEN if r["config"] == brow["config"] else "#cfcfcf" for r in rows]
ax.bar(range(len(rows)), [100 * r["val_acc"] for r in rows], color=cols, width=0.6)
for k, r in enumerate(rows):
    ax.text(k, 100 * r["val_acc"] + 0.05, f"{100 * r['val_acc']:.2f}", ha="center", fontsize=8, color=C.INK2)
lo = min(100 * r["val_acc"] for r in rows)
ax.set_ylim(lo - 1, 100.2)
ax.set_xticks(range(len(rows)), labels, fontsize=8)
ax.set_ylabel("Validation accuracy (%)")
ax.set_xlabel("Filters / dropout / augmentation (selected model in green)")
fig.tight_layout()
C.save(fig, "m01_tuning")

# ------------------------------------------------------------------ feature analysis: occlusion sensitivity
p_va = model.predict(Xva, verbose=0)
base = p_va[np.arange(len(yva)), yva]
occ = np.zeros((8, 8))
occ_cls = np.zeros((10, 8, 8))
for r in range(8):
    for c in range(8):
        Xo = Xva.copy()
        Xo[:, r, c, 0] = 0.0                       # erase one pixel (one 4x4 block of the original page)
        drop = base - model.predict(Xo, verbose=0)[np.arange(len(yva)), yva]
        occ[r, c] = drop.mean()
        for d in range(10):
            occ_cls[d, r, c] = drop[yva == d].mean()
C.dump("model", {"occlusion": {"max": round(float(occ.max()), 4), "argmax": [int(v) for v in np.unravel_index(occ.argmax(), occ.shape)],
                               "border_mean": round(float(np.r_[occ[0], occ[-1], occ[:, 0], occ[:, -1]].mean()), 5),
                               "centre_mean": round(float(occ[2:6, 2:6].mean()), 5),
                               "per_class_max": [round(float(occ_cls[d].max()), 4) for d in range(10)]}})
np.save(C.RES / "occlusion_by_class.npy", occ_cls)
fig = plt.figure(figsize=(11, 4.4))
a = fig.add_subplot(1, 2, 1)
im = a.imshow(occ, cmap=C.GREENS)
for r in range(8):
    for c in range(8):
        a.text(c, r, f"{100 * occ[r, c]:.1f}", ha="center", va="center", fontsize=6.5,
               color="white" if occ[r, c] > 0.6 * occ.max() else C.INK)
a.set_xticks(range(8)); a.set_yticks(range(8)); a.grid(False)
a.set_title("a  Mean drop in true-class probability (%)")
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
for d in range(10):
    b = fig.add_subplot(2, 10, (d // 5) * 10 + 6 + d % 5)
    b.imshow(np.clip(occ_cls[d], 0, None), cmap=C.GREENS, vmin=0, vmax=max(occ_cls[d].max(), 1e-6))
    b.contour(Xva[yva == d, :, :, 0].mean(0), levels=[0.5], colors=[C.PINK], linewidths=0.8)
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    b.set_title(f"{d} (max {100 * occ_cls[d].max():.0f}%)", fontsize=8, loc="center")
fig.text(0.75, 0.97, "b  Per-class maps, each on its own scale (pink: mean digit outline)", fontsize=10.5, ha="center", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.92))
C.save(fig, "m02_occlusion")

# ------------------------------------------------------------------ baselines on the same split
Ftr, Fva, Fte = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1)
knn_scores = {k: KNeighborsClassifier(k).fit(Ftr, ytr).score(Fva, yva) for k in (1, 3, 5, 7)}
k_best = max(knn_scores, key=lambda k: (knn_scores[k], -k))
knn = KNeighborsClassifier(k_best).fit(Ftr, ytr)
lr = LogisticRegression(max_iter=5000, C=1.0).fit(Ftr, ytr)

# ------------------------------------------------------------------ predictions for step 4
p_te = model.predict(Xte, verbose=0)
emb_model = keras.Model(model.input, model.get_layer("embedding").output)
emb_te = emb_model.predict(Xte, verbose=0)
np.savez_compressed(C.RES / "predictions.npz", p_cnn=p_te, p_knn=knn.predict_proba(Fte), p_lr=lr.predict_proba(Fte),
                    y=yte, emb=emb_te, p_va=p_va, y_va=yva)
C.dump("model", {"knn_val_acc": {str(k): round(float(v), 4) for k, v in knn_scores.items()}, "knn_k": int(k_best),
                 "lr_val_acc": round(float(lr.score(Fva, yva)), 4),
                 "versions": {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__}})
print("best", brow, "knn", knn_scores)
