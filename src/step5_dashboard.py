"""Step 5 - interact: build the self-contained Digit Lab page (dashboard/index.html).

Run after step 4:  python src/step5_dashboard.py
"""
import json

import numpy as np

import common as C

tra, tes = C.load_tables()
P = np.load(C.RES / "predictions.npz")
data = {
    "train_x": tra[C.PIXELS].to_numpy().astype(int).tolist(),
    "train_y": tra["digit"].astype(int).tolist(),
    "test_x": tes[C.PIXELS].to_numpy().astype(int).tolist(),
    "test_y": tes["digit"].astype(int).tolist(),
    "test_p": np.round(P["p_cnn"], 4).tolist(),
}
tpl = (C.ROOT / "src" / "dashboard_template.html").read_text(encoding="utf-8")
(C.DASH / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
print("written", C.DASH / "index.html")
