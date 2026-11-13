import json
from pathlib import Path

from fair_universe_demo.tasks.histogram import HistogramTask


def test_parse_snapshot_flat_format(tmp_path: Path) -> None:
    snapshot = {
        "est=nf_signal_1jet&syst=c_0.5&ensem=0&fold=0": "/x/nf1.ckpt",
        "est=nf_background_1jet&syst=c_0.5&ensem=0&fold=0": "/x/nf2.ckpt",
        "est=classifier&syst=nominal&ensem=0&fold=0": "/x/clf.ckpt",
    }
    path = tmp_path / "dag_snapshot.json"
    path.write_text(json.dumps(snapshot))

    nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(str(path))

    assert set(nf_ckpts) == {
        "est=nf_signal_1jet&syst=c_0.5&ensem=0&fold=0",
        "est=nf_background_1jet&syst=c_0.5&ensem=0&fold=0",
    }
    assert classifier_ckpt == {"classifier": "/x/clf.ckpt"}
