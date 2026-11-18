import json
import sys
import types
from datetime import datetime

import h5py
import numpy as np
import pytest


def _install_fake_slac_db(areas_to_devices, device_to_pvs):
    module = types.ModuleType("slac_db")
    device_mod = types.ModuleType("slac_db.device")

    def get_devices(area=None, include_inactive=False):
        return list(areas_to_devices.get(area, []))

    def get_all_addresses(device):
        return list(device_to_pvs.get(device, []))

    device_mod.get_devices = get_devices
    device_mod.get_all_addresses = get_all_addresses
    module.device = device_mod
    sys.modules["slac_db"] = module
    sys.modules["slac_db.device"] = device_mod


def _install_fake_epics(caget_many_fn):
    module = types.ModuleType("epics")
    module.caget_many = caget_many_fn
    sys.modules["epics"] = module


@pytest.fixture
def archiver(monkeypatch):
    _install_fake_slac_db(
        areas_to_devices={
            "GUNB": ["SOLN:IN20:121", "QUAD:IN20:121"],
            "HTR": ["QUAD:IN20:121", "QUAD:IN20:361"],
        },
        device_to_pvs={
            "SOLN:IN20:121": ["SOLN:IN20:121:BACT", "SOLN:IN20:121:BCTRL"],
            "QUAD:IN20:121": ["QUAD:IN20:121:BACT", "QUAD:IN20:121:BCTRL"],
            "QUAD:IN20:361": ["QUAD:IN20:361:BACT"],
        },
    )
    sys.modules.pop("ml_tto.archiving", None)
    sys.modules.pop("ml_tto.archiving.archiver", None)
    from ml_tto.archiving import archiver as mod

    yield mod


def test_build_pv_list_dedup_and_extras(archiver):
    pv_loc = archiver.build_pv_list(
        ["GUNB", "HTR"],
        extra_pvs=["OTRS:IN20:571:XRMS", "QUAD:IN20:121:BACT"],
    )
    pvs = [p for p, _ in pv_loc]
    locs = dict(pv_loc)

    assert pvs.count("QUAD:IN20:121:BACT") == 1
    assert locs["SOLN:IN20:121:BACT"] == "GUNB"
    assert locs["QUAD:IN20:121:BACT"] == "GUNB"
    assert locs["QUAD:IN20:361:BACT"] == "HTR"
    assert locs["OTRS:IN20:571:XRMS"] == ""
    assert pvs[-1] == "OTRS:IN20:571:XRMS"


def test_snapshot_writes_spec_conformant_h5(archiver, tmp_path):
    image = np.arange(12, dtype=np.float64).reshape(3, 4)

    def fake_caget(pvs, timeout=None):
        return [image if pv.endswith(":IMAGE") else float(i) for i, pv in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "snap.h5"
    archiver.snapshot(
        ["GUNB"],
        extra_pvs=["OTRS:IN20:571:IMAGE"],
        path=out,
        run_source="TEST",
    )

    with h5py.File(out, "r") as f:
        assert f.attrs["Data_Standard_Version"] == "0.1.2"
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        assert len(ids) == 1
        batch = f[ids[0]]
        assert tuple(batch.attrs["batch_dims"]) == (1,)
        assert batch.attrs["run_information_source"] == "TEST"

        obs = batch["observables"]
        ts = obs["Timestamp"]
        assert ts.shape == (1,)
        raw = ts[0]
        ts_str = raw.decode() if isinstance(raw, bytes) else raw
        datetime.fromisoformat(ts_str)
        assert ts.attrs["read_only"]
        assert ts.attrs["num_feature_dims"] == 0

        scalar = obs["SOLN:IN20:121:BACT"]
        assert scalar.shape == (1,)
        assert scalar.attrs["location"] == "GUNB"
        assert scalar.attrs["num_feature_dims"] == 0
        assert scalar.attrs["read_only"]

        img = obs["OTRS:IN20:571:IMAGE"]
        assert img.shape == (1, 3, 4)
        assert img.attrs["location"] == ""
        assert img.attrs["num_feature_dims"] == 2
        assert img.attrs["bin_size"] == 1.0
        assert img.attrs["offset"] == 0.0
        np.testing.assert_array_equal(img[0], image)


def test_continuous_appends_n_samples(archiver, tmp_path):
    counter = {"n": 0}

    def fake_caget(pvs, timeout=None):
        counter["n"] += 1
        return [float(counter["n"] + i) for i, _ in enumerate(pvs)]

    _install_fake_epics(fake_caget)

    out = tmp_path / "stream.h5"
    archiver.continuous(
        "GUNB",
        path=out,
        rate_hz=50.0,
        n_samples=5,
        metadata={"operator": "test"},
    )

    with h5py.File(out, "r") as f:
        ids = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["IDs"]]
        batch = f[ids[0]]
        assert tuple(batch.attrs["batch_dims"]) == (5,)
        notes = json.loads(batch.attrs["run_information_notes"])
        assert notes["metadata"] == {"operator": "test"}
        assert notes["requested_rate_hz"] == 50.0
        assert "achieved_rate_hz" in notes
        assert notes["achieved_rate_hz"] > 0

        obs = batch["observables"]
        ts = obs["Timestamp"]
        assert ts.shape == (5,)
        strs = [s.decode() if isinstance(s, bytes) else s for s in ts[:]]
        parsed = [datetime.fromisoformat(s) for s in strs]
        assert parsed == sorted(parsed)

        scalar_pvs = [k for k in obs.keys() if k != "Timestamp"]
        assert scalar_pvs, "expected at least one PV dataset"
        for pv in scalar_pvs:
            d = obs[pv]
            assert d.shape == (5,)
            assert np.all(np.diff(d[:]) == 1.0)
