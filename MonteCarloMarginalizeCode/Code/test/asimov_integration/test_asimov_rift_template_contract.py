import configparser
import pathlib
import random
import types

import pytest


TEMPLATE = pathlib.Path(__file__).resolve().parents[2] / "RIFT" / "asimov" / "rift.ini"


def _liquid_render(template_text, context):
    liquid = pytest.importorskip("liquid")

    if hasattr(liquid, "Environment"):
        env = liquid.Environment()
        template = env.from_string(template_text)
        return template.render(**context)

    if hasattr(liquid, "Liquid"):
        template = liquid.Liquid(template_text, from_file=False)
        return template.render(**context)

    if hasattr(liquid, "Template"):
        template = liquid.Template(template_text)
        return template.render(**context)

    pytest.skip("installed liquid package does not expose a supported renderer")


def _base_meta():
    return {
        "name": "rift-v5PHM-calmarg",
        "engine": "RIFT",
        "interferometers": ["H1", "L1"],
        "event time": 1240340820.676,
        "data": {
            "channels": {
                "H1": "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                "L1": "L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
            },
            "frame types": {
                "H1": "H1_HOFT_CLEAN_SUB60HZ_C01",
                "L1": "L1_HOFT_CLEAN_SUB60HZ_C01",
            },
            "segment length": 4,
        },
        "quality": {
            "minimum frequency": {"H1": 20, "L1": 20},
            "maximum frequency": {"H1": 1700, "L1": 1700},
        },
        "likelihood": {
            "start frequency": 20,
            "sample rate": 1024,
            "assume": {},
            "marginalization": {"distance": False},
        },
        "waveform": {
            "approximant": "SEOBNRv5PHM",
            "pn amplitude order": 5,
            "maximum mode": 4,
            "gwsignal arguments": {
                "lmax_nyquist": 1,
                "enable_antisymmetric_modes": True,
            },
        },
        "priors": {
            "spin 1": {"maximum": 0.99},
            "spin 2": {"maximum": 0.99},
            "chirp mass": {"minimum": 33.4, "maximum": 233.4},
            "mass ratio": {"minimum": 0.05, "maximum": 1.0},
            "mass 1": {"minimum": 1, "maximum": 1000},
            "luminosity distance": {
                "minimum": 10,
                "maximum": 20000,
                "type": "bilby.gw.prior.UniformSourceFrame",
            },
        },
        "sampler": {
            "cip": {"sampling method": "AV", "explode jobs": 5},
            "ile": {
                "n eff": 12,
                "copies": 2,
                "sampling method": "AV",
                "freezeadapt": False,
                "jobs per worker": 30,
            },
        },
        "scheduler": {
            "accounting group": "ligo.dev.o4.cbc.pe.rift",
            "osg": False,
        },
    }


def _render(meta):
    production = types.SimpleNamespace(
        name=meta["name"],
        meta=meta,
        category="C01_offline",
        event=types.SimpleNamespace(
            name="GW190426_190642",
            repository=types.SimpleNamespace(directory="/tmp/rift-asimov-repo"),
        ),
        xml_psds={
            ifo: f"/tmp/rift-asimov-repo/C01_offline/psds/psd_{ifo}.xml.gz"
            for ifo in meta["interferometers"]
        },
    )
    context = {
        "production": production,
        "config": {
            "general": {"webroot": "/tmp/rift-web"},
            "pipelines": {"environment": "/opt/igwn"},
            "condor": {"user": "riftci"},
        },
    }
    rendered = _liquid_render(TEMPLATE.read_text(), context)
    assert "{{" not in rendered
    assert "{%" not in rendered
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read_string(rendered)
    return rendered, parser


def test_rift_liquid_template_renders_realistic_baseline_ledger():
    meta = _base_meta()
    rendered, parser = _render(meta)

    assert parser.get("analysis", "ifos") == "['H1', 'L1']"
    assert parser.get("analysis", "engine") == "RIFT"
    assert parser.get("condor", "accounting_group") == "ligo.dev.o4.cbc.pe.rift"
    assert '"H1":"H1_HOFT_CLEAN_SUB60HZ_C01"' in parser.get("datafind", "types")
    assert '"L1":"L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"' in parser.get("data", "channels")
    assert parser.get("engine", "fref") == "20"
    assert parser.get("engine", "fmin-template") == "20"
    assert parser.get("engine", "approx").strip() == "SEOBNRv5PHM"
    assert parser.get("engine", "amporder") == "5"
    assert parser.get("engine", "seglen") == "4"
    assert parser.get("engine", "srate").strip() == "1024"
    assert parser.get("engine", "H1-psd") == '"/tmp/rift-asimov-repo/C01_offline/psds/psd_H1.xml.gz"'
    assert parser.get("engine", "chirpmass-min") == "33.4"
    assert parser.get("engine", "distance-max") == "20000"
    assert parser.get("rift-pseudo-pipe", "cip-explode-jobs").strip() == "5"
    assert parser.get("rift-pseudo-pipe", "ile-n-eff").strip() == "12"
    assert parser.get("rift-pseudo-pipe", "ile-copies").strip() == "2"
    assert parser.get("rift-pseudo-pipe", "use_osg").strip() == "False"
    assert "manual-extra-ile-args=--internal-waveform-extra-kwargs" in rendered


@pytest.mark.parametrize(
    "distance_prior,expected",
    [
        ("bilby.gw.prior.UniformSourceFrame", "cosmo_sourceframe"),
        ("bilby.gw.prior.UniformComovingVolume", "cosmo"),
        ("Uniform", "pseudo_cosmo"),
    ],
)
def test_rift_liquid_template_distance_prior_blocks(distance_prior, expected):
    meta = _base_meta()
    meta["priors"]["luminosity distance"]["type"] = distance_prior
    _rendered, parser = _render(meta)
    assert parser.get("rift-pseudo-pipe", "ile-distance-prior").strip("'\"") == expected


def test_rift_liquid_template_option_blocks_land_safely():
    meta = _base_meta()
    meta["data"]["calibration"] = {
        "H1": "/tmp/cal/H1.dat",
        "L1": "/tmp/cal/L1.dat",
    }
    meta["likelihood"]["assume"] = {
        "eccentric": True,
        "nonprecessing": True,
        "matter secondary": True,
    }
    meta["likelihood"]["marginalization"] = {
        "distance": True,
        "distance lookup": "/tmp/lookup.npy",
    }
    meta["likelihood"]["roll off time"] = 1.0
    meta["sampler"]["likelihood"] = {
        "calibration": {
            "sample": True,
            "bilby ini file": "/tmp/bilby.ini",
        }
    }
    meta["sampler"]["extra eccentric arguments"] = {
        "force-ecc-min": 0.0,
        "force-ecc-max": 0.2,
    }
    meta["sampler"]["manual grid"] = "/tmp/initial-grid.xml.gz"
    meta["sampler"]["force iterations"] = 3
    meta["sampler"]["correlate parameters default"] = True
    meta["sampler"]["use rescaled transverse spin coordinates"] = True
    meta["sampler"]["n output samples"] = 7000
    meta["sampler"]["n output samples last"] = 25000
    meta["sampler"]["n input samples"] = 4000
    meta["sampler"]["ile"]["manual extra args"] = ["--zero-likelihood", "--vectorized"]
    meta["sampler"]["ile"]["rotate phase"] = True
    meta["sampler"]["ile"]["request disk"] = "2 GB"
    meta["sampler"]["cip"]["request disk"] = "1 GB"
    meta["sampler"]["ile"]["condor commands"] = {
        "request_memory": "4096",
        "+WantOSG": "True",
    }
    meta["scheduler"]["osg"] = True
    meta["scheduler"]["additional files"] = ["/tmp/waveform-cache.h5", "/tmp/table.dat"]

    rendered, parser = _render(meta)

    assert parser.get("engine", "enable-spline-calibration") == ""
    assert parser.get("engine", "H1-spcal-envelope") == '"/tmp/cal/H1.dat"'
    assert parser.get("engine", "ecc_min") == "0.0"
    assert parser.get("engine", "ecc_max") == "0.2"
    assert parser.get("rift-pseudo-pipe", "calibration-reweighting") == "True"
    assert parser.get("rift-pseudo-pipe", "bilby-ini-file") == '"/tmp/bilby.ini"'
    assert parser.get("rift-pseudo-pipe", "internal-force-iterations") == "3"
    assert parser.get("rift-pseudo-pipe", "internal-correlate-default") == "True"
    assert parser.get("rift-pseudo-pipe", "internal-use-rescaled-transverse-spin-coordinates") == "True"
    assert parser.get("rift-pseudo-pipe", "internal-ile-rotate-phase") == "True"
    assert parser.get("rift-pseudo-pipe", "assume-eccentric") == "True"
    assert parser.get("rift-pseudo-pipe", "assume-nonprecessing") == "True"
    assert parser.get("rift-pseudo-pipe", "assume-matter-but-primary-bh") == "True"
    assert parser.get("rift-pseudo-pipe", "use-meanPerAno") == "True"
    assert parser.get("rift-pseudo-pipe", "force-ecc-max") == "0.2"
    assert parser.get("rift-pseudo-pipe", "internal-marginalize-distance") == "True"
    assert parser.get("rift-pseudo-pipe", "internal-marginalize-distance-file") == "/tmp/lookup.npy"
    assert parser.get("rift-pseudo-pipe", "internal-ile-data-tukey-window-time") == "1.0"
    assert parser.get("rift-pseudo-pipe", "manual-initial-grid") == "'/tmp/initial-grid.xml.gz'"
    assert parser.get("rift-pseudo-pipe", "use_osg").strip() == "True"
    assert parser.get("rift-pseudo-pipe", "internal-use-oauth-files").strip() == "'scitokens'"
    assert parser.get("rift-pseudo-pipe", "n-output-samples").strip() == "7000"
    assert parser.get("rift-pseudo-pipe", "n-output-samples-last").strip() == "25000"
    assert parser.get("rift-pseudo-pipe", "internal-n-evaluations-per-iteration") == "4000"
    assert parser.get("rift-pseudo-pipe", "internal-ile-request-disk") == '"2 GB"'
    assert parser.get("rift-pseudo-pipe", "internal-cip-request-disk") == '"1 GB"'
    assert parser.get("rift-ile-condor", "request_memory") == "4096"
    assert parser.get("rift-ile-condor", "+WantOSG") == "True"
    assert "ile-additional-files-to-transfer=" in rendered
    assert "--zero-likelihood" in parser.get("rift-pseudo-pipe", "manual-extra-ile-args")


def test_rift_liquid_template_randomized_ledger_sanity():
    rng = random.Random(190426)
    approximants = ["SEOBNRv5PHM", "IMRPhenomXPHM", "TaylorF2"]
    sample_rates = [512, 1024, 2048]

    for _ in range(20):
        meta = _base_meta()
        ifos = rng.sample(["H1", "L1", "V1"], rng.choice([2, 3]))
        meta["interferometers"] = ifos
        meta["data"]["channels"] = {ifo: f"{ifo}:TEST-STRAIN" for ifo in ifos}
        meta["data"]["frame types"] = {ifo: f"{ifo}_TEST_FRAME" for ifo in ifos}
        meta["quality"]["minimum frequency"] = {ifo: rng.choice([10, 15, 20]) for ifo in ifos}
        meta["quality"]["maximum frequency"] = {ifo: rng.choice([512, 1024, 1700]) for ifo in ifos}
        meta["likelihood"]["start frequency"] = rng.choice([10, 15, 20, 30])
        meta["likelihood"]["sample rate"] = rng.choice(sample_rates)
        meta["waveform"]["approximant"] = rng.choice(approximants)
        if rng.choice([True, False]):
            meta["waveform"]["reference frequency"] = rng.choice([20, 30, 40])
        meta["waveform"]["pn amplitude order"] = rng.choice([0, 3, 5])
        meta["waveform"]["maximum mode"] = rng.choice([2, 3, 4])
        meta["priors"]["spin 1"]["maximum"] = round(rng.uniform(0.2, 0.99), 3)
        meta["priors"]["spin 2"]["maximum"] = round(rng.uniform(0.2, 0.99), 3)
        mc_min = round(rng.uniform(1.0, 80.0), 3)
        meta["priors"]["chirp mass"] = {"minimum": mc_min, "maximum": round(mc_min + rng.uniform(10.0, 200.0), 3)}
        meta["priors"]["luminosity distance"]["maximum"] = rng.choice([1000, 5000, 20000])
        meta["sampler"]["cip"]["explode jobs"] = rng.randint(1, 8)
        meta["sampler"]["ile"]["n eff"] = rng.randint(5, 50)
        meta["scheduler"]["osg"] = rng.choice([True, False])

        _rendered, parser = _render(meta)

        assert parser.get("engine", "approx").strip() == meta["waveform"]["approximant"]
        assert parser.get("engine", "fmin-template") == str(meta["likelihood"]["start frequency"])
        expected_fref = meta["waveform"].get("reference frequency", meta["likelihood"]["start frequency"])
        assert parser.get("engine", "fref") == str(expected_fref)
        assert parser.get("engine", "srate").strip() == str(meta["likelihood"]["sample rate"])
        assert parser.get("engine", "amporder") == str(meta["waveform"]["pn amplitude order"])
        assert parser.get("rift-pseudo-pipe", "l-max") == str(meta["waveform"]["maximum mode"])
        assert parser.get("rift-pseudo-pipe", "use_osg").strip() == str(meta["scheduler"]["osg"])
        assert parser.get("rift-pseudo-pipe", "cip-explode-jobs").strip() == str(meta["sampler"]["cip"]["explode jobs"])
        assert parser.get("rift-pseudo-pipe", "ile-n-eff").strip() == str(meta["sampler"]["ile"]["n eff"])
        for ifo in ifos:
            assert f'"{ifo}":"{ifo}_TEST_FRAME"' in parser.get("datafind", "types")
            assert f'"{ifo}":"{ifo}:TEST-STRAIN"' in parser.get("data", "channels")
