"""Independent cohort audit guards and finite-sample statistics."""

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"scripts"))
from audit_current_v7_rgbd_evidence import fit_circle, inventory, paired_counts, randomization_audit, wilson95


def test_wilson_interval_does_not_treat_30_of_30_as_proven_100_percent():
    low, high = wilson95(30, 30)
    assert low == pytest.approx(.8864866068260314)
    assert high == pytest.approx(1.)
    assert wilson95(25, 30) == pytest.approx([.6643564949358398, .9266345762815146])
    with pytest.raises(ValueError):
        wilson95(0, 0)


def test_mcnemar_uses_only_same_seed_discordant_pairs():
    first = {i: i >= 5 for i in range(30)}
    second = {i: True for i in range(30)}
    result = paired_counts(first, second)
    assert result["both_success"] == 25
    assert result["first_only"] == 0
    assert result["second_only"] == 5
    assert result["exact_two_sided_mcnemar_p"] == .0625
    assert paired_counts(second, second)["exact_two_sided_mcnemar_p"] == 1.
    with pytest.raises(ValueError):
        paired_counts(first, {0: True})
    with pytest.raises(ValueError):
        paired_counts({0: None}, {0: True})


def test_independent_circle_fit_recovers_partial_known_radius_surface():
    centre = np.array([.777, -.051])
    theta = np.linspace(.2, 1.8, 100)
    points = centre + .012*np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    fitted, residual = fit_circle(points, .012, [.78, -.05])
    assert fitted == pytest.approx(centre, abs=1e-9)
    assert residual < 1e-10
    with pytest.raises(ValueError):
        fit_circle(points, 0., [.78, -.05])


def test_inventory_refuses_complete_flag_with_missing_trace(tmp_path):
    row = {"seed": 1600, "mode": "rgbd_shared", "path": "seed_1600/rgbd_shared",
           "artifacts": {"trace": {"path": "trace.json.gz", "bytes": 10}}}
    protocol = {"seeds": [1600], "modes": ["rgbd_shared"]}
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    (tmp_path/"summary.json").write_text(json.dumps({"complete": True, "protocol": protocol, "results": [row]}))
    state, _, _ = inventory(tmp_path)
    assert not state["accepted"]
    assert "seed_1600/rgbd_shared/trace.json.gz" in state["missing_or_partial_files"]


def test_randomization_checks_seed_draw_actual_manifest_and_scene(tmp_path):
    seed = 1600
    rng = random.Random(0x50620000+seed)
    draw = {"xy_error_m": [rng.uniform(-.005, .005), rng.uniform(-.005, .005)],
            "mass_factor": rng.uniform(.8, 1.2), "friction_factor": rng.uniform(.8, 1.2)}
    target = {"object_id": "tool_marker", "role": "target", "pose": [.8, 0., .455],
              "mass_kg": .045, "friction": [.65, .01, .001], "shape": "cylinder", "dimensions": [.012, .075]}
    other = [{"object_id": f"other{i}", "role": "distractor" if i < 3 else "target"} for i in range(5)]
    changed = dict(target, pose=[.8+draw["xy_error_m"][0], draw["xy_error_m"][1], .455],
                   mass_kg=.045*draw["mass_factor"], friction=[v*draw["friction_factor"] for v in target["friction"]])
    scene = tmp_path/"scene.xml"
    scene.write_text('<mujoco><worldbody><body name="v7_obj_tool_marker" pos="'+" ".join(map(str, changed["pose"]))
        +'"><geom name="v7_geom_tool_marker" type="cylinder" mass="'+str(changed["mass_kg"])
        +'" friction="'+" ".join(map(str, changed["friction"]))+'" size="0.012 0.075"/></body></worldbody></mujoco>')
    nominal, actual = {"objects": [target]+other}, {"objects": [changed]+other}
    assert randomization_audit(seed, nominal, actual, {"perturbation": draw}, scene)["accepted"]
    assert not randomization_audit(seed+1, nominal, actual, {"perturbation": draw}, scene)["accepted"]
    changed["mass_kg"] *= 2
    assert not randomization_audit(seed, nominal, actual, {"perturbation": draw}, scene)["accepted"]
