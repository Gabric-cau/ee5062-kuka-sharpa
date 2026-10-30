"""Reproducible separate V2 task and unchanged hand-aligned baseline comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from kuka_sharpa_grasp.scene import build_scene
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene, WORKCELL_VERSION
from kuka_sharpa_grasp.pick_place import Scenario, run_task
from kuka_sharpa_grasp.rollout import GraspRollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/v2_pick_place')
    parser.add_argument('--summary', type=Path, default=PROJECT / 'results/v2_pick_place_summary.json')
    args = parser.parse_args()
    if args.summary.exists():
        parser.error('summary already exists; choose a new path to preserve previous results')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in list((PROJECT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
                     if not p.name.startswith('._')}
    scene = build_pick_place_scene(args.output / 'scene.xml', 'ycb_can')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    baseline_scene = build_scene(args.output / 'original_baseline_scene.xml', 'ycb_can')
    baseline_model = mujoco.MjModel.from_xml_path(str(baseline_scene))
    baseline_data = mujoco.MjData(baseline_model)
    trials, baseline = [], []
    started = time.perf_counter()
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.58, -.02, .52]
        camera.distance = 1.65
        camera.azimuth = 135
        camera.elevation = -25
        renderer.update_scene(data, camera=camera)
        renderer.render()
        import OpenGL.GL as gl
        gpu = {key: gl.glGetString(value).decode() for key, value in
               [('vendor', gl.GL_VENDOR), ('renderer', gl.GL_RENDERER), ('version', gl.GL_VERSION)]}
        for seed in range(args.seed, args.seed + args.seeds):
            folder = args.output / f'seed_{seed:03d}'
            folder.mkdir()
            frames = []

            def frame(stage, live):
                renderer.update_scene(live, camera=camera)
                pixels = renderer.render().copy()
                path = folder / f'{len(frames):02d}_{stage}.png'
                Image.fromarray(pixels).save(path)
                frames.append(dict(stage=stage, timestamp=float(live.time),
                                   file=path.name, pixel_std=float(pixels.std())))

            result, trace = run_task(model, data, Scenario(seed), frame_callback=frame)
            result['frames'] = frames
            (folder / 'trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
            (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            trials.append(result)
            old = GraspRollout(baseline_model, baseline_data)
            baseline.append(dict(seed=seed, task='old_hand_aligned_grasp_lift_hold',
                                 result=asdict(old.run(seed=seed, controller='force'))))
            print(f"seed {seed}: success={result['success']} failure={result['failure']} duration={result['duration']:.2f}", flush=True)
    summary = dict(physics_backend='classic_mujoco_cpu', rendering=gpu,
                   workcell_version=WORKCELL_VERSION,
                   mujoco=mujoco.__version__, python=sys.version,
                   wall_seconds=time.perf_counter() - started,
                   trials=trials, baseline_comparison=baseline,
                   comparison_note='Historical regression in ORIGINAL unchanged baseline scene. Different initial scenes and success definitions; not paired performance improvement.',
                   source_sha256=source_hashes,
                   successes=sum(r['success'] for r in trials),
                   failures=sum(not r['success'] for r in trials),
                   safe_abort_attempts=sum(r['safe_abort_attempts'] for r in trials),
                   successful_recoveries=sum(r['successful_recoveries'] for r in trials))
    with args.summary.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
