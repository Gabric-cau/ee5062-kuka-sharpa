"""V6 RGB-D tasks with source-backed masses and bounded physical control."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from kuka_sharpa_grasp.camera import CameraConfig
from kuka_sharpa_grasp.catalogue import GRASP_SPECIFICATIONS
from kuka_sharpa_grasp.pick_place import Scenario, run_task
from kuka_sharpa_grasp.pick_place_scene import build_pick_place_scene


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--kind', choices=(*GRASP_SPECIFICATIONS, 'all'), default='ycb_can')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--trials-per-object', type=int, default=1)
    parser.add_argument('--timestep', type=float, choices=(.002, .001), default=.002)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--depth-noise', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--no-support-gate', action='store_true')
    parser.add_argument('--trace-stride', type=int, default=10)
    args = parser.parse_args()
    if min(args.trials_per_object, args.max_attempts, args.trace_stride) < 1:
        parser.error('counts must be positive')
    try:
        CameraConfig(depth_noise_std=args.depth_noise, dropout=args.dropout)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or PROJECT / 'results' / f'v6_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output.mkdir(parents=True, exist_ok=False)
    kinds = list(GRASP_SPECIFICATIONS) if args.kind == 'all' else [args.kind]
    files = [*sorted((PROJECT / 'src').rglob('*.py')), Path(__file__)]
    hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if not path.name.startswith('._')}
    schedule = [{'kind': kind, 'seed': args.seed + i} for i in range(args.trials_per_object) for kind in kinds]
    write(output / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        physics_profile='physical-v6', physics_backend='classic_mujoco_cpu',
        role='engineering development; not a held-out success-rate estimate',
        schedule=schedule, maximum_attempts=args.max_attempts, timestep_s=args.timestep,
        support_required=not args.no_support_gate,
        assumptions=['known upright catalogue identity', 'zero initial yaw',
                     'authored red appearance prior', 'uniform-density mesh inertia',
                     'simulated instrumented table load; no hardware calibration'],
        source_sha256=hashes, mujoco=mujoco.__version__, python=sys.version,
        platform=platform.platform()))
    rows = []
    for index, case in enumerate(schedule):
        folder = output / f"trial_{index:03d}_{case['kind']}_seed_{case['seed']}"
        folder.mkdir()
        row = {**case, 'success': False, 'runner_error': None}
        started = time.monotonic()
        try:
            scene = build_pick_place_scene(folder / 'scene.xml', case['kind'],
                                          physics_profile='physical-v6', timestep=args.timestep)
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
            model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
            frames = []
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [.58, -.04, .52]
                camera.distance, camera.azimuth, camera.elevation = 1.6, 135, -25

                def frame(stage, live):
                    renderer.update_scene(live, camera=camera)
                    pixels = renderer.render().copy()
                    name = f'{len(frames):02d}_{stage}.png'
                    Image.fromarray(pixels).save(folder / name)
                    frames.append(dict(stage=stage, timestamp=float(live.time), file=name,
                                       pixel_std=float(pixels.std())))

                result, trace = run_task(model, data, Scenario(case['seed'], case['kind']),
                    frame_callback=frame, max_attempts=args.max_attempts,
                    support_load_sensor=not args.no_support_gate,
                    support_required=not args.no_support_gate,
                    grasp_spec=GRASP_SPECIFICATIONS[case['kind']],
                    camera_config=CameraConfig(depth_noise_std=args.depth_noise,
                                               dropout=args.dropout, seed=case['seed']))
            result['frames'] = frames
            result['wall_seconds'] = time.monotonic() - started
            write(folder / 'result.json', result)
            write(folder / 'trace.json', trace[::args.trace_stride] + [trace[-1]])
            row.update(success=result['success'], failure=result['failure'],
                       lifted=any(record['oracle']['lifted'] and record['oracle']['hand_contact'] for record in trace),
                       placement_error_m=result['placement_error'],
                       maximum_hand_total_force_n=result['maximum_hand_total_force'],
                       confirmed_handoffs=len(result['confirmed_handoffs']),
                       rendering=result['rendering'], wall_seconds=result['wall_seconds'])
        except Exception as exc:
            row['runner_error'] = f'{type(exc).__name__}: {exc}'
        if any(hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() != value for path, value in hashes.items()):
            raise RuntimeError('source changed during batch; partial rows are not a frozen experiment')
        rows.append(row)
        write(output / f'row_{index:03d}.json', row)
        print(json.dumps(row), flush=True)
    write(output / 'summary.json', dict(complete=len(rows) == len(schedule),
        scheduled=len(schedule), successes=sum(row['success'] for row in rows),
        runner_errors=sum(row['runner_error'] is not None for row in rows), rows=rows,
        groups={kind: {'attempted': sum(row['kind'] == kind for row in rows),
                      'successes': sum(row['kind'] == kind and row['success'] for row in rows)} for kind in kinds}))
    return int(any(row['runner_error'] for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
