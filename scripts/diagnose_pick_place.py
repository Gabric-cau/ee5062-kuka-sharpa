"""Read-only analysis of failed trials; oracle labels are used offline only."""
import json
from pathlib import Path
import sys
import argparse
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from kuka_sharpa_grasp.pick_place import Scenario, reset_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=PROJECT / 'results/v2_pick_place/workcell_rate_limited')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/v2_pick_place/workcell_rate_limited_diagnosis.json')
    args = parser.parse_args()
    folder = args.folder
    model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
    data = mujoco.MjData(model)
    _, kin, _ = reset_task(model, data, Scenario(0))
    initial_contacts = []
    for contact in data.contact[:data.ncon]:
        initial_contacts.append(dict(distance=float(contact.dist), bodies=[
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
            for g in (contact.geom1, contact.geom2)]))
    phases = []
    for trial in sorted(folder.glob('seed_*')):
        seed = int(trial.name.split('_')[1])
        trace = json.loads((trial / 'trace.json').read_text())
        for state in dict.fromkeys(row['state'] for row in trace):
            rows = [row for row in trace if row['state'] == state]
            phases.append(dict(seed=seed, state=state,
                tracking_error_max_m=max(row['arm_tracking_error'] for row in rows),
                object_start=rows[0]['oracle']['position'], object_end=rows[-1]['oracle']['position'],
                final_contacts=rows[-1]['contacts'], peak_fingertip_force=max(row['force'] for row in rows),
                maximum_robot_table_force_n=max((c['normal_force_n'] for row in rows for c in row['robot_table_contacts']), default=0.),
                minimum_robot_table_contact_distance_m=min((c['distance_m'] for row in rows for c in row['robot_table_contacts']), default=0.),
                maximum_thumb_force_n=max(row['finger_forces']['thumb'] for row in rows),
                hand_command_min=float(np.min([row['hand_command'] for row in rows])),
                hand_command_max=float(np.max([row['hand_command'] for row in rows]))))
    result = dict(initial_contacts=initial_contacts, stages=phases,
                  analysis_only=True, hand_reference_origin=kin.origin.tolist(),
                  hand_reference_rotation=kin.rotation.tolist())
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(initial_contacts, indent=2))


if __name__ == '__main__':
    main()
