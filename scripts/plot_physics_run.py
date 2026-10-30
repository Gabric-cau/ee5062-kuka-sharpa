"""Plot recorded V6 forces, support and motor limits without rerunning physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.loads((args.trial / 'result.json').read_text())
    trace = json.loads((args.trial / 'trace.json').read_text())
    output = args.output or args.trial / 'physical_diagnostics.png'
    if output.exists():
        parser.error('plot already exists; preserved evidence is not overwritten')
    times = np.array([row['timestamp'] for row in trace])
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm'):
        axes[0, 0].plot(times, [row['hand_segment_forces'][finger] for row in trace], label=finger, linewidth=1.2)
    axes[0, 0].set(title='Hand-object normal contact forces', ylabel='N')
    axes[0, 0].legend(ncol=3, fontsize=8)
    loads = [row.get('table_load_sensor', {}).get('raw_load_n', 0.) for row in trace]
    filtered = [row.get('table_load_sensor', {}).get('filtered_load_n', 0.) for row in trace]
    axes[0, 1].plot(times, loads, label='Vertical table load', color='#177a7d')
    axes[0, 1].plot(times, filtered, label='Filtered load', color='#bb642c')
    support = result.get('table_support_parameters')
    if support:
        axes[0, 1].axhline(support['minimum_load_n'], color='#777777', linestyle='--', label='Release threshold')
    for handoff in result['confirmed_handoffs']:
        axes[0, 1].axvline(handoff['timestamp'], color='#307c40', linestyle=':')
    axes[0, 1].set(title='Continuous support before release', ylabel='N')
    axes[0, 1].legend(fontsize=8)
    torques = np.array([row['arm_actuator_torque_nm'] for row in trace])
    speeds = np.array([row['arm_joint_velocity_rad_s'] for row in trace])
    torque_limits = np.array(result['arm_limits']['torque_nm'])[:, 1]
    speed_limits = np.array(result['arm_limits']['velocity_rad_s'])
    for joint in range(7):
        axes[1, 0].plot(times, torques[:, joint] / torque_limits[joint], label=f'A{joint + 1}', linewidth=1)
        axes[1, 1].plot(times, speeds[:, joint] / speed_limits[joint], label=f'A{joint + 1}', linewidth=1)
    for axis, title in ((axes[1, 0], 'Motor torque / published peak limit'),
                        (axes[1, 1], 'Actual joint speed / configured limit')):
        axis.axhline(1, color='#aa3030', linestyle='--')
        axis.axhline(-1, color='#aa3030', linestyle='--')
        axis.set(title=title, ylabel='Ratio')
        axis.legend(ncol=7, fontsize=7)
    object_positions = np.array([row['oracle']['position'] for row in trace])
    axes[2, 0].plot(times, (object_positions[:, 2] - object_positions[0, 2]) * 1000,
                    label='Measured object rise (evaluation only)', color='#177a7d')
    axes[2, 0].axhline(70, color='#777777', linestyle='--', label='Evaluation lift threshold')
    axes[2, 0].set(title='Object lift and drops', ylabel='mm')
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(times, [row['oracle']['placement_error'] * 1000 for row in trace],
                    label='Object placement error (evaluation only)', color='#bb642c')
    axes[2, 1].plot(times, [row['arm_tracking_error'] * 1000 for row in trace],
                    label='Wrist tracking error', color='#177a7d')
    axes[2, 1].axhline(45, color='#777777', linestyle='--', label='Placement tolerance')
    axes[2, 1].set(title='Placement and arm tracking', ylabel='mm')
    axes[2, 1].legend(fontsize=8)
    stages = [(times[0], trace[0]['state'])]
    for previous, current in zip(trace, trace[1:]):
        if current['state'] != previous['state']:
            stages.append((current['timestamp'], current['state']))
    for axis in axes.flat:
        axis.set_xlabel('Simulation time (s)')
        axis.grid(alpha=.2)
        for index, (started, stage) in enumerate(stages):
            ended = stages[index + 1][0] if index + 1 < len(stages) else times[-1]
            if index % 2 == 0:
                axis.axvspan(started, ended, color='#777777', alpha=.045, zorder=0)
            axis.axvline(started, color='#777777', alpha=.25, linewidth=.6)
    for started, stage in stages:
        axes[2, 0].text(started, .04, stage, rotation=90, fontsize=7,
                        transform=axes[2, 0].get_xaxis_transform(), va='bottom')
    figure.suptitle(f"{result['object_kind']} | seed {result['seed']} | "
                   f"{'success' if result['success'] else result['failure']}\n"
                   f"Sampled traces; full-step hand force peak {result['maximum_hand_total_force']:.2f} N",
                   fontsize=13)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
