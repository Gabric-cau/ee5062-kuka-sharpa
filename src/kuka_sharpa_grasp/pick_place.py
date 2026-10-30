"""Isolated physical task; reset is the only write to live generalized positions."""
from dataclasses import dataclass, asdict, replace
import time
import mujoco
import numpy as np
from .rollout import GraspRollout
from .hand_grasp import HandConfig, HandGraspController, HandContactReader, CALIBRATED_CAN_CLOSURE
from .motion import ArmCommandLimiter
from .physics import model_physics_version, arm_joint_velocity_limits, arm_joint_acceleration_limits
from .support import SupportConfig, TableSupportGate
from .camera import CameraObserver
from .perception import estimate_object
from .pick_place_controller import TaskController, State, candidates, interpolate, completed_success
from .pick_place_evaluator import PlacementEvaluator, held_lift_evidence
from .catalogue import GRASP_SPECIFICATIONS


@dataclass(frozen=True)
class Scenario:
    seed: int
    object_kind: str = 'ycb_can'
    zone: tuple = (.65, -.18, .38)

    @property
    def position(self):
        rng = np.random.default_rng(self.seed)
        return (.72 + rng.uniform(-.015, .015), rng.uniform(-.025, .025), .38)


class RobotKinematics:
    def __init__(self, model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
        self.home = np.array([0., .785398, 0., -1.5708, 0., 0., 0.])
        self.scratch.qpos[:7] = self.home
        mujoco.mj_forward(model, self.scratch)
        self.origin = self.scratch.xpos[self.body].copy()
        self.rotation = self.scratch.xmat[self.body].reshape(3, 3).copy()
        self.approach_axis = self.rotation[:, 2].copy()

    def solve(self, offset, seed=None, tolerance=.002):
        q = self.home.copy() if seed is None else np.asarray(seed).copy()
        jp = np.zeros((3, self.model.nv))
        jr = jp.copy()
        for _ in range(150):
            self.scratch.qpos[:7] = q
            mujoco.mj_forward(self.model, self.scratch)
            rot = self.scratch.xmat[self.body].reshape(3, 3)
            error = self.origin + offset - self.scratch.xpos[self.body]
            orient = sum(np.cross(rot[:, i], self.rotation[:, i]) for i in range(3)) / 2
            if np.linalg.norm(error) < tolerance and np.linalg.norm(orient) < min(.015, tolerance * 7.5):
                return q
            mujoco.mj_jacBody(self.model, self.scratch, jp, jr, self.body)
            jac = np.vstack([jp[:, :7], jr[:, :7]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .001 * np.eye(6), np.r_[error, orient])
            q += np.clip(delta, -.06, .06)
            q = np.clip(q, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        return None

    def position(self, joints):
        self.scratch.qpos[:7] = joints
        mujoco.mj_forward(self.model, self.scratch)
        return self.scratch.xpos[self.body].copy()

    def joint_positions(self, data):
        """Robot proprioception excludes all hand and target generalized positions."""
        return data.qpos[:7].copy()

    def minimum_table_clearance(self, joints):
        self.position(joints)
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        distances = [mujoco.mj_geomDistance(self.model, self.scratch, table, gid, 1., None)
                     for gid in range(self.model.ngeom)
                     if self.model.geom_bodyid[gid] != 0
                     and self.model.geom_contype[gid] != 0
                     and (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(self.model.geom_bodyid[gid])) or '') != 'target']
        return min(distances)

    def copy_open_hand(self, data):
        for joint in range(7, self.model.njnt):
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                adr = self.model.jnt_qposadr[joint]
                self.scratch.qpos[adr] = data.qpos[adr]


class MotionFailure(RuntimeError):
    pass


class RobotGravityCompensation:
    """Robot-only feedforward, sharing the V6 motor's total torque saturation.

    Scratch data never copies target joints or live qfrc_bias. The object and
    contact dynamics therefore cannot supply privileged forces to this control.
    """
    def __init__(self, model, limit=80.):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.limit = limit
        self.robot_qpos = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                           if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.limited = model_physics_version(model) >= 6
        self.last_torque = np.zeros(7)
        self.last_base = None
        self.last_compensated = None

    def apply(self, data, arm_command=None):
        self.scratch.qpos[self.robot_qpos] = data.qpos[self.robot_qpos]
        self.scratch.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.scratch)
        self.last_torque = np.clip(self.scratch.qfrc_bias[:7], -self.limit, self.limit)
        if self.limited:
            base = data.ctrl[:7].copy() if arm_command is None else np.asarray(arm_command).copy()
            if (arm_command is None and self.last_compensated is not None
                    and np.array_equal(base, self.last_compensated)):
                base = self.last_base.copy()
            # PD's position offset adds feedforward inside the bounded motor.
            gains = self.model.actuator_gainprm[:7, 0]
            if np.any(gains <= 0):
                raise ValueError('positive arm actuator gains are required for gravity compensation')
            data.ctrl[:7] = base + self.last_torque / gains
            data.qfrc_applied[:7] = 0.
            self.last_base, self.last_compensated = base, data.ctrl[:7].copy()
        else:
            data.qfrc_applied[:7] = self.last_torque
        return self.last_torque.copy()


def reset_task(model, data, scenario):
    baseline = GraspRollout(model, data)
    kin = RobotKinematics(model)
    start_height = .34 if model_physics_version(model) >= 6 and scenario.object_kind == 'ycb_cracker_box' else .18
    start = kin.solve(np.array([0., 0., start_height]))
    if start is None:
        raise RuntimeError('start IK failed')
    baseline.reset(seed=scenario.seed, object_pos=scenario.position)
    data.qpos[:7] = start
    baseline._write_arm_ctrl(tuple(start))
    HandGraspController(model).initialize(data)
    mujoco.mj_forward(model, data)
    return baseline, kin, start


def run_task(model, data, scenario, frame_callback=None, force_failed_lift=False,
             step_callback=None, max_attempts=2, first_grasp_bias=None,
             cartesian_motion=True, visual_transfer=True, support_load_sensor=False,
             force_mode='contact', placement_retention=False, camera_config=None,
             support_required=False, grasp_spec=None, hand_config_override=None):
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    if force_mode not in ('contact', 'fixed', 'fixed_calibrated'):
        raise ValueError('force_mode must be contact, fixed, or fixed_calibrated')
    if force_mode == 'fixed_calibrated' and scenario.object_kind != 'ycb_can':
        raise ValueError('the calibrated fixed profile is specific to ycb_can')
    if camera_config is not None and camera_config.delay:
        raise ValueError('run_task requires synchronous camera delivery; nonzero delay is unsupported')
    if support_required and not support_load_sensor:
        raise ValueError('support_required needs the declared instrumented-table load sensor')
    robot, kin, command = reset_task(model, data, scenario)
    controller = TaskController()
    evaluator = PlacementEvaluator(model, scenario.zone)
    trace = []
    peak = 0.
    all_contact_peak = 0.
    peak_pair = None
    released = False
    attempts = []
    recovery_count = 0
    previous_slip_adjustments = 0
    physical_v6 = model_physics_version(model) >= 6
    if physical_v6 and grasp_spec is None:
        grasp_spec = GRASP_SPECIFICATIONS[scenario.object_kind]
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    # Catalogue mass is a declared prior, not a live object-pose observation.
    grip_target = min(8., 2. * float(model.body_mass[target_body]) / .15)
    hand_config = (HandConfig(contact_threshold_n=grip_target, maximum_force_n=12.,
                              retention_rate=.06, retention_extra=.10,
                              retention_minimum_force_n=.8 * grip_target)
                   if physical_v6 else HandConfig())
    if physical_v6 and scenario.object_kind == 'ycb_can':
        hand_config = HandConfig(contact_threshold_n=3.5, maximum_force_n=12.,
                                 thumb_opposition_rad=1.1, thumb_mcp_flexion_ratio=.4,
                                 thumb_ip_flexion_ratio=1.2, retention_rate=.10,
                                 retention_extra=.25, retention_minimum_force_n=2.8)
    if hand_config_override is not None:
        hand_config = hand_config_override
    hand = HandGraspController(model, hand_config)
    fixed_targets = (dict(CALIBRATED_CAN_CLOSURE) if force_mode == 'fixed_calibrated'
                     else {finger: .85 for finger in hand.closure})
    hand_reader = HandContactReader(model)
    vision_updates = []
    initial_gap = evaluator.clearance(model, data)
    gravity = RobotGravityCompensation(model)
    velocity_limits = arm_joint_velocity_limits(model)
    acceleration_limits = arm_joint_acceleration_limits(model)
    limiter = (ArmCommandLimiter(command, velocity_limits, acceleration_limits)
               if physical_v6 else None)
    support_config = SupportConfig(minimum_load_n=.7 * float(model.body_mass[target_body]) * abs(float(model.opt.gravity[2])))
    support_gate = TableSupportGate(support_config) if support_required else None
    handoffs = []
    overload_seconds = 0.
    rendering = {}

    def observe():
        camera = (CameraObserver(model) if camera_config is None
                  else CameraObserver(model, camera_config))
        with camera:
            observation = camera.poll(data)
            current_belief = estimate_object(observation, scenario.object_kind)
            if hasattr(camera, 'renderer'):
                from OpenGL import GL
                for name, token in (('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER),
                                    ('version', GL.GL_VERSION)):
                    value = GL.glGetString(token)
                    rendering[name] = None if value is None else value.decode('ascii', errors='replace')
        vision_updates.append(dict(state=controller.state, timestamp=float(data.time),
                                   camera='wrist_rgbd', confidence=current_belief.confidence,
                                   centroid=None if current_belief.centroid is None else current_belief.centroid.tolist()))
        return current_belief

    def record():
        nonlocal peak, all_contact_peak, peak_pair
        total = robot.reader.total(robot.reader.read(data))
        peak = max(peak, total.normal)
        wrench = np.zeros(6)
        for contact_id in range(data.ncon):
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            if abs(float(wrench[0])) > all_contact_peak:
                all_contact_peak = abs(float(wrench[0]))
                contact = data.contact[contact_id]
                peak_pair = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                             for g in (contact.geom1, contact.geom2)]
        labels = evaluator.sample(model, data, data.xpos[kin.body], released)
        readings = robot.reader.read(data)
        segment_readings = hand_reader.read(data)
        segment_total = hand_reader.total(segment_readings)
        trace.append(dict(timestamp=float(data.time), state=controller.state,
                          robot_table_contacts=robot_table_contacts(model, data),
                          finger_forces={finger: value.normal for finger, value in readings.items()},
                          contacts=total.contacts, force=total.normal,
                          hand_segment_forces={finger: value.normal for finger, value in segment_readings.items()},
                          hand_segment_contacts=segment_total.contacts,
                          hand_segment_total_force=segment_total.normal,
                          arm_command=(command if physical_v6 else data.ctrl[:7]).tolist(),
                          arm_motor_control=data.ctrl[:7].tolist(),
                          arm_joint_position=kin.joint_positions(data).tolist(),
                          gravity_torque_nm=(gravity.last_torque if physical_v6 else data.qfrc_applied[:7]).tolist(),
                          hand_command=data.ctrl[7:].tolist(), wrist_position=data.xpos[kin.body].tolist(),
                          wrist_rotation_world=data.xmat[kin.body].reshape(3, 3).tolist(),
                          arm_tracking_error=float(np.linalg.norm(data.xpos[kin.body] - kin.position(command))),
                          oracle=labels))
        if physical_v6:
            trace[-1].update(arm_actuator_torque_nm=data.qfrc_actuator[:7].tolist(),
                             arm_joint_velocity_rad_s=data.qvel[:7].tolist(),
                             arm_command_velocity_rad_s=limiter.velocity.tolist())
        if support_gate is not None:
            # record() can be called twice at endpoints; a sensor sample is new
            # only when physics time has advanced.
            if support_gate.last_time is None or data.time > support_gate.last_time:
                support_gate.update(float(data.time), hand.table_support(data))
            trace[-1]['table_load_sensor'] = dict(
                raw_load_n=hand.table_support(data), filtered_load_n=support_gate.filtered_load_n,
                confirmed=support_gate.fresh_confirmation(float(data.time)))
        if step_callback:
            step_callback(controller.state, data)
        return segment_total

    def move(state, target, duration, closure):
        nonlocal command, overload_seconds
        if target is None:
            raise MotionFailure('MOTION_TARGET_IK')
        previous_state = controller.state
        controller.enter(state, data.time)
        if frame_callback:
            frame_callback(state.value + '_start', data)
        start = command.copy()
        steps = int(duration / model.opt.timestep)
        loss_duration = 0.
        start_hand = hand.closure.copy()
        start_opposition = hand.opposition
        start_position = kin.position(start)
        end_position = kin.position(target)
        segment_start = command.copy()
        segment_end = command.copy()
        if state == State.PLACE and previous_state != State.PLACE.value:
            hand.hold_reference = hand.closure.copy()
        segment_steps = max(1, round(.02 / model.opt.timestep))
        for step in range(steps):
            if cartesian_motion and step % segment_steps == 0:
                segment_start = command.copy()
                end_step = min(steps, step + segment_steps)
                next_position = interpolate(start_position, end_position, end_step / steps)
                segment_end = kin.solve(next_position - kin.origin, seed=command, tolerance=.0001)
                if segment_end is None:
                    raise MotionFailure('CARTESIAN_PATH_IK')
            if cartesian_motion:
                fraction = (step % segment_steps + 1) / min(segment_steps, steps - step // segment_steps * segment_steps)
                desired_command = segment_start + fraction * (segment_end - segment_start)
            else:
                desired_command = interpolate(start, target, (step + 1) / steps)
            command = limiter.update(desired_command, model.opt.timestep) if limiter is not None else desired_command
            if physical_v6 and (np.any(command < model.jnt_range[:7, 0])
                                or np.any(command > model.jnt_range[:7, 1])):
                raise MotionFailure('ARM_COMMAND_JOINT_LIMIT')
            robot._write_arm_ctrl(tuple(command))
            if state == State.CLOSE:
                if force_mode == 'contact':
                    hand.close(hand_reader.read(data), model.opt.timestep)
                else:
                    hand.approach_fixed_targets(fixed_targets, model.opt.timestep)
            elif force_mode == 'contact' and state in (State.TRIAL_LIFT, State.TRANSFER):
                hand.retain(hand_reader.read(data), model.opt.timestep)
            elif force_mode == 'contact' and state == State.PLACE and placement_retention and (not support_load_sensor or hand.table_support(data) < .7):
                hand.retain(hand_reader.read(data), model.opt.timestep)
            elif state not in (State.TRIAL_LIFT, State.TRANSFER, State.PLACE):
                alpha = (step + 1) / steps
                blend = 3 * alpha**2 - 2 * alpha**3
                hand.closure = {finger: start_hand[finger] + blend * (closure - start_hand[finger])
                                for finger in start_hand}
                if state in (State.RELEASE, State.VERIFY, State.RECOVER):
                    hand.opposition = start_opposition * (1. - blend)
            hand.write(data)
            gravity.apply(data, command if physical_v6 else None)
            mujoco.mj_step(model, data)
            total = record()
            if physical_v6:
                overload_seconds = overload_seconds + model.opt.timestep if total.normal > 120. else 0.
                if overload_seconds >= .02:
                    raise MotionFailure('SUSTAINED_HAND_OVERLOAD')
                if np.any(np.abs(data.qvel[:7]) > velocity_limits * 1.05):
                    raise MotionFailure('ARM_VELOCITY_LIMIT')
            table_contacts = robot_table_contacts(model, data)
            if state != State.RECOVER and any(item['distance_m'] < -.002 for item in table_contacts):
                raise MotionFailure('ROBOT_TABLE_PENETRATION')
            wrist = data.xpos[kin.body]
            if state != State.RECOVER and not (.3 < wrist[0] < 1.05 and abs(wrist[1]) < .45 and .4 < wrist[2] < 1.1):
                raise MotionFailure('WORKSPACE_VIOLATION')
            if state in (State.TRIAL_LIFT, State.TRANSFER):
                loss_duration = loss_duration + model.opt.timestep if total.contacts == 0 else 0.
                if loss_duration >= .10:
                    raise MotionFailure('SUSTAINED_CONTACT_LOSS')
            elif state == State.PLACE:
                # Near the endpoint, losing hand contact can be a table handoff.
                # Earlier loss is a drop, detected without target pose/table load.
                early_loss = total.contacts == 0 and wrist[2] > end_position[2] + .02
                loss_duration = loss_duration + model.opt.timestep if early_loss else 0.
                if loss_duration >= .10:
                    raise MotionFailure('EARLY_PLACE_CONTACT_LOSS')
        total = record()
        if frame_callback:
            frame_callback(state.value, data)
        if state != State.RECOVER and np.linalg.norm(data.xpos[kin.body] - kin.position(target)) > .04:
            raise MotionFailure('APPROACH_TRACKING_TIMEOUT')
        return total

    def plan(options):
        kin.copy_open_hand(data)
        deadline = time.monotonic() + 3.
        for point in options:
            if time.monotonic() > deadline:
                reasons.append('PLANNER_TIMEOUT')
                break
            offset = point - [.72, 0., .38]
            aperture_offset = np.array([0., .025, .05] if grasp_spec is None else grasp_spec.aperture_offset)
            grasp_offset = offset + aperture_offset
            waypoints = [grasp_offset - kin.approach_axis * .13, grasp_offset,
                         grasp_offset + [0, 0, .15],
                         np.array(scenario.zone) - [.72, 0., .38] + aperture_offset + [0, 0, .15],
                         np.array(scenario.zone) - [.72, 0., .38] + aperture_offset + [0., 0., .015],
                         np.array(scenario.zone) - [.72, 0., .38] + [0, 0, .22]]
            solved = [kin.solve(value) for value in waypoints]
            if all(value is not None for value in solved):
                minimum = min(kin.minimum_table_clearance(interpolate(solved[0], solved[1], alpha))
                              for alpha in np.linspace(0., 1., 21))
                if minimum > .002:
                    return point, solved
                reasons.append('INSERTION_TABLE_CLEARANCE')
            else:
                reasons.append('IK_TIMEOUT')
        return None, None

    def handoff(place):
        if support_gate is None:
            return place
        endpoint = kin.position(place)
        extra = 0.
        while not support_gate.fresh_confirmation(float(data.time)):
            if hand_reader.total(hand_reader.read(data)).contacts == 0:
                raise MotionFailure('UNSUPPORTED_HAND_CONTACT_LOSS')
            if extra + 1e-12 >= support_config.maximum_extra_descent_m:
                raise MotionFailure('TABLE_SUPPORT_NOT_CONFIRMED')
            extra = min(support_config.maximum_extra_descent_m, extra + support_config.descent_step_m)
            place = kin.solve(endpoint - [0., 0., extra] - kin.origin, seed=command)
            move(State.PLACE, place, support_config.descent_seconds, .85)
        handoffs.append(dict(timestamp=float(data.time), extra_descent_m=extra,
                             raw_load_n=hand.table_support(data),
                             filtered_load_n=support_gate.filtered_load_n,
                             reason='CONTINUOUS_TABLE_SUPPORT'))
        return place

    def execute(targets):
        nonlocal released
        pre, grasp, lift, transfer, place, retreat = targets
        try:
            move(State.APPROACH, pre, 1.5, 0.)
            move(State.ALIGN, grasp, 1.5, 0.)
            total = move(State.CLOSE, grasp, 3., .85)
            if total.contacts == 0:
                controller.fail(data.time, 'MISSING_CONTACT')
            else:
                wrist_before = data.xpos[kin.body, 2]
                total = move(State.TRIAL_LIFT, lift, 1.5, .85)
                total = move(State.TRIAL_LIFT, lift, 1.2, .85)
                if controller.accept_lift(data.time, total.contacts, data.xpos[kin.body, 2] - wrist_before > .07 and not force_failed_lift):
                    held_belief = observe()
                    held_options, held_reasons = (candidates(held_belief, data.time) if grasp_spec is None
                                                 else grasp_spec.candidates(held_belief, data.time))
                    if held_options and visual_transfer:
                        # Current RGB-D, not the initial object pose, determines
                        # the Cartesian transfer translation after grasp motion.
                        delta = np.r_[np.asarray(scenario.zone[:2]) - held_options[0][:2], 0.]
                        transfer_offset = kin.position(lift) - kin.origin + delta
                        transfer = kin.solve(transfer_offset)
                        place = kin.solve(transfer_offset - [0., 0., .135])
                        retreat = kin.solve(transfer_offset - kin.approach_axis * .13 + [0., 0., .10])
                        if any(value is None for value in (transfer, place, retreat)):
                            raise MotionFailure('VISUAL_TRANSFER_IK')
                    elif not held_options:
                        raise MotionFailure('HOLD_' + held_reasons[-1])
                    total = move(State.TRANSFER, transfer, 4., .85)
                    if total.contacts == 0:
                        controller.fail(data.time, 'DROP')
                    else:
                        move(State.PLACE, place, 3., .85)
                        move(State.PLACE, place, .5, .85)
                        place = handoff(place)
                        released = True
                        move(State.RELEASE, place, 2., 0.)
                        move(State.VERIFY, retreat, 3. if physical_v6 else 1.8, 0.)
                        move(State.VERIFY, retreat, 2., 0.)
                        placed_belief = observe()
                        placed_options, _ = (candidates(placed_belief, data.time) if grasp_spec is None
                                             else grasp_spec.candidates(placed_belief, data.time))
                        placed_center = None if not placed_options else placed_options[0][:2]
                        # Known-can visual check: verify position and a visible
                        # upright-height cue, without reading the target state.
                        visual_ok = (placed_center is not None and placed_belief.confidence >= .15
                                     and np.linalg.norm(placed_center - scenario.zone[:2]) < .06
                                     and np.percentile(placed_belief.points[:, 2], 95) >
                                     (.465 if grasp_spec is None else grasp_spec.minimum_visible_height))
                        if not visual_ok:
                            raise MotionFailure('PLACEMENT_VISUAL_CHECK')
        except MotionFailure as error:
            if frame_callback:
                frame_callback(controller.state + '_failure', data)
            controller.fail(data.time, str(error))

    controller.enter(State.SEARCH, data.time)
    if frame_callback:
        frame_callback('initial', data)
    reasons, targets, point = [], None, None
    belief = None
    initial_belief = None
    for attempt in range(max_attempts):
        if attempt:
            controller.failure = None
            previous_slip_adjustments += hand.slip_adjustments
            retry_config = hand_config
            if not physical_v6 and force_mode == 'contact':
                retry_config = replace(hand_config, retention_minimum_force_n=2.,
                                       retention_rate=max(.06, hand_config.retention_rate),
                                       retention_extra=max(.06, hand_config.retention_extra),
                                       maximum_force_n=12.)
            hand = HandGraspController(model, retry_config)
            released = False
            evaluator.stable_since = None
        try:
            move(State.SEARCH, command, .4 if attempt == 0 else 1.5, 0.)
            belief = observe()
            if initial_belief is None:
                initial_belief = belief
            options, rejected = (candidates(belief, data.time) if grasp_spec is None
                                 else grasp_spec.candidates(belief, data.time))
            reasons.extend(rejected)
            if attempt == 0 and first_grasp_bias is not None:
                options = [candidate + np.asarray(first_grasp_bias) for candidate in options]
            point, targets = plan(options)
            if targets is None:
                controller.fail(data.time, reasons[-1] if reasons else 'NO_CANDIDATE')
            else:
                execute(targets)
        except MotionFailure as error:
            controller.fail(data.time, str(error))
        if belief is None:
            belief = observe()
            initial_belief = belief
        attempts.append(dict(attempt=attempt + 1, failure=controller.failure,
                             belief_timestamp=belief.timestamp,
                             slip_adjustment_steps=hand.slip_adjustments,
                             hand_parameters=asdict(hand.config),
                             selected_candidate=None if point is None else point.tolist()))
        if controller.failure is None:
            break
        recovery_count += 1
        try:
            if hand_reader.total(hand_reader.read(data)).contacts and targets is not None:
                # Return a retained object to the last observed table grasp height
                # before opening; this is a physical move, never a scene reset.
                move(State.PLACE, targets[1], 3., .85)
                handoff(command)
                move(State.RELEASE, command, 2., 0.)
            retreat_position = kin.position(command) - kin.approach_axis * .13 + [0., 0., .10]
            safe = kin.solve(retreat_position - kin.origin, seed=command)
            if safe is None:
                raise MotionFailure('RECOVERY_IK')
            move(State.RECOVER, safe, 1.8, 0.)
            safe_height = .34 if physical_v6 and scenario.object_kind == 'ycb_cracker_box' else .18
            move(State.RECOVER, kin.solve(np.array([0., 0., safe_height])), 1.8, 0.)
        except MotionFailure as error:
            controller.fail(data.time, 'RECOVERY_' + str(error))
            controller.enter(State.ABORT, data.time, controller.failure)
            break
        if attempt == max_attempts - 1:
            controller.enter(State.ABORT, data.time, controller.failure)
    record()
    final = trace[-1]['oracle']
    lift_evidence = held_lift_evidence(trace) if physical_v6 else None
    physical_success = final['success'] and (not physical_v6 or lift_evidence['observed'])
    success = completed_success(controller.transitions, physical_success, controller.failure)
    evaluation_failure = ('ACTUAL_HELD_LIFT_NOT_CONFIRMED'
                          if physical_v6 and not lift_evidence['observed'] else 'PLACEMENT_VERIFICATION')
    if frame_callback:
        frame_callback('final_verification', data)
    result = dict(seed=scenario.seed, task=('v5_cartesian_rgbd_pick_place' if cartesian_motion
                                         else 'v4_held_belief_pick_place'),
                  success=success,
                  failure=controller.failure or (None if success else evaluation_failure),
                  safe_abort_attempts=recovery_count,
                  successful_recoveries=int(success and recovery_count > 0),
                  attempts=attempts, maximum_attempts=max_attempts,
                  first_grasp_bias=first_grasp_bias,
                  initial_hand_object_gap_m=initial_gap,
                  duration=float(data.time), placement_error=final['placement_error'],
                  maximum_contact_force=all_contact_peak, maximum_fingertip_total_force=peak,
                  maximum_contact_pair=peak_pair,
                  transitions=controller.transitions,
                  candidate_rejections=reasons, initial_position=list(scenario.position),
                  belief_centroid=None if initial_belief.centroid is None else initial_belief.centroid.tolist(),
                  belief_timestamp=initial_belief.timestamp, belief_confidence=initial_belief.confidence,
                  selected_candidate=None if targets is None else point.tolist(),
                  zone=list(scenario.zone), final_oracle=final)
    result['approach_axis_world'] = kin.approach_axis.tolist()
    result['gravity_compensation_limit_nm'] = gravity.limit
    result['slip_adjustment_steps'] = previous_slip_adjustments + hand.slip_adjustments
    result['vision_updates'] = vision_updates
    result['force_mode'] = force_mode
    result['maximum_hand_total_force'] = max(row['hand_segment_total_force'] for row in trace)
    result['cartesian_motion'] = cartesian_motion
    result['visual_transfer'] = visual_transfer
    result['placement_retention'] = placement_retention
    result['rendering'] = rendering
    result['camera_parameters'] = None if camera_config is None else asdict(camera_config)
    result['support_load_sensor'] = support_load_sensor
    result['placement_contact_guard'] = dict(sustained_loss_seconds=.10,
                                             above_planned_endpoint_m=.02)
    result['sensor_contract'] = ['wrist_rgbd', 'robot_proprioception', 'simulated_finger_segment_contact_wrenches'] + (['simulated_instrumented_table_load'] if support_load_sensor else [])
    result['simulated_safety_monitors'] = ['robot_table_collision_distance', 'robot_workspace_bounds']
    if physical_v6:
        result['sensor_contract'].append('simulated_robot_table_collision_monitor')
    result['physics_profile'] = 'physical-v6' if physical_v6 else 'legacy-v5'
    result['controller_revision'] = ('v6_braking_thumb_synergy_held_lift_r2' if physical_v6
                                     else 'v6_bounded_physics_and_force_holding_recovery')
    result['object_kind'] = scenario.object_kind
    result['support_required'] = support_required
    result['confirmed_handoffs'] = handoffs
    if physical_v6:
        result['task'] = 'v6_bounded_rgbd_contact_pick_place'
        result['actual_held_lift_evidence'] = lift_evidence
        result['verify_retreat_duration_s'] = 3.
        result['grasp_specification'] = asdict(grasp_spec)
        result['arm_limits'] = dict(velocity_rad_s=velocity_limits.tolist(),
                                   commanded_acceleration_rad_s2=acceleration_limits.tolist(),
                                   torque_nm=model.jnt_actfrcrange[:7].tolist())
        result['maximum_arm_torque_nm'] = np.max(np.abs([row['arm_actuator_torque_nm'] for row in trace]), axis=0).tolist()
        result['maximum_arm_velocity_rad_s'] = np.max(np.abs([row['arm_joint_velocity_rad_s'] for row in trace]), axis=0).tolist()
        result['hand_force_relief_trigger_n_per_finger'] = hand.config.maximum_force_n
        result['table_support_parameters'] = asdict(support_config) if support_required else None
    result['hand_control'] = dict(closure_rate_per_second=hand.config.close_rate,
                                contact_trigger_n_per_finger=hand.config.contact_threshold_n,
                                retention_extra_closure_limit=hand.config.retention_extra,
                                retention_rate_per_second=hand.config.retention_rate,
                                retention_minimum_force_n=hand.config.retention_minimum_force_n,
                                retention_enabled=force_mode == 'contact',
                                fixed_closure_targets=None if force_mode == 'contact' else fixed_targets,
                                fixed_profile_development_seeds=(list(range(10))
                                    if force_mode == 'fixed_calibrated' else None))
    result['hand_parameters'] = asdict(hand.config)
    result['initial_hand_parameters'] = asdict(hand_config)
    return result, trace


def robot_table_contacts(model, data):
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    result = []
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact[:data.ncon]):
        pair = [int(contact.geom1), int(contact.geom2)]
        if table not in pair:
            continue
        other = pair[1] if pair[0] == table else pair[0]
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ''
        if body.startswith(('link', 'left_')):
            mujoco.mj_contactForce(model, data, i, wrench)
            result.append(dict(body=body, distance_m=float(contact.dist), normal_force_n=abs(float(wrench[0]))))
    return result
