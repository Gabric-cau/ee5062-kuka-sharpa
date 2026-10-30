"""Evaluation-only ground truth; never imported by perception or camera."""
import mujoco


def oracle_body_position(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if body_id < 0:
        raise ValueError("target missing")
    return data.xpos[body_id].copy(), float(data.time)
