import numpy as np

from kuka_sharpa_grasp.v7_vision import fit_known_radius
from kuka_sharpa_grasp.camera import CameraObservation, save_observation


def test_known_radius_localizes_occluded_circle_with_map_error():
    centre = np.array([.784, -.054])
    angles = np.linspace(.2, 1.5, 60)
    points = centre + .012 * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    estimated, residual = fit_known_radius(points, .012, (.78, -.05))
    assert np.linalg.norm(estimated - centre) < 1e-6
    assert residual < 1e-6


def test_degenerate_side_is_rejected():
    estimated, residual = fit_known_radius(np.tile((.792, -.05), (30, 1)), .012, (.78, -.05))
    assert estimated is None and residual is None


def test_metric_depth_export_preserves_invalid_pixels_and_units(tmp_path):
    from PIL import Image

    depth = np.array([[.12345, np.nan], [1.23456, 0.]], dtype=np.float32)
    observation = CameraObservation(np.zeros((2, 2, 3), dtype=np.uint8), depth,
                                    .5, .5, np.zeros(3), np.eye(3), 100., "test")
    metadata = save_observation(observation, tmp_path)
    png = np.asarray(Image.open(tmp_path / "wrist_depth.png"))
    assert png.tolist() == [[123, 0], [1235, 0]]
    raw = np.load(tmp_path / "wrist_depth_m.npy", allow_pickle=False)
    np.testing.assert_allclose(raw, depth, equal_nan=True)
    assert metadata["finite_fraction"] == .5
    assert metadata["png_unit"] == "millimetres"
