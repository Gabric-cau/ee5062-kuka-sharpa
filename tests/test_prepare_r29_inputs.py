"""Cold-checkout input preparation must reject corruption before installation."""

import hashlib
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_r29_inputs import prepare


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    data = {"results/prefix/commands.json": b"commands", "results/prior.json": b"prior",
            "vendor/object.stl": b"mesh"}
    records = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "archive": name.startswith("results/")} for name, content in data.items()]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "r29_runtime_inputs_v1", "files": records,
        "archive_input_bytes": sum(r["bytes"] for r in records if r["archive"])}))
    mesh = root / "vendor/object.stl"
    mesh.parent.mkdir()
    mesh.write_bytes(data["vendor/object.stl"])
    return root, manifest, data


def archive_at(path, data, *, extras=(), duplicate=False, symlink=False):
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, content) in enumerate(data.items()):
            if not name.startswith("results/"):
                continue
            member = zipfile.ZipInfo(name)
            if symlink and index == 0:
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, content)
        for name in extras:
            archive.writestr(name, b"extra")
        if duplicate:
            archive.writestr("results/prior.json", data["results/prior.json"])
    return path


def test_cold_install_checks_every_input_and_is_idempotent(inputs, tmp_path):
    root, manifest, data = inputs
    archive = archive_at(tmp_path / "runtime.zip", data)
    result = prepare(root, manifest, archive)
    assert result["ok"] and result["verified_files"] == 3 and len(result["installed"]) == 2
    assert prepare(root, manifest)["ok"]
    again = prepare(root, manifest, archive)
    assert again["installed"] == [] and len(again["already_present"]) == 2


def test_last_payload_corruption_leaves_no_earlier_destination(inputs, tmp_path):
    root, manifest, data = inputs
    data["results/prior.json"] = b"wrong"
    archive = archive_at(tmp_path / "bad.zip", data)
    with pytest.raises(ValueError, match="checksum"):
        prepare(root, manifest, archive)
    assert not (root / "results").exists()


@pytest.mark.parametrize("extra", ["../escape", "/absolute", "results\\escape", "other.txt"])
def test_unlisted_or_unsafe_archive_members_rejected(inputs, tmp_path, extra):
    root, manifest, data = inputs
    archive = archive_at(tmp_path / "bad.zip", data, extras=[extra])
    with pytest.raises(ValueError):
        prepare(root, manifest, archive)
    assert not (root / "results").exists()


def test_duplicate_archive_member_rejected(inputs, tmp_path):
    root, manifest, data = inputs
    with pytest.warns(UserWarning, match="Duplicate"):
        archive = archive_at(tmp_path / "duplicate.zip", data, duplicate=True)
    with pytest.raises(ValueError, match="inventory"):
        prepare(root, manifest, archive)
    assert not (root / "results").exists()


def test_archive_symlink_rejected(inputs, tmp_path):
    root, manifest, data = inputs
    archive = archive_at(tmp_path / "link.zip", data, symlink=True)
    with pytest.raises(ValueError, match="regular"):
        prepare(root, manifest, archive)
    assert not (root / "results").exists()


def test_conflicting_existing_input_is_preserved_before_any_install(inputs, tmp_path):
    root, manifest, data = inputs
    prior = root / "results/prior.json"
    prior.parent.mkdir()
    prior.write_bytes(b"local edit")
    archive = archive_at(tmp_path / "runtime.zip", data)
    with pytest.raises(ValueError, match="refusing overwrite"):
        prepare(root, manifest, archive)
    assert prior.read_bytes() == b"local edit"
    assert not (root / "results/prefix").exists()


@pytest.mark.parametrize("link_file", [False, True])
def test_symlink_destination_or_parent_is_rejected(inputs, tmp_path, link_file):
    root, manifest, data = inputs
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        if link_file:
            (root / "results").mkdir()
            (outside / "prior.json").write_bytes(data["results/prior.json"])
            (root / "results/prior.json").symlink_to(outside / "prior.json")
        else:
            (root / "results").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip("Symlink creation unavailable on this platform: " + str(error))
    archive = archive_at(tmp_path / "runtime.zip", data)
    with pytest.raises(ValueError, match="symlink"):
        prepare(root, manifest, archive)
    assert not (outside / "prefix").exists()


def test_check_reports_missing_and_changed_inputs(inputs, tmp_path):
    root, manifest, data = inputs
    with pytest.raises(ValueError, match="Missing or mismatched"):
        prepare(root, manifest)
    archive = archive_at(tmp_path / "runtime.zip", data)
    prepare(root, manifest, archive)
    (root / "results/prior.json").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Missing or mismatched"):
        prepare(root, manifest)


def test_missing_vendored_mesh_blocks_install(inputs, tmp_path):
    root, manifest, data = inputs
    (root / "vendor/object.stl").unlink()
    archive = archive_at(tmp_path / "runtime.zip", data)
    with pytest.raises(ValueError, match="Vendored input missing"):
        prepare(root, manifest, archive)
    assert not (root / "results").exists()
