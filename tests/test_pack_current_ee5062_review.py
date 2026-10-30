"""Current-review selection and report-asset safety, using synthetic fixtures."""

import json
from pathlib import Path
import sys
import zipfile

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import package_project
import pack_ee5062_review as review
import pack_current_ee5062_review as current


def put(root, name, content="fixture\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    for name in package_project.REQUIRED_FILES + package_project.REQUIRED_ASSETS:
        put(root, name)
    for name in package_project.SOURCE_TREES + package_project.ASSET_TREES:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in current.REPORT_ASSETS + current.R15_PROVENANCE_ASSETS + current.ORIGINAL_PATH_ASSETS:
        put(root, name)
    put(root, "docs/EE5062_REPORT_DRAFT.md", "![figure](../results/v7_carried_obstacle_current_20261022_seed0/video/tool_screwdriver_peak_tactile.png)\n[run](../V7_RUNBOOK.md)\n[diagnosis](R15_NOMINAL_GEOMETRY_DIAGNOSIS_20261022.md)\n")
    put(root, current.REPORT_PDF, "%PDF-1.4\nsynthetic fixture, not a rendered report\n")
    put(root, current.INTRO_VIDEO, "synthetic fixture, not video evidence\n")
    monkeypatch.setattr(review, "PROJECT", root)
    monkeypatch.setattr(review, "REPORT", root / "docs/EE5062_REPORT_DRAFT.md")
    monkeypatch.setattr(current, "PROJECT", root)
    monkeypatch.setattr(review, "historical_catalogue", lambda: pytest.fail("current pack must not discover historical evidence"))
    return root


def failure_evidence(root):
    path = root / "results/v7_carried_obstacle_current_20261021_seed0"
    for name in ("result.json", "scene.xml", "physical_trace.jsonl.gz", "native_contacts.jsonl.gz", "action_log.json"):
        put(path, name, '{"success": false}\n' if name.endswith(".json") else "fixture\n")
    return [("current_carried_dependency_failure", path)]


def test_current_defaults_are_explicit_and_include_negative_results():
    labels = dict(current.CURRENT_PATHS)
    assert "current_carried_dependency_failure" in labels
    assert "current_public_ycb_r15_dynamic_negative" in labels
    assert "current_independent_rgbd_audit" in labels
    assert len(labels) == len(current.CURRENT_PATHS)
    assert all("20261014" not in value for value in labels.values())
    assert "current" in current.INTRO_VIDEO
    assert current.REPORT_PDF == "docs/EE5062_REPORT_DRAFT.pdf"
    assert not any("current random public-YCB baseline is 3/18" in value for value in current.LIMITATIONS)


def test_failed_task_is_complete_evidence_and_marker_is_hashed(checkout):
    evidence = failure_evidence(checkout)
    marker = put(evidence[0][1], ".complete", "retrieval completed\n")
    check = current.check_completeness(evidence)[0]
    assert check["complete"] is True
    assert check["completion_marker"]["sha256"] == review.digest(marker)
    (evidence[0][1] / "scene.xml").unlink()
    check = current.check_completeness(evidence)[0]
    assert check["complete"] is False
    assert "scene.xml" in check["missing_required_files"]


def test_false_completion_flag_and_partial_transfer_are_not_ready(checkout):
    evidence = failure_evidence(checkout)
    put(evidence[0][1], "summary.json", '{"complete": false}\n')
    put(evidence[0][1], "native.json.gz.part")
    check = current.check_completeness(evidence)[0]
    assert not check["complete"]
    assert "summary.complete is not true" in check["issues"]
    assert any("partial transfer" in issue for issue in check["issues"])


def test_secret_or_symlink_evidence_is_rejected(checkout, tmp_path):
    evidence = failure_evidence(checkout)
    secret = put(evidence[0][1], "credentials.json", "test-only-secret\n")
    with pytest.raises(ValueError, match="Credential-like"):
        current.check_completeness(evidence)
    secret.unlink()
    outside = put(tmp_path, "outside.json", "{}\n")
    (evidence[0][1] / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        current.check_completeness(evidence)


@pytest.mark.parametrize("kind", ["outside", "symlink", "ancestor", "directory"])
def test_report_assets_reject_unsafe_paths_before_build(checkout, tmp_path, kind):
    outside = put(tmp_path, "private.json", "test-only-secret\n")
    if kind == "outside":
        asset = outside
    elif kind == "symlink":
        asset = checkout / "linked.png"
        asset.symlink_to(outside)
    elif kind == "ancestor":
        (checkout / "linked_dir").symlink_to(tmp_path)
        asset = checkout / "linked_dir/private.json"
    else:
        asset = checkout / "docs"
    with pytest.raises((ValueError, FileNotFoundError)):
        review.checked_report_assets([asset])


def test_current_inventory_never_calls_legacy_default_evidence(checkout, tmp_path, monkeypatch):
    evidence = failure_evidence(checkout)
    brief = put(tmp_path, "course_brief.pdf", "%PDF-1.4\nfixture\n")
    monkeypatch.setattr(current, "current_evidence", lambda: evidence)
    monkeypatch.setattr(review, "default_evidence", lambda: pytest.fail("legacy evidence used"))
    selected, status = current.inventory_current(brief, checkout / current.REPORT_PDF,
                                                checkout / current.INTRO_VIDEO, compact=True)
    assert selected == evidence
    assert status["scope_limitations"] == current.LIMITATIONS
    assert status["submission_ready"] is False
    assert status["evidence_completeness"][0]["complete"]


def test_report_hrefs_and_compact_inclusion_locations_are_preserved(checkout, tmp_path):
    evidence = failure_evidence(checkout)
    carried = checkout / "results/v7_carried_obstacle_current_20261022_seed0"
    evidence += [("carried_figure", carried)]
    brief = put(tmp_path, "course_brief.pdf", "%PDF-1.4\nfixture\n")
    result = review.make_pack(tmp_path / "fixture_review", evidence, checkout / current.INTRO_VIDEO,
        checkout / current.REPORT_PDF, compact=True, brief=brief, include_historical=False,
        report_assets=[checkout / name for name in current.REPORT_ASSETS],
        original_path_assets=[checkout / name for name in current.ORIGINAL_PATH_ASSETS],
        status_overrides={"scope_limitations": current.LIMITATIONS, "review_notes": "Historical paths refer to recovered records only."})
    content = Path(result["output"]) / "review"
    assert "RAW_PHYSICAL_MARKER_DEMO.mp4" not in (content / "README_REVIEW.md").read_text()
    for name in current.ORIGINAL_PATH_ASSETS:
        assert (content / name).is_file()
        assert review.digest(content / name) == review.digest(checkout / name)
    assert (content / "PHYSICS_AND_CONTACT_CONTROL.md").is_file()
    assert (content / "V7_RUNBOOK.md").is_file()
    for name in ("R15_NOMINAL_GEOMETRY_DIAGNOSIS_20261022.md", "V7_OPERATOR_RECORDING_AUDIT.md",
                 "EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md"):
        assert (content / "report" / name).is_file()
    figure = current.REPORT_ASSETS[-1]
    assert (content / figure).is_file()
    raw = json.loads((content / "RAW_EVIDENCE_INVENTORY.json").read_text())
    entry = next(item for item in raw["files"] if item["source"].endswith("peak_tactile.png"))
    assert entry["included_in_evidence_folder"] is False
    assert entry["included_anywhere_in_review"] is True
    assert entry["included_elsewhere_in_review"] == figure
    manifest = json.loads((content / "PACK_MANIFEST.json").read_text())
    for item in manifest["files"]:
        assert review.digest(content / item["path"]) == item["sha256"]
    with zipfile.ZipFile(result["zip"]) as archive:
        assert archive.testzip() is None
        assert not any("historical_ycb_snapshot" in name for name in archive.namelist())


def test_cli_requires_explicit_authoritative_brief(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pack_current_ee5062_review.py", "--inventory-only"])
    with pytest.raises(SystemExit) as error:
        current.main()
    assert error.value.code == 2


@pytest.mark.parametrize("changed", ["brief", "report_pdf", "intro_video", "summary"])
def test_inventory_to_copy_input_changes_are_rejected(checkout, tmp_path, monkeypatch, changed):
    evidence = failure_evidence(checkout)
    summary = put(evidence[0][1], "summary.json", '{"complete": true}\n')
    brief = put(tmp_path, "course_brief.pdf", "%PDF-1.4\nfixture\n")
    monkeypatch.setattr(current, "current_evidence", lambda: evidence)
    _, status = current.inventory_current(brief, checkout / current.REPORT_PDF,
                                          checkout / current.INTRO_VIDEO, compact=True)
    targets = {"brief": brief, "report_pdf": checkout / current.REPORT_PDF,
               "intro_video": checkout / current.INTRO_VIDEO, "summary": summary}
    if changed == "summary":
        targets[changed].write_text('{"complete": false}\n')
    else:
        with targets[changed].open("a") as handle:
            handle.write("changed after validation\n")
    with pytest.raises(ValueError, match="declared input|bindings changed"):
        review.make_pack(tmp_path / "changed_review", evidence, checkout / current.INTRO_VIDEO,
            checkout / current.REPORT_PDF, compact=True, brief=brief, include_historical=False,
            status_overrides=status, validate_copied_inputs=current.validate_current_copied_inputs)
    assert not (tmp_path / "changed_review/EE5062_LOCAL_REVIEW.zip").exists()
