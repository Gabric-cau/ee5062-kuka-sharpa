"""Current documentation inclusion without widening package discovery."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_project", PROJECT / "scripts/package_project.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)

CURRENT_DOCS = {
    "ACTIVE_DEVELOPMENT.md", "PHYSICS_AND_CONTACT_CONTROL.md", "CONNECTIONS_20261021.md",
    "\u5f53\u524d\u5165\u53e3\u4e0e\u5b9e\u9a8c\u72b6\u6001_20261021.md",
    "docs/R14_CONSTRAINED_JOINT_DESIGN_20261021.md",
    "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md",
    "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021_FINAL2.pdf",
    "docs/V7_OPERATOR_RECORDING_AUDIT.md",
    "docs/EE5062_REPORT_DRAFT.md", "docs/EE5062_REPORT_DRAFT.pdf",
}


def put(project, relative, content="synthetic packaging test fixture\n"):
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def checkout(tmp_path):
    project = tmp_path / "checkout"
    project.mkdir()
    for name in packager.REQUIRED_FILES + packager.REQUIRED_ASSETS:
        put(project, name)
    for name in packager.SOURCE_TREES + packager.ASSET_TREES:
        (project / name).mkdir(parents=True, exist_ok=True)
    put(project, "src/example.py", "VALUE = 1\n")
    return project


def names(project):
    return {path.relative_to(project).as_posix() for path in packager.collect_files(project)}


def test_current_documents_are_explicit_optional_entries(checkout):
    assert CURRENT_DOCS <= set(packager.OPTIONAL_FILES)
    assert "docs" not in packager.SOURCE_TREES
    for name in CURRENT_DOCS:
        put(checkout, name)
    assert CURRENT_DOCS <= names(checkout)


def test_current_documents_can_be_absent_in_an_older_checkout(checkout):
    included = names(checkout)
    assert not CURRENT_DOCS & included
    assert set(packager.REQUIRED_FILES) <= included


def test_unlisted_docs_old_pdfs_and_machine_secrets_stay_out(checkout):
    rejected = {
        "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021_FINAL.pdf",
        "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021_R13.pdf",
        "docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021_R14.pdf",
        "docs/unlisted_private_notes.md", "docs/credentials.json", "docs/connection_keys.pem",
        "docs/.env", "results/private_result.json", "scratch/unlisted_scratch.py",
        "scripts/.env", "src/.ssh/id_ed25519", "src/credentials.json", "config/secrets.json",
        "scripts/session.pem", "scripts/session.key", "scripts/known_hosts",
        "scripts/results/private_result.json", "scripts/scratch/unlisted_scratch.py",
    }
    for name in rejected:
        put(checkout, name, "test-only-secret-marker\n")
    put(checkout, "docs/EE5062_REPORT_DRAFT.md")
    included = names(checkout)
    assert not rejected & included
    assert "docs/EE5062_REPORT_DRAFT.md" in included


def test_allowlisted_document_cannot_be_a_symlink(checkout, tmp_path):
    outside = put(tmp_path, "private_credentials.txt", "test-only-secret\n")
    (checkout / "docs").mkdir()
    (checkout / "docs/EE5062_REPORT_DRAFT.md").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinks are not allowed"):
        packager.collect_files(checkout)


def test_source_symlink_cannot_escape_release_root(checkout, tmp_path):
    outside = put(tmp_path, "private_module.py")
    (checkout / "src/linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="Only regular files"):
        packager.collect_files(checkout)


def test_temporary_fixture_packages_are_reproducible_and_manifested(checkout, tmp_path):
    for name in CURRENT_DOCS:
        put(checkout, name)
    put(checkout, "docs/private_notes.md", "test-only-secret-marker\n")
    first = packager.package(checkout, tmp_path / "first_fixture.zip")
    second = packager.package(checkout, tmp_path / "second_fixture.zip")
    assert first["zip_sha256"] == second["zip_sha256"]
    assert Path(first["output"]).read_bytes() == Path(second["output"]).read_bytes()
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert CURRENT_DOCS <= {item["path"] for item in manifest["files"]}
    with zipfile.ZipFile(first["output"]) as archive:
        assert CURRENT_DOCS <= set(archive.namelist())
        assert "docs/private_notes.md" not in archive.namelist()
        assert not any(b"test-only-secret-marker" in archive.read(name) for name in archive.namelist())
        for record in manifest["files"]:
            assert len(archive.read(record["path"])) == record["size_bytes"]


def test_existing_package_output_is_preserved(checkout, tmp_path):
    output = tmp_path / "existing_fixture.zip"
    output.write_bytes(b"existing-test-fixture")
    with pytest.raises(FileExistsError):
        packager.package(checkout, output)
    assert output.read_bytes() == b"existing-test-fixture"
