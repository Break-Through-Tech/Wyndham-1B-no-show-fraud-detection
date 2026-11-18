"""Case-review API tests. Run: python test_cases_api.py (also pytest-compatible)

Uses a temporary SQLite file via WYNDHAM_REVIEW_DB; never touches review_cases.sqlite3.
"""
import os
import tempfile
import threading
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["WYNDHAM_REVIEW_DB"] = str(Path(_tmp) / "test_cases.sqlite3")

from fastapi.testclient import TestClient  # noqa: E402
import api  # noqa: E402

client = TestClient(api.app)
DS = "dataset_abc"


def patch(cid, **body):
    return client.patch(f"/cases/{DS}/{cid}", json=body)


def test_uses_temp_db():
    assert str(api.DB_PATH).startswith(_tmp)


def test_note_then_status_lifecycle_and_persistence():
    cid = "LC1"
    assert patch(cid, note="first").json()["case"]["status"] == "flagged"
    assert patch(cid, status="cleared", note="x").status_code == 409  # must review first
    assert patch(cid, status="reviewing").json()["case"]["status"] == "reviewing"
    assert patch(cid, status="cleared").status_code == 400  # rationale required
    r = patch(cid, status="cleared", note="  looks fine  ").json()["case"]
    assert r["status"] == "cleared" and [n["text"] for n in r["notes"]] == ["first", "looks fine"]
    # notes can still be added after closing, status stays
    r = patch(cid, note="post-close note").json()["case"]
    assert r["status"] == "cleared" and len(r["notes"]) == 3
    saved = client.get(f"/cases/{DS}").json()["cases"][cid]
    assert saved["status"] == "cleared" and len(saved["notes"]) == 3
    # reopen keeps notes
    assert patch(cid, status="reviewing").json()["case"]["notes"] == saved["notes"]


def test_validation():
    assert patch("LC2").status_code == 400
    assert patch("LC2", note="   ").status_code == 400
    assert patch("LC2", status="bogus").status_code == 422
    assert patch("x" * 201, note="a").status_code == 400
    assert patch("LC2", note="a" * 5001).status_code == 422


def test_datasets_are_isolated():
    patch("LC3", note="only here")
    assert "LC3" in client.get(f"/cases/{DS}").json()["cases"]
    assert client.get("/cases/other_dataset").json()["cases"] == {}


def test_slash_in_confirmation_number():
    r = client.patch(f"/cases/{DS}/A%2FB", json={"note": "slash"})
    assert r.status_code == 200, r.text
    assert "A/B" in client.get(f"/cases/{DS}").json()["cases"]


def test_corrupt_notes_do_not_break_listing():
    with api.db_connection() as c:
        c.execute("INSERT INTO case_reviews VALUES(?,?,?,?,?)", ("dsbad", "BAD", "flagged", "{not json", "t"))
    assert client.get("/cases/dsbad").json()["cases"]["BAD"]["notes"] == []


def test_concurrent_notes_are_not_lost():
    cid = "LC4"
    codes = []

    def worker(i):
        codes.append(patch(cid, note=f"n{i}").status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 12
    assert len(client.get(f"/cases/{DS}").json()["cases"][cid]["notes"]) == 12


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as error:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(error))
    raise SystemExit(failed)
