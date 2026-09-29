import json

from tjev.utils import canonical, file_hash, fingerprint, write_json


def test_fingerprint_ignores_key_order():
    assert canonical({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_write_json_is_atomic_and_hashable(tmp_path):
    path = tmp_path / "sub" / "x.json"
    write_json(path, {"é": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"é": 1}
    assert not list(path.parent.glob("*.tmp*"))
    assert len(file_hash(path)) == 64
