from __future__ import annotations

import copy

import pytest
from commonground_scenarios import (
    HumanSnapshotValidationError,
    find_direct_identifier,
    validate_human_snapshot,
)


def valid_human_snapshot() -> dict[str, object]:
    participants = [f"p{index:03d}" for index in range(10)]
    votes = [[1, -1] if index % 2 == 0 else [-1, 1] for index in range(10)]
    return {
        "session_id": "reviewed-session",
        "statements": [
            {"index": 0, "text": "Fund the pilot."},
            {"index": 1, "text": "Publish the aggregate results."},
        ],
        "participants": participants,
        "votes": votes,
        "masked_cells": [],
        "held_out": {},
        "clusters": [
            {
                "id": 0,
                "members": participants[:5],
                "member_indices": list(range(5)),
                "center": [],
            },
            {
                "id": 1,
                "members": participants[5:],
                "member_indices": list(range(5, 10)),
                "center": [],
            },
        ],
        "stats": {
            "comment": [
                {
                    "commentIndex": index,
                    "agrees": 5,
                    "disagrees": 5,
                    "unsure": 0,
                    "total": 10,
                    "responded": 10,
                    "extremity": None,
                    "divisiveness": None,
                }
                for index in range(2)
            ]
        },
        "meta": {
            "synthetic": False,
            "k_anonymity": 5,
            "source": "context-engine-session",
            "seed": 42,
            "consent_scope": "public-benchmark",
            "redistribution_rights_approved": True,
            "schema_version": "commonground-human-snapshot-v2",
            "exporter_version": "1.2.0",
            "source_commit": "a" * 40,
            "privacy_review": {
                "attested": True,
                "reviewed_at": "2026-08-26",
                "checks": [
                    "direct-identifiers",
                    "free-text",
                    "participant-pseudonyms",
                ],
            },
        },
    }


def test_valid_human_snapshot_is_canonical_and_defensively_copied() -> None:
    source = valid_human_snapshot()

    validated = validate_human_snapshot(source)

    assert validated == source
    assert validated is not source
    assert validated["participants"] is not source["participants"]


def test_human_snapshot_contract_requires_explicit_schema_version() -> None:
    snapshot = valid_human_snapshot()
    del snapshot["meta"]["schema_version"]

    with pytest.raises(HumanSnapshotValidationError, match=r"missing=.*schema_version"):
        validate_human_snapshot(snapshot)


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Contact person@example.org", "email"),
        (f"Wallet 0x{'a' * 40}", "EVM address"),
        ("Delegate person.eth", "ENS name"),
        ("Call +14165550199", "phone number"),
        ("Origin 192.0.2.4", "IP address"),
        ("Origin 2001:db8::4", "IP address"),
        ("Approved by Dr Alice Example", "honorific-name"),
    ],
)
def test_direct_identifier_detector_covers_publication_risks(
    text: str, category: str
) -> None:
    assert find_direct_identifier(text) == category


@pytest.mark.parametrize("text", ["Review dated 2026-08-26", "Version 1.2.0"])
def test_direct_identifier_detector_avoids_common_metadata_false_positives(
    text: str,
) -> None:
    assert find_direct_identifier(text) is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("meta", "consent_scope"), "internal-research", "consent_scope"),
        (
            ("meta", "redistribution_rights_approved"),
            False,
            "redistribution_rights_approved",
        ),
        (("meta", "schema_version"), "v2", "schema_version"),
        (("meta", "exporter_version"), "latest", "semantic version"),
        (("meta", "source_commit"), "abc123", "40-character"),
        (("meta", "privacy_review", "attested"), False, "attested"),
        (("meta", "privacy_review", "reviewed_at"), "20260826", "YYYY-MM-DD"),
        (("masked_cells",), [[0, 0]], "no evaluation masks"),
        (("held_out",), {"0,0": 1}, "no evaluation labels"),
    ],
)
def test_human_snapshot_contract_fails_closed_on_missing_release_authority(
    path: tuple[str, ...], value: object, message: str
) -> None:
    snapshot = valid_human_snapshot()
    target: dict[str, object] = snapshot
    for field in path[:-1]:
        target = target[field]  # type: ignore[assignment]
    target[path[-1]] = value

    with pytest.raises(HumanSnapshotValidationError, match=message):
        validate_human_snapshot(snapshot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda snapshot: snapshot.__setitem__(
                "participants", ["p000", "p001", "p002", "p003"]
            ),
            "at least 5",
        ),
        (
            lambda snapshot: snapshot["participants"].__setitem__(0, "Alice"),
            "positional pseudonyms",
        ),
        (
            lambda snapshot: snapshot["statements"][0].__setitem__(
                "text", "Contact Dr Alice Example"
            ),
            "honorific-name",
        ),
        (
            lambda snapshot: snapshot["clusters"][0].__setitem__(
                "member_indices", [0, 1, 2, 3]
            ),
            "at least k=5",
        ),
        (
            lambda snapshot: snapshot["stats"]["comment"][0].__setitem__("agrees", 4),
            "agrees must equal 5",
        ),
        (
            lambda snapshot: snapshot["clusters"][0].__setitem__("center", [0.0, 0.0]),
            "center must be empty",
        ),
        (
            lambda snapshot: snapshot["stats"]["comment"][0].__setitem__(
                "extremity", 0.0
            ),
            "extremity must be null",
        ),
        (
            lambda snapshot: snapshot["stats"]["comment"][0].__setitem__(
                "divisiveness", 0.0
            ),
            "divisiveness must be null",
        ),
    ],
)
def test_human_snapshot_contract_checks_privacy_and_data_semantics(
    mutation: object, message: str
) -> None:
    snapshot = copy.deepcopy(valid_human_snapshot())
    mutation(snapshot)

    with pytest.raises(HumanSnapshotValidationError, match=message):
        validate_human_snapshot(snapshot)


def test_human_snapshot_recomputes_counts_with_missing_votes() -> None:
    snapshot = copy.deepcopy(valid_human_snapshot())
    snapshot["votes"][0][0] = None
    first_comment = snapshot["stats"]["comment"][0]
    first_comment.update({"agrees": 4, "total": 9, "responded": 9})

    validated = validate_human_snapshot(snapshot)

    assert validated["stats"]["comment"][0]["agrees"] == 4
    assert validated["stats"]["comment"][0]["total"] == 9


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "statements",
            [{"index": 0, "text": "Bounded."}] * 1_001,
            "at most 1000",
        ),
        (
            "participants",
            [f"p{index:03d}" for index in range(10_001)],
            "at most 10000",
        ),
        ("clusters", [{}] * 101, "at most 100"),
    ],
)
def test_human_snapshot_rejects_oversized_collections(
    field: str, value: object, message: str
) -> None:
    snapshot = valid_human_snapshot()
    snapshot[field] = value

    with pytest.raises(HumanSnapshotValidationError, match=message):
        validate_human_snapshot(snapshot)


def test_human_snapshot_rejects_vote_matrix_above_cell_budget() -> None:
    snapshot = valid_human_snapshot()
    snapshot["statements"] = [
        {"index": index, "text": f"Bounded statement {index}."}
        for index in range(1_000)
    ]
    snapshot["participants"] = [f"p{index:03d}" for index in range(1_001)]

    with pytest.raises(HumanSnapshotValidationError, match="at most 1000000 cells"):
        validate_human_snapshot(snapshot)
