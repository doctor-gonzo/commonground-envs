"""Fail-closed validation for publishable human Context Engine snapshots."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

MIN_K_ANONYMITY = 5
HUMAN_SNAPSHOT_SCHEMA_VERSION = "commonground-human-snapshot-v2"
MAX_PARTICIPANTS = 10_000
MAX_STATEMENTS = 1_000
MAX_VOTE_CELLS = 1_000_000
MAX_CLUSTERS = 100
VALID_VOTES = frozenset({-1, 0, 1})
SNAPSHOT_FIELDS = frozenset(
    {
        "session_id",
        "statements",
        "participants",
        "votes",
        "masked_cells",
        "held_out",
        "clusters",
        "stats",
        "meta",
    }
)
META_FIELDS = frozenset(
    {
        "synthetic",
        "k_anonymity",
        "source",
        "seed",
        "consent_scope",
        "redistribution_rights_approved",
        "schema_version",
        "exporter_version",
        "source_commit",
        "privacy_review",
    }
)
PRIVACY_REVIEW_FIELDS = frozenset({"attested", "reviewed_at", "checks"})
REQUIRED_PRIVACY_CHECKS = frozenset(
    {"direct-identifiers", "participant-pseudonyms", "free-text"}
)
CLUSTER_FIELDS = frozenset({"id", "members", "member_indices", "center"})
COMMENT_STAT_FIELDS = frozenset(
    {
        "commentIndex",
        "agrees",
        "disagrees",
        "unsure",
        "total",
        "responded",
        "extremity",
        "divisiveness",
    }
)

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
_EVM_PATTERN = re.compile(r"(?<![0-9a-f])0x[0-9a-f]{40}(?![0-9a-f])", re.I)
_ENS_PATTERN = re.compile(r"(?<![\w@.])[\w-]+(?:\.[\w-]+)*\.eth\b", re.I)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?[1-9]\d{7,14}|(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]\d{3}[-. ]\d{4})(?!\w)"
)
_IPV4_CANDIDATE_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_CANDIDATE_PATTERN = re.compile(
    r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", re.I
)
_HONORIFIC_NAME_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+[A-Z][A-Za-z'\N{RIGHT SINGLE QUOTATION MARK}-]+"
    r"(?:\s+[A-Z][A-Za-z'\N{RIGHT SINGLE QUOTATION MARK}-]+)?\b"
)
_SEMVER_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class HumanSnapshotValidationError(ValueError):
    """Raised when a human snapshot is malformed or unsafe to redistribute."""


def find_direct_identifier(text: str) -> str | None:
    """Return the detected direct-identifier category, if any."""

    normalized = unicodedata.normalize("NFKC", text)
    for label, pattern in (
        ("email", _EMAIL_PATTERN),
        ("EVM address", _EVM_PATTERN),
        ("ENS name", _ENS_PATTERN),
        ("phone number", _PHONE_PATTERN),
        ("honorific-name", _HONORIFIC_NAME_PATTERN),
    ):
        if pattern.search(normalized):
            return label
    for label, pattern in (
        ("IP address", _IPV4_CANDIDATE_PATTERN),
        ("IP address", _IPV6_CANDIDATE_PATTERN),
    ):
        for candidate in pattern.findall(normalized):
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            return label
    return None


def contains_direct_identifier(text: str) -> bool:
    """Return whether text contains a supported direct-identifier pattern."""

    return find_direct_identifier(text) is not None


def validate_human_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate and return a canonical, publication-safe human snapshot.

    This contract is intentionally stricter than the synthetic snapshot schema:
    public human inputs must use positional pseudonyms, carry no evaluation labels,
    satisfy declared k-anonymity, and include explicit rights and privacy attestations.
    """

    root = _exact_object(snapshot, SNAPSHOT_FIELDS, "snapshot")
    session_id = _safe_text(root["session_id"], "session_id")
    statements = _validate_statements(root["statements"])
    participants = _validate_participants(root["participants"])
    votes = _validate_votes(root["votes"], len(participants), len(statements))

    if root["masked_cells"] != []:
        raise HumanSnapshotValidationError(
            "human snapshot requires empty masked_cells (no evaluation masks)"
        )
    if root["held_out"] != {}:
        raise HumanSnapshotValidationError(
            "human snapshot requires empty held_out (no evaluation labels)"
        )

    meta = _validate_meta(root["meta"])
    clusters = _validate_clusters(
        root["clusters"],
        participants,
        k_anonymity=meta["k_anonymity"],
    )
    stats = _validate_stats(root["stats"], votes)
    return {
        "session_id": session_id,
        "statements": statements,
        "participants": participants,
        "votes": votes,
        "masked_cells": [],
        "held_out": {},
        "clusters": clusters,
        "stats": stats,
        "meta": meta,
    }


def _validate_statements(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise HumanSnapshotValidationError("statements must be a non-empty list")
    if len(value) > MAX_STATEMENTS:
        raise HumanSnapshotValidationError(
            f"statements must contain at most {MAX_STATEMENTS} entries"
        )
    statements: list[dict[str, Any]] = []
    for index, raw_statement in enumerate(value):
        statement = _exact_object(
            raw_statement, {"index", "text"}, f"statement {index}"
        )
        if type(statement["index"]) is not int or statement["index"] != index:
            raise HumanSnapshotValidationError(
                f"statement {index} must use positional index {index}"
            )
        text = _safe_text(statement["text"], f"statement {index}.text")
        statements.append({"index": index, "text": text})
    return statements


def _validate_participants(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) < MIN_K_ANONYMITY:
        raise HumanSnapshotValidationError(
            f"participants must contain at least {MIN_K_ANONYMITY} entries"
        )
    if len(value) > MAX_PARTICIPANTS:
        raise HumanSnapshotValidationError(
            f"participants must contain at most {MAX_PARTICIPANTS} entries"
        )
    expected = [f"p{index:03d}" for index in range(len(value))]
    if value != expected:
        raise HumanSnapshotValidationError(
            "participants must use positional pseudonyms p000, p001, ..."
        )
    return expected


def _validate_votes(
    value: Any, participant_count: int, statement_count: int
) -> list[list[int | None]]:
    if participant_count * statement_count > MAX_VOTE_CELLS:
        raise HumanSnapshotValidationError(
            f"votes must contain at most {MAX_VOTE_CELLS} cells"
        )
    if not isinstance(value, list) or len(value) != participant_count:
        raise HumanSnapshotValidationError("votes must be participant-major")
    votes: list[list[int | None]] = []
    observed = 0
    for participant_index, raw_row in enumerate(value):
        if not isinstance(raw_row, list) or len(raw_row) != statement_count:
            raise HumanSnapshotValidationError(
                f"votes row {participant_index} must contain {statement_count} cells"
            )
        row: list[int | None] = []
        for statement_index, vote in enumerate(raw_row):
            if vote is not None and (type(vote) is not int or vote not in VALID_VOTES):
                raise HumanSnapshotValidationError(
                    f"invalid vote at {participant_index},{statement_index}: {vote!r}"
                )
            observed += vote is not None
            row.append(vote)
        votes.append(row)
    if observed == 0:
        raise HumanSnapshotValidationError("votes must contain observed responses")
    return votes


def _validate_meta(value: Any) -> dict[str, Any]:
    meta = _exact_object(value, META_FIELDS, "meta")
    if meta["synthetic"] is not False:
        raise HumanSnapshotValidationError("meta.synthetic must be false")
    k_anonymity = meta["k_anonymity"]
    if type(k_anonymity) is not int or k_anonymity < MIN_K_ANONYMITY:
        raise HumanSnapshotValidationError(
            f"meta.k_anonymity must be at least {MIN_K_ANONYMITY}"
        )
    if meta["source"] != "context-engine-session":
        raise HumanSnapshotValidationError(
            'meta.source must equal "context-engine-session"'
        )
    if type(meta["seed"]) is not int:
        raise HumanSnapshotValidationError("meta.seed must be an integer")
    if meta["consent_scope"] != "public-benchmark":
        raise HumanSnapshotValidationError(
            'meta.consent_scope must equal "public-benchmark"'
        )
    if meta["redistribution_rights_approved"] is not True:
        raise HumanSnapshotValidationError(
            "meta.redistribution_rights_approved must be true"
        )
    if meta["schema_version"] != HUMAN_SNAPSHOT_SCHEMA_VERSION:
        raise HumanSnapshotValidationError(
            f"meta.schema_version must equal {HUMAN_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    exporter_version = meta["exporter_version"]
    if not isinstance(exporter_version, str) or not _SEMVER_PATTERN.fullmatch(
        exporter_version
    ):
        raise HumanSnapshotValidationError(
            "meta.exporter_version must be a semantic version"
        )
    source_commit = meta["source_commit"]
    if not isinstance(source_commit, str) or not _SOURCE_COMMIT_PATTERN.fullmatch(
        source_commit
    ):
        raise HumanSnapshotValidationError(
            "meta.source_commit must be a 40-character lowercase git commit"
        )
    privacy_review = _validate_privacy_review(meta["privacy_review"])
    return {
        "synthetic": False,
        "k_anonymity": k_anonymity,
        "source": "context-engine-session",
        "seed": meta["seed"],
        "consent_scope": "public-benchmark",
        "redistribution_rights_approved": True,
        "schema_version": HUMAN_SNAPSHOT_SCHEMA_VERSION,
        "exporter_version": exporter_version,
        "source_commit": source_commit,
        "privacy_review": privacy_review,
    }


def _validate_privacy_review(value: Any) -> dict[str, Any]:
    review = _exact_object(value, PRIVACY_REVIEW_FIELDS, "meta.privacy_review")
    if review["attested"] is not True:
        raise HumanSnapshotValidationError("meta.privacy_review.attested must be true")
    reviewed_at = _canonical_date(
        review["reviewed_at"], "meta.privacy_review.reviewed_at"
    )
    checks = review["checks"]
    if not isinstance(checks, list) or checks != sorted(REQUIRED_PRIVACY_CHECKS):
        raise HumanSnapshotValidationError(
            "meta.privacy_review.checks must contain the canonical required checks"
        )
    return {"attested": True, "reviewed_at": reviewed_at, "checks": list(checks)}


def _validate_clusters(
    value: Any,
    participants: Sequence[str],
    *,
    k_anonymity: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise HumanSnapshotValidationError("clusters must be a non-empty list")
    if len(value) > MAX_CLUSTERS:
        raise HumanSnapshotValidationError(
            f"clusters must contain at most {MAX_CLUSTERS} entries"
        )
    member_indices: list[int] = []
    cluster_ids: set[int] = set()
    clusters: list[dict[str, Any]] = []
    for cluster_index, raw_cluster in enumerate(value):
        cluster = _exact_object(raw_cluster, CLUSTER_FIELDS, f"cluster {cluster_index}")
        cluster_id = cluster["id"]
        if type(cluster_id) is not int or cluster_id in cluster_ids:
            raise HumanSnapshotValidationError(
                f"cluster {cluster_index} requires a unique integer id"
            )
        cluster_ids.add(cluster_id)
        raw_indices = cluster["member_indices"]
        if (
            not isinstance(raw_indices, list)
            or len(raw_indices) < k_anonymity
            or any(type(index) is not int for index in raw_indices)
        ):
            raise HumanSnapshotValidationError(
                f"cluster {cluster_index} must contain at least k={k_anonymity} member_indices"
            )
        expected_members = [
            participants[index]
            for index in raw_indices
            if 0 <= index < len(participants)
        ]
        raw_members = cluster["members"]
        if (
            len(expected_members) != len(raw_indices)
            or not isinstance(raw_members, list)
            or raw_members != expected_members
        ):
            raise HumanSnapshotValidationError(
                f"cluster {cluster_index} members must match participant indices"
            )
        center = cluster["center"]
        if center != []:
            raise HumanSnapshotValidationError(
                f"cluster {cluster_index} center must be empty in the human-v2 contract"
            )
        member_indices.extend(raw_indices)
        clusters.append(
            {
                "id": cluster_id,
                "members": list(raw_members),
                "member_indices": list(raw_indices),
                "center": [],
            }
        )
    if sorted(member_indices) != list(range(len(participants))):
        raise HumanSnapshotValidationError(
            "cluster member_indices must partition all participants exactly once"
        )
    return clusters


def _validate_stats(
    value: Any, votes: Sequence[Sequence[int | None]]
) -> dict[str, Any]:
    stats = _exact_object(value, {"comment"}, "stats")
    comments = stats["comment"]
    statement_count = len(votes[0])
    if not isinstance(comments, list) or len(comments) != statement_count:
        raise HumanSnapshotValidationError(
            f"stats.comment must contain {statement_count} entries"
        )
    validated: list[dict[str, int | None]] = []
    for statement_index, raw_comment in enumerate(comments):
        comment = _exact_object(
            raw_comment, COMMENT_STAT_FIELDS, f"stats.comment {statement_index}"
        )
        if (
            type(comment["commentIndex"]) is not int
            or comment["commentIndex"] != statement_index
        ):
            raise HumanSnapshotValidationError(
                f"stats.comment {statement_index} must use its positional commentIndex"
            )
        column = [row[statement_index] for row in votes]
        expected = {
            "agrees": column.count(1),
            "disagrees": column.count(-1),
            "unsure": column.count(0),
            "responded": sum(vote is not None for vote in column),
        }
        expected["total"] = expected["responded"]
        for field, expected_value in expected.items():
            if type(comment[field]) is not int or comment[field] != expected_value:
                raise HumanSnapshotValidationError(
                    f"stats.comment {statement_index}.{field} must equal {expected_value}"
                )
        for field in ("extremity", "divisiveness"):
            if comment[field] is not None:
                raise HumanSnapshotValidationError(
                    f"stats.comment {statement_index}.{field} must be null in the human-v2 contract"
                )
        validated.append(
            {
                "commentIndex": statement_index,
                **expected,
                "extremity": None,
                "divisiveness": None,
            }
        )
    return {"comment": validated}


def _safe_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise HumanSnapshotValidationError(f"{label} must be non-empty canonical text")
    category = find_direct_identifier(value)
    if category is not None:
        article = (
            "an"
            if category in {"email", "EVM address", "ENS name", "IP address"}
            else "a"
        )
        raise HumanSnapshotValidationError(f"{label} contains {article} {category}")
    return value


def _canonical_date(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise HumanSnapshotValidationError(f"{label} must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise HumanSnapshotValidationError(f"{label} must be a valid date") from error
    if parsed.isoformat() != value:
        raise HumanSnapshotValidationError(f"{label} must use YYYY-MM-DD")
    return value


def _exact_object(
    value: Any, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanSnapshotValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != set(expected):
        raise HumanSnapshotValidationError(
            f"{label} fields mismatch: missing={sorted(set(expected) - actual)} "
            f"unexpected={sorted(actual - set(expected))}"
        )
    return value
