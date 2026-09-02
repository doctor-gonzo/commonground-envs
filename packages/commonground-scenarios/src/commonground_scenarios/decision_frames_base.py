"""Hand-authored decision frames for the six non-helper scenario templates.

These frames name the policy decision represented by each planted issue.  They
are authored from the committed document passages and canonical question, not
derived by slicing tokens from either one.  Generated variants may add scope
language, but the underlying decision represented by a base plant stays fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

type DecisionFrameKey = tuple[str, str]
type DecisionFrame = Mapping[str, str]


def _frame(
    *,
    actor: str,
    action: str,
    condition: str,
    anchor_outcome: str,
    alternative_outcome: str,
) -> DecisionFrame:
    """Create one read-only, complete decision frame."""

    return MappingProxyType(
        {
            "actor": actor,
            "action": action,
            "condition": condition,
            "anchor_outcome": anchor_outcome,
            "alternative_outcome": alternative_outcome,
        }
    )


AUTHORED_BASE_DECISION_FRAMES: Final[Mapping[DecisionFrameKey, DecisionFrame]] = (
    MappingProxyType(
        {
            ("customer-support-handbook", "credit-threshold"): _frame(
                actor="support agents",
                action="issue goodwill credits",
                condition="material customer inconvenience",
                anchor_outcome="use a small-credit and material-inconvenience standard",
                alternative_outcome="set the credit amount and qualifying evidence",
            ),
            ("customer-support-handbook", "refund-window-conflict"): _frame(
                actor="support agents",
                action="accept subscription refund requests",
                condition="refund request timing",
                anchor_outcome="accept requests through the current billing cycle",
                alternative_outcome="require requests before the second weekly review",
            ),
            ("customer-support-handbook", "owner-unavailable-gap"): _frame(
                actor="support agents",
                action="authorize account recovery",
                condition="the account owner cannot be reached",
                anchor_outcome="escalate the access failure to the security queue",
                alternative_outcome="authorize recovery without reaching the owner",
            ),
            ("learning-platform-operations", "substantive-feedback"): _frame(
                actor="instructors",
                action="provide project feedback",
                condition="before grading is final",
                anchor_outcome="provide substantive feedback before final grading",
                alternative_outcome="define evidence that satisfies the substantive standard",
            ),
            ("learning-platform-operations", "revision-conflict"): _frame(
                actor="learners",
                action="submit project revisions",
                condition="after grading",
                anchor_outcome="allow one revision after grading",
                alternative_outcome="allow resubmission until the rubric is met",
            ),
            ("learning-platform-operations", "live-access-gap"): _frame(
                actor="learner success",
                action="grant an accessibility accommodation",
                condition="a live session has already started",
                anchor_outcome="route the accessibility request to learner success",
                alternative_outcome="grant an accommodation after the session starts",
            ),
            ("civic-assistant-guidance", "urgent-request"): _frame(
                actor="case workers",
                action="move requests ahead in the queue",
                condition="a request may be urgent",
                anchor_outcome="move urgent requests ahead",
                alternative_outcome="define the conditions that qualify as urgent",
            ),
            ("civic-assistant-guidance", "appeal-channel-conflict"): _frame(
                actor="residents",
                action="submit an official appeal",
                condition="using the digital portal or counter form",
                anchor_outcome="treat the digital portal submission as official",
                alternative_outcome="treat the paper counter form as equally official",
            ),
            ("civic-assistant-guidance", "no-address-gap"): _frame(
                actor="staff",
                action="deliver resident notices",
                condition="a resident has no verified address",
                anchor_outcome="mail notices to the verified address on file",
                alternative_outcome="use a non-digital notice channel without an address",
            ),
            ("benefits-communications", "reasonable-notice"): _frame(
                actor="employees",
                action="give notice before planned leave",
                condition="different kinds of planned leave",
                anchor_outcome="use a general reasonable-notice standard",
                alternative_outcome="set a separate notice period for each leave kind",
            ),
            ("benefits-communications", "stipend-approval-conflict"): _frame(
                actor="managers",
                action="approve learning purchases",
                condition="before purchase or after receiving the receipt",
                anchor_outcome="require approval before purchase for reimbursement",
                alternative_outcome="allow approval after the receipt is submitted",
            ),
            ("benefits-communications", "contractor-benefit-gap"): _frame(
                actor="contractors",
                action="seek learning benefit access",
                condition="contractor eligibility is unspecified",
                anchor_outcome="ask a manager about benefit access",
                alternative_outcome="make contractors eligible for learning benefits",
            ),
            ("freight-cooperative-manual", "unsafe-weather"): _frame(
                actor="dispatchers",
                action="pause a route",
                condition="conditions may be unsafe",
                anchor_outcome="pause the route when conditions are unsafe",
                alternative_outcome="define observable conditions that require a pause",
            ),
            ("freight-cooperative-manual", "handoff-authority-conflict"): _frame(
                actor="the duty coordinator",
                action="transfer a delayed load",
                condition="after hours without the assigned dispatcher",
                anchor_outcome="transfer the load to an available route",
                alternative_outcome="allow only the assigned dispatcher to transfer the load",
            ),
            ("freight-cooperative-manual", "unreachable-driver-gap"): _frame(
                actor="the duty coordinator",
                action="reassign a route",
                condition="the assigned driver cannot be reached",
                anchor_outcome="contact the assigned driver before changing the route",
                alternative_outcome="reassign the route when the driver is unreachable",
            ),
            ("creator-marketplace-playbook", "timely-response"): _frame(
                actor="creators",
                action="respond to buyer concerns",
                condition="different dispute classes",
                anchor_outcome="respond in a timely way",
                alternative_outcome="set a response window for each dispute class",
            ),
            ("creator-marketplace-playbook", "listing-removal-conflict"): _frame(
                actor="trust reviewers",
                action="remove a misleading listing",
                condition="while a dispute is under review",
                anchor_outcome="remove the misleading listing immediately",
                alternative_outcome="keep the listing active during review",
            ),
            ("creator-marketplace-playbook", "dashboard-access-gap"): _frame(
                actor="suspended creators",
                action="submit an appeal",
                condition="without listing-dashboard access",
                anchor_outcome="submit appeals from the listing dashboard",
                alternative_outcome="allow an appeal without dashboard access",
            ),
        }
    )
)


__all__ = [
    "AUTHORED_BASE_DECISION_FRAMES",
    "DecisionFrame",
    "DecisionFrameKey",
]
