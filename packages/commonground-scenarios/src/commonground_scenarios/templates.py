"""Committed domain templates for deterministic planted scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainTemplate:
    """A complete, immutable-by-convention scenario planting template."""

    template_id: str
    template_set: str
    sector: str
    organization_names: tuple[str, ...]
    factions: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, str], ...]
    planted_items: tuple[dict[str, Any], ...]
    distractors: tuple[dict[str, str], ...]


TRAIN_TEMPLATES = (
    DomainTemplate(
        template_id="customer-support-handbook",
        template_set="train",
        sector="subscription support",
        organization_names=(
            "Lantern Support Cooperative",
            "Harbor Desk Guild",
            "Juniper Service Works",
        ),
        factions=(
            {
                "faction_id": "support",
                "name": "Customer support",
                "summary": "Wants latitude to resolve customer harm quickly.",
                "priors": {
                    "credit-latitude": 0.9,
                    "refund-window": 0.8,
                    "escalation-gap": 0.7,
                },
            },
            {
                "faction_id": "finance",
                "name": "Finance controls",
                "summary": "Wants predictable approvals and bounded concessions.",
                "priors": {
                    "credit-latitude": -0.8,
                    "refund-window": -0.9,
                    "escalation-gap": 0.0,
                },
            },
            {
                "faction_id": "risk",
                "name": "Risk review",
                "summary": "Wants reviewable exceptions and explicit escalation ownership.",
                "priors": {
                    "credit-latitude": 0.0,
                    "refund-window": -0.6,
                    "escalation-gap": -0.9,
                },
            },
        ),
        documents=(
            {
                "doc_id": "service-policy",
                "title": "Service recovery policy",
                "style": "policy",
                "text": "Agents may issue a small goodwill credit when a customer has experienced material inconvenience. Credits are recorded in the case ledger before closure.",
            },
            {
                "doc_id": "refund-guide",
                "title": "Refund eligibility guide",
                "style": "handbook",
                "text": "Subscription refunds are available until the end of the current billing cycle. Duplicate charges are always routed to billing review.",
            },
            {
                "doc_id": "support-macro",
                "title": "Refund reply macro",
                "style": "macro",
                "text": "Tell customers that refund requests must be received before the second weekly review. Escalate account access failures to the security queue.",
            },
        ),
        planted_items=(
            {
                "plant_id": "credit-threshold",
                "doc_id": "service-policy",
                "anchor_quote": "Agents may issue a small goodwill credit when a customer has experienced material inconvenience.",
                "type": "ambiguity",
                "canonical_question": "Should support agents decide the credit amount and evidence that qualify as small and materially inconvenient?",
                "canonical_question_aliases": [],
                "target_dimension": "credit-latitude",
            },
            {
                "plant_id": "refund-window-conflict",
                "doc_id": "refund-guide",
                "anchor_quote": "Subscription refunds are available until the end of the current billing cycle.",
                "type": "contradiction",
                "canonical_question": "Should the billing-cycle window control instead of the support macro's weekly-review deadline?",
                "canonical_question_aliases": [],
                "target_dimension": "refund-window",
            },
            {
                "plant_id": "owner-unavailable-gap",
                "doc_id": "support-macro",
                "anchor_quote": "Escalate account access failures to the security queue.",
                "type": "gap",
                "canonical_question": "Should support agents authorize recovery when the account owner cannot be reached?",
                "canonical_question_aliases": [],
                "target_dimension": "escalation-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "service-policy",
                "anchor_quote": "Credits are recorded in the case ledger before closure.",
                "reason": "The action and timing are explicit.",
            },
            {
                "doc_id": "refund-guide",
                "anchor_quote": "Duplicate charges are always routed to billing review.",
                "reason": "The routing rule is precise and consistent.",
            },
        ),
    ),
    DomainTemplate(
        template_id="learning-platform-operations",
        template_set="train",
        sector="online learning",
        organization_names=(
            "Mosaic Learning Circle",
            "Northstar Lesson Studio",
            "Cedar Course Cooperative",
        ),
        factions=(
            {
                "faction_id": "instructors",
                "name": "Instructors",
                "summary": "Want discretion over feedback and revision pacing.",
                "priors": {
                    "feedback-depth": 0.9,
                    "revision-limit": -0.7,
                    "access-gap": 0.6,
                },
            },
            {
                "faction_id": "learner-success",
                "name": "Learner success",
                "summary": "Wants generous revision and accessibility support.",
                "priors": {
                    "feedback-depth": 0.6,
                    "revision-limit": 0.9,
                    "access-gap": 0.9,
                },
            },
            {
                "faction_id": "quality",
                "name": "Quality assurance",
                "summary": "Wants consistent workload and documented accommodations.",
                "priors": {
                    "feedback-depth": -0.8,
                    "revision-limit": -0.8,
                    "access-gap": -0.7,
                },
            },
        ),
        documents=(
            {
                "doc_id": "feedback-standard",
                "title": "Instructor feedback standard",
                "style": "standard",
                "text": "Each project receives substantive feedback before grading is final. Rubric categories are copied into the feedback record.",
            },
            {
                "doc_id": "revision-faq",
                "title": "Revision FAQ",
                "style": "faq",
                "text": "A learner may submit one revision after grading. Questions about rubric wording go to curriculum review.",
            },
            {
                "doc_id": "coach-playbook",
                "title": "Learner coaching playbook",
                "style": "playbook",
                "text": "Invite learners to resubmit until the rubric is met. Accessibility requests are routed to learner success.",
            },
        ),
        planted_items=(
            {
                "plant_id": "substantive-feedback",
                "doc_id": "feedback-standard",
                "anchor_quote": "Each project receives substantive feedback before grading is final.",
                "type": "ambiguity",
                "canonical_question": "Should instructors decide what evidence makes feedback substantive enough to satisfy the standard?",
                "canonical_question_aliases": [],
                "target_dimension": "feedback-depth",
            },
            {
                "plant_id": "revision-conflict",
                "doc_id": "revision-faq",
                "anchor_quote": "A learner may submit one revision after grading.",
                "type": "contradiction",
                "canonical_question": "Should learners be allowed to revise after grading until the rubric is met?",
                "canonical_question_aliases": [],
                "target_dimension": "revision-limit",
            },
            {
                "plant_id": "live-access-gap",
                "doc_id": "coach-playbook",
                "anchor_quote": "Accessibility requests are routed to learner success.",
                "type": "gap",
                "canonical_question": "Should learner success grant an accommodation after a live session has already started?",
                "canonical_question_aliases": [],
                "target_dimension": "access-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "feedback-standard",
                "anchor_quote": "Rubric categories are copied into the feedback record.",
                "reason": "The required record content is explicit.",
            },
            {
                "doc_id": "revision-faq",
                "anchor_quote": "Questions about rubric wording go to curriculum review.",
                "reason": "The escalation destination is unambiguous.",
            },
        ),
    ),
    DomainTemplate(
        template_id="civic-assistant-guidance",
        template_set="train",
        sector="municipal services",
        organization_names=(
            "Maple Civic Lab",
            "Open Borough Studio",
            "Riverbend Service Office",
        ),
        factions=(
            {
                "faction_id": "caseworkers",
                "name": "Case workers",
                "summary": "Want flexible triage and direct resolution.",
                "priors": {
                    "urgent-triage": 0.9,
                    "appeal-route": 0.8,
                    "offline-gap": 0.7,
                },
            },
            {
                "faction_id": "auditors",
                "name": "Program auditors",
                "summary": "Want uniform queues and complete records.",
                "priors": {
                    "urgent-triage": -0.8,
                    "appeal-route": -0.9,
                    "offline-gap": -0.7,
                },
            },
            {
                "faction_id": "residents",
                "name": "Resident advocates",
                "summary": "Want accessible channels and rapid review.",
                "priors": {
                    "urgent-triage": 0.7,
                    "appeal-route": 0.9,
                    "offline-gap": 0.9,
                },
            },
        ),
        documents=(
            {
                "doc_id": "triage-note",
                "title": "Service triage note",
                "style": "memo",
                "text": "Urgent requests should be moved ahead in the queue. Every queue change records the case number and operator.",
            },
            {
                "doc_id": "appeals-page",
                "title": "Appeals page",
                "style": "web page",
                "text": "Residents submit an appeal through the digital portal. Portal receipts are retained with the case.",
            },
            {
                "doc_id": "counter-script",
                "title": "Service-counter script",
                "style": "script",
                "text": "Provide a paper appeal form at the counter and treat it as the official submission. Mail notices to the verified address on file.",
            },
        ),
        planted_items=(
            {
                "plant_id": "urgent-request",
                "doc_id": "triage-note",
                "anchor_quote": "Urgent requests should be moved ahead in the queue.",
                "type": "ambiguity",
                "canonical_question": "Should case workers decide which conditions make a request urgent enough to move ahead?",
                "canonical_question_aliases": [],
                "target_dimension": "urgent-triage",
            },
            {
                "plant_id": "appeal-channel-conflict",
                "doc_id": "appeals-page",
                "anchor_quote": "Residents submit an appeal through the digital portal.",
                "type": "contradiction",
                "canonical_question": "Should a counter form be equally official as a digital portal appeal?",
                "canonical_question_aliases": [],
                "target_dimension": "appeal-route",
            },
            {
                "plant_id": "no-address-gap",
                "doc_id": "counter-script",
                "anchor_quote": "Mail notices to the verified address on file.",
                "type": "gap",
                "canonical_question": "Should staff use a non-digital notice channel when a resident has no verified address?",
                "canonical_question_aliases": [],
                "target_dimension": "offline-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "triage-note",
                "anchor_quote": "Every queue change records the case number and operator.",
                "reason": "The audit record is fully specified.",
            },
            {
                "doc_id": "appeals-page",
                "anchor_quote": "Portal receipts are retained with the case.",
                "reason": "The retention location is explicit.",
            },
        ),
    ),
    DomainTemplate(
        template_id="benefits-communications",
        template_set="train",
        sector="workplace benefits",
        organization_names=(
            "Willow People Collective",
            "Brightpath Benefits Studio",
            "Commonleaf Workplace Guild",
        ),
        factions=(
            {
                "faction_id": "people-ops",
                "name": "People operations",
                "summary": "Want accessible benefits and flexible administration.",
                "priors": {
                    "notice-window": 0.8,
                    "stipend-approval": 0.7,
                    "contractor-gap": 0.9,
                },
            },
            {
                "faction_id": "managers",
                "name": "Team managers",
                "summary": "Want local discretion and fast decisions.",
                "priors": {
                    "notice-window": 0.9,
                    "stipend-approval": 0.8,
                    "contractor-gap": 0.0,
                },
            },
            {
                "faction_id": "payroll",
                "name": "Payroll controls",
                "summary": "Want advance approval and clear eligibility boundaries.",
                "priors": {
                    "notice-window": -0.8,
                    "stipend-approval": -0.9,
                    "contractor-gap": -0.8,
                },
            },
        ),
        documents=(
            {
                "doc_id": "leave-guide",
                "title": "Leave communication guide",
                "style": "guide",
                "text": "Employees give reasonable notice before planned leave. Emergency leave may be reported through the duty channel.",
            },
            {
                "doc_id": "stipend-policy",
                "title": "Learning stipend policy",
                "style": "policy",
                "text": "Only learning purchases approved before purchase are reimbursed during the active benefit cycle. Receipts identify the vendor and course.",
            },
            {
                "doc_id": "manager-checklist",
                "title": "Manager reimbursement checklist",
                "style": "checklist",
                "text": "Managers may approve a learning purchase after the employee submits the receipt. Contractors ask their manager about benefit access.",
            },
        ),
        planted_items=(
            {
                "plant_id": "reasonable-notice",
                "doc_id": "leave-guide",
                "anchor_quote": "Employees give reasonable notice before planned leave.",
                "type": "ambiguity",
                "canonical_question": "Should teams set the reasonable notice period separately for each kind of planned leave?",
                "canonical_question_aliases": [],
                "target_dimension": "notice-window",
            },
            {
                "plant_id": "stipend-approval-conflict",
                "doc_id": "stipend-policy",
                "anchor_quote": "Only learning purchases approved before purchase are reimbursed during the active benefit cycle.",
                "type": "contradiction",
                "canonical_question": "Should a manager be allowed to approve a learning purchase after receiving the receipt?",
                "canonical_question_aliases": [],
                "target_dimension": "stipend-approval",
            },
            {
                "plant_id": "contractor-benefit-gap",
                "doc_id": "manager-checklist",
                "anchor_quote": "Contractors ask their manager about benefit access.",
                "type": "gap",
                "canonical_question": "Should contractors be eligible for learning benefits?",
                "canonical_question_aliases": [],
                "target_dimension": "contractor-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "leave-guide",
                "anchor_quote": "Emergency leave may be reported through the duty channel.",
                "reason": "The reporting route is explicit.",
            },
            {
                "doc_id": "stipend-policy",
                "anchor_quote": "Receipts identify the vendor and course.",
                "reason": "The required receipt fields are precise.",
            },
        ),
    ),
)


HELDOUT_TEMPLATES: tuple[DomainTemplate, ...] = (
    DomainTemplate(
        template_id="freight-cooperative-manual",
        template_set="heldout",
        sector="freight logistics",
        organization_names=(
            "Blue Heron Freight Cooperative",
            "Granite Route Guild",
            "Prairie Relay Works",
        ),
        factions=(
            {
                "faction_id": "dispatch",
                "name": "Dispatchers",
                "summary": "Want routing discretion and fast exception handling.",
                "priors": {
                    "weather-threshold": 0.9,
                    "handoff-authority": 0.8,
                    "silent-driver-gap": 0.7,
                },
            },
            {
                "faction_id": "safety",
                "name": "Safety officers",
                "summary": "Want conservative stops and explicit handoff authority.",
                "priors": {
                    "weather-threshold": -0.9,
                    "handoff-authority": -0.8,
                    "silent-driver-gap": -0.9,
                },
            },
            {
                "faction_id": "customers",
                "name": "Customer coordinators",
                "summary": "Want continuity and timely delivery updates.",
                "priors": {
                    "weather-threshold": 0.6,
                    "handoff-authority": 0.9,
                    "silent-driver-gap": 0.8,
                },
            },
        ),
        documents=(
            {
                "doc_id": "route-card",
                "title": "Severe-weather route card",
                "style": "field card",
                "text": "Record every pause in the dispatch log. Pause a route when conditions become unsafe.",
            },
            {
                "doc_id": "handoff-appendix",
                "title": "Load handoff appendix",
                "style": "appendix",
                "text": "The receiving driver confirms the seal identifier. Only the assigned dispatcher may transfer a load.",
            },
            {
                "doc_id": "radio-script",
                "title": "After-hours radio script",
                "style": "radio script",
                "text": "The duty coordinator may transfer a delayed load to an available route. Contact the assigned driver before changing the route.",
            },
        ),
        planted_items=(
            {
                "plant_id": "unsafe-weather",
                "doc_id": "route-card",
                "anchor_quote": "Pause a route when conditions become unsafe.",
                "type": "ambiguity",
                "canonical_question": "Should dispatchers decide which observable conditions require a route pause?",
                "canonical_question_aliases": [
                    "Should dispatchers choose which observable conditions warrant pausing a route?"
                ],
                "target_dimension": "weather-threshold",
            },
            {
                "plant_id": "handoff-authority-conflict",
                "doc_id": "radio-script",
                "anchor_quote": "The duty coordinator may transfer a delayed load to an available route.",
                "type": "contradiction",
                "canonical_question": "Should the duty coordinator transfer a load after hours without the assigned dispatcher?",
                "canonical_question_aliases": [],
                "target_dimension": "handoff-authority",
            },
            {
                "plant_id": "unreachable-driver-gap",
                "doc_id": "radio-script",
                "anchor_quote": "Contact the assigned driver before changing the route.",
                "type": "gap",
                "canonical_question": "Should the duty coordinator reassign a route when the assigned driver cannot be reached?",
                "canonical_question_aliases": [],
                "target_dimension": "silent-driver-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "route-card",
                "anchor_quote": "Record every pause in the dispatch log.",
                "reason": "The record location is explicit.",
            },
            {
                "doc_id": "handoff-appendix",
                "anchor_quote": "The receiving driver confirms the seal identifier.",
                "reason": "The confirmation requirement is precise.",
            },
        ),
    ),
    DomainTemplate(
        template_id="creator-marketplace-playbook",
        template_set="heldout",
        sector="digital marketplace",
        organization_names=(
            "Copper Kite Marketplace",
            "Fieldnote Creator Exchange",
            "Silver Loom Cooperative",
        ),
        factions=(
            {
                "faction_id": "creators",
                "name": "Creators",
                "summary": "Want flexible remedies and control over listings.",
                "priors": {
                    "timely-response": 0.9,
                    "removal-authority": -0.8,
                    "appeal-gap": 0.9,
                },
            },
            {
                "faction_id": "trust",
                "name": "Trust and safety",
                "summary": "Want rapid enforcement and reviewable appeals.",
                "priors": {
                    "timely-response": -0.8,
                    "removal-authority": 0.9,
                    "appeal-gap": 0.8,
                },
            },
            {
                "faction_id": "buyers",
                "name": "Buyer advocates",
                "summary": "Want clear deadlines and stable remedies.",
                "priors": {
                    "timely-response": -0.7,
                    "removal-authority": 0.7,
                    "appeal-gap": -0.8,
                },
            },
        ),
        documents=(
            {
                "doc_id": "message-guide",
                "title": "Creator message guide",
                "style": "style guide",
                "text": "Confirm resolution in the order thread. Respond to buyer concerns in a timely way.",
            },
            {
                "doc_id": "listing-rules",
                "title": "Listing integrity rules",
                "style": "rules",
                "text": "Removed listing records retain the cited rule. Trust reviewers may remove a misleading listing immediately.",
            },
            {
                "doc_id": "creator-playbook",
                "title": "Creator dispute playbook",
                "style": "decision tree",
                "text": "Appeals are submitted from the listing dashboard. Appeal receipts are attached to the dispute record. A creator keeps a listing active while a dispute is under review.",
            },
        ),
        planted_items=(
            {
                "plant_id": "timely-response",
                "doc_id": "message-guide",
                "anchor_quote": "Respond to buyer concerns in a timely way.",
                "type": "ambiguity",
                "canonical_question": "Should creators decide what response window counts as timely for each dispute class?",
                "canonical_question_aliases": [],
                "target_dimension": "timely-response",
            },
            {
                "plant_id": "listing-removal-conflict",
                "doc_id": "listing-rules",
                "anchor_quote": "Trust reviewers may remove a misleading listing immediately.",
                "type": "contradiction",
                "canonical_question": "Should immediate trust removal override the creator's keep-active rule during review?",
                "canonical_question_aliases": [],
                "target_dimension": "removal-authority",
            },
            {
                "plant_id": "dashboard-access-gap",
                "doc_id": "creator-playbook",
                "anchor_quote": "Appeals are submitted from the listing dashboard.",
                "type": "gap",
                "canonical_question": "Should a suspended creator be allowed to appeal without dashboard access?",
                "canonical_question_aliases": [],
                "target_dimension": "appeal-gap",
            },
        ),
        distractors=(
            {
                "doc_id": "message-guide",
                "anchor_quote": "Confirm resolution in the order thread.",
                "reason": "The confirmation location is explicit.",
            },
            {
                "doc_id": "creator-playbook",
                "anchor_quote": "Appeal receipts are attached to the dispute record.",
                "reason": "The receipt location is precisely specified.",
            },
        ),
    ),
)


def _additional_heldout_template(
    spec: tuple[str, ...], pattern_code: int
) -> DomainTemplate:
    """Build one compact, authored held-out domain with a unique plant layout."""

    (
        template_id,
        sector,
        ambiguity_anchor,
        ambiguity_question,
        contradiction_anchor,
        conflicting_rule,
        contradiction_question,
        gap_anchor,
        gap_question,
    ) = spec
    dimensions = {
        "ambiguity": f"{template_id}-threshold",
        "contradiction": f"{template_id}-authority",
        "gap": f"{template_id}-exception",
    }
    fillers = (
        "The document owner records each revision.",
        "Approved copies carry a control code.",
        "Archived versions remain read-only.",
    )

    def document(
        doc_id: str,
        title: str,
        style: str,
        anchor: str,
        distractor: str | None,
        prefix_count: int,
        *extra: str,
    ) -> dict[str, str]:
        sentences = [*fillers[:prefix_count], anchor]
        if distractor is not None:
            sentences.append(distractor)
        sentences.extend(extra)
        return {
            "doc_id": doc_id,
            "title": title,
            "style": style,
            "text": " ".join(sentences),
        }

    ambiguity_distractor = "Completed reviews are dated in the case record."
    contradiction_distractor = "The responsible desk logs each completed handoff."
    documents = [
        document(
            "scope-note",
            "Scope review note",
            "review note",
            ambiguity_anchor,
            ambiguity_distractor,
            pattern_code % 3,
        ),
        document(
            "authority-bulletin",
            "Authority bulletin",
            "authority bulletin",
            contradiction_anchor,
            contradiction_distractor,
            (pattern_code // 3) % 3,
        ),
        document(
            "exception-card",
            "Exception response card",
            "contingency card",
            gap_anchor,
            None,
            2 + pattern_code // 9,
            conflicting_rule,
        ),
    ]
    rotation = pattern_code % len(documents)
    documents = documents[rotation:] + documents[:rotation]
    return DomainTemplate(
        template_id=template_id,
        template_set="heldout",
        sector=sector,
        organization_names=(
            f"Northwind {sector.title()} Cooperative",
            f"Juniper {sector.title()} Network",
            f"Harbor {sector.title()} Guild",
        ),
        factions=(
            {
                "faction_id": "operators",
                "name": "Front-line operators",
                "summary": "Want practical discretion and continuity of service.",
                "priors": {dimension: 0.9 for dimension in dimensions.values()},
            },
            {
                "faction_id": "assurance",
                "name": "Assurance reviewers",
                "summary": "Want conservative thresholds and explicit authority.",
                "priors": {dimension: -0.9 for dimension in dimensions.values()},
            },
            {
                "faction_id": "community",
                "name": "Affected community",
                "summary": "Wants accessible exceptions and timely resolution.",
                "priors": {dimension: 0.7 for dimension in dimensions.values()},
            },
        ),
        documents=tuple(documents),
        planted_items=(
            {
                "plant_id": "scope-threshold",
                "doc_id": "scope-note",
                "anchor_quote": ambiguity_anchor,
                "type": "ambiguity",
                "canonical_question": ambiguity_question,
                "canonical_question_aliases": [],
                "target_dimension": dimensions["ambiguity"],
            },
            {
                "plant_id": "authority-conflict",
                "doc_id": "authority-bulletin",
                "anchor_quote": contradiction_anchor,
                "type": "contradiction",
                "canonical_question": contradiction_question,
                "canonical_question_aliases": [],
                "target_dimension": dimensions["contradiction"],
            },
            {
                "plant_id": "uncovered-exception",
                "doc_id": "exception-card",
                "anchor_quote": gap_anchor,
                "type": "gap",
                "canonical_question": gap_question,
                "canonical_question_aliases": [],
                "target_dimension": dimensions["gap"],
            },
        ),
        distractors=(
            {
                "doc_id": "scope-note",
                "anchor_quote": ambiguity_distractor,
                "reason": "The review record and timing are explicit.",
            },
            {
                "doc_id": "authority-bulletin",
                "anchor_quote": contradiction_distractor,
                "reason": "The handoff record is precisely specified.",
            },
        ),
    )


_ADDITIONAL_HELDOUT_SPECS = (
    (
        "community-clinic-scheduling",
        "community health scheduling",
        "Priority patients are offered an early appointment.",
        "Should schedulers decide which conditions make a patient eligible for an early appointment?",
        "Telehealth visits require identity confirmation before booking.",
        "Urgent telehealth slots may be booked before identity confirmation.",
        "Should urgent telehealth booking override the identity-confirmation requirement?",
        "Preparation instructions are sent through the patient portal.",
        "Should staff use another channel when a patient cannot access the portal?",
    ),
    (
        "cooperative-housing-maintenance",
        "cooperative housing maintenance",
        "Urgent repairs should be handled promptly.",
        "Should maintenance coordinators decide which repairs are urgent and how quickly they must respond?",
        "A manager must approve entry to an occupied unit.",
        "Emergency crews may enter an occupied unit immediately to stop active damage.",
        "Should emergency entry proceed without manager approval when damage is active?",
        "Repair notices are posted in the resident application.",
        "Should residents without application access receive repair notices another way?",
    ),
    (
        "library-digitization-rules",
        "library digitization",
        "Culturally sensitive materials receive appropriate access restrictions.",
        "Should archivists decide which materials are culturally sensitive and which restrictions are appropriate?",
        "Researchers may download full-resolution preservation scans.",
        "The rights manual permits researchers to receive excerpt images only.",
        "Should the excerpt-only rights rule override full-resolution scan access?",
        "Takedown requests are submitted through the catalog account.",
        "Should a requester without catalog access have another takedown route?",
    ),
    (
        "food-bank-distribution",
        "community food distribution",
        "Households with high need receive additional staple items.",
        "Should intake volunteers decide what qualifies as high need and how many staples to add?",
        "Reserved food boxes remain assigned until the distribution window closes.",
        "Unclaimed boxes may be reassigned thirty minutes after the scheduled pickup.",
        "Should the closing-time reservation rule override the thirty-minute reassignment rule?",
        "A proxy collector presents the household identification card.",
        "Should a proxy collect food when the household identification card is unavailable?",
    ),
    (
        "renewable-microgrid-operations",
        "community microgrid operations",
        "Operators maintain an adequate battery reserve overnight.",
        "Should operators decide what battery level is adequate for the overnight reserve?",
        "Battery discharge below the safety reserve is prohibited.",
        "Peak-response operators may discharge below the reserve to prevent a feeder trip.",
        "Should feeder protection permit discharge below the stated safety reserve?",
        "The site lead is contacted before the microgrid enters island mode.",
        "Should operators enter island mode when the site lead cannot be reached?",
    ),
    (
        "museum-loan-handling",
        "museum collection loans",
        "Loans with significant damage receive immediate conservation review.",
        "Should registrars decide what damage is significant enough for immediate conservation review?",
        "Only the assigned registrar may authorize a courier transfer.",
        "A courier may transfer a delayed loan to another approved courier.",
        "Should a delayed-loan transfer proceed without the assigned registrar's authorization?",
        "The borrower returns the object through the designated border crossing.",
        "Should another return route be allowed when the designated border crossing closes?",
    ),
    (
        "open-source-grant-program",
        "open source grants",
        "Projects with substantial community benefit receive priority review.",
        "Should grant reviewers decide what counts as substantial community benefit?",
        "Expenses are reimbursed only when approved before purchase.",
        "The steering group may ratify an essential project expense after purchase.",
        "Should retrospective steering-group ratification override the preapproval rule?",
        "The lead maintainer signs the final grant report.",
        "Should another maintainer sign when the lead maintainer is unavailable?",
    ),
    (
        "agricultural-water-allocation",
        "agricultural water allocation",
        "Severe shortages trigger reduced irrigation allocations.",
        "Should coordinators decide which measurements constitute a severe shortage?",
        "Weekly water allocations remain fixed until the next published schedule.",
        "The emergency desk may adjust allocations daily during a canal failure.",
        "Should emergency daily adjustments override the published weekly allocation?",
        "Field allocations rely on the upstream flow sensor.",
        "Should coordinators allocate water when the upstream flow sensor is offline?",
    ),
    (
        "disaster-shelter-intake",
        "emergency shelter intake",
        "Vulnerable residents receive priority for available shelter beds.",
        "Should intake workers decide which circumstances make a resident vulnerable?",
        "A bed is held only after the resident presents identification.",
        "Crisis intake may reserve a bed before identification is available.",
        "Should crisis intake reserve a bed without the identification required by the holding rule?",
        "Family members are assigned together when they arrive as a group.",
        "Should staff reunite family members who arrive separately after assignments are complete?",
    ),
    (
        "translation-quality-service",
        "public translation services",
        "Sensitive content receives an appropriately senior language review.",
        "Should coordinators decide which content is sensitive and which reviewer is senior enough?",
        "No translation is published before senior review is complete.",
        "Urgent safety notices may be published before senior review.",
        "Should urgent safety publication override the completed-review requirement?",
        "Requests are assigned to a language listed in the reviewer roster.",
        "Should an unlisted language request be routed outside the reviewer roster?",
    ),
    (
        "research-computing-queue",
        "research computing allocation",
        "High-impact computing runs receive expedited queue placement.",
        "Should queue operators decide which research runs are high impact?",
        "A compute allocation cannot be transferred between projects.",
        "The night operator may transfer idle capacity to a delayed project.",
        "Should the night operator transfer idle capacity despite the nontransfer rule?",
        "A principal investigator confirms every emergency queue change.",
        "Should an emergency queue change proceed when the principal investigator is unreachable?",
    ),
    (
        "makerspace-tool-safety",
        "community makerspace safety",
        "Tools judged dangerous require advanced orientation.",
        "Should floor supervisors decide which tools are dangerous enough to require advanced orientation?",
        "Guests may use powered tools only after completing orientation.",
        "Event guests may use a powered tool while directly supervised.",
        "Should direct event supervision override the completed-orientation requirement?",
        "A designated supervisor remains present throughout powered-tool use.",
        "Should tool use stop when the designated supervisor must leave unexpectedly?",
    ),
    (
        "transit-accessibility-service",
        "accessible public transit",
        "Operators provide reasonable boarding assistance on request.",
        "Should operators decide which boarding assistance is reasonable in each situation?",
        "Large mobility devices require a reservation before boarding.",
        "Operators must board a waiting mobility-device user when safe space is available.",
        "Should available safe space override the mobility-device reservation requirement?",
        "Riders are directed to the station lift during step-free boarding.",
        "Should operators provide another boarding method when the station lift is out of service?",
    ),
    (
        "fisheries-cooperative-landing",
        "cooperative fisheries landing",
        "Vessels return early when weather becomes poor.",
        "Should skippers decide which observed weather conditions require an early return?",
        "Only the dock officer may authorize transfer of a catch between vessels.",
        "A skipper may transfer a catch at sea to preserve it during delay.",
        "Should preservation needs permit an at-sea transfer without dock-officer authorization?",
        "The vessel contacts the landing desk before changing its destination.",
        "Should a vessel change destination when communications with the landing desk are lost?",
    ),
    (
        "regional-archives-access",
        "regional archives access",
        "Sensitive files receive suitable reading-room restrictions.",
        "Should archivists decide which files are sensitive and which restrictions are suitable?",
        "Researchers may not make copies in the restricted reading room.",
        "The researcher guide permits phone photographs of cited pages.",
        "Should the no-copy reading-room rule prohibit photographs allowed by the researcher guide?",
        "Permission requests are sent to the record's living author.",
        "Should another authority decide permission when the record's author has died?",
    ),
    (
        "rural-broadband-outages",
        "rural broadband operations",
        "Widespread outages receive an accelerated field response.",
        "Should dispatchers decide how many affected connections make an outage widespread?",
        "Only the network operations center may dispatch a field engineer.",
        "The duty lead may dispatch an engineer directly during an overnight outage.",
        "Should overnight duty-lead dispatch override network-operations-center authority?",
        "Customers receive restoration instructions by text message.",
        "Should disconnected customers receive restoration instructions through another channel?",
    ),
    (
        "school-meal-accommodations",
        "school meal services",
        "Exceptional dietary needs receive a customized meal plan.",
        "Should meal coordinators decide which dietary needs are exceptional?",
        "Accommodation forms must arrive before the monthly menu is finalized.",
        "School nurses may approve a dietary accommodation after menu finalization.",
        "Should nurse approval override the menu-finalization deadline?",
        "A guardian confirms every change to a student's meal plan.",
        "Should a safety-related meal change proceed when the guardian cannot be reached?",
    ),
    (
        "water-quality-laboratory",
        "water quality laboratory",
        "Anomalous readings receive additional laboratory review.",
        "Should analysts decide which deviation makes a reading anomalous?",
        "Results are released only after a duplicate test confirms them.",
        "The incident guide requires immediate release of a potential contamination result.",
        "Should potential contamination be released before the duplicate test is complete?",
        "Every result cites the instrument's current calibration record.",
        "Should a result be withheld when the current calibration record is missing?",
    ),
)

HELDOUT_TEMPLATES += tuple(
    _additional_heldout_template(spec, pattern_code)
    for pattern_code, spec in enumerate(_ADDITIONAL_HELDOUT_SPECS)
)


_TEMPLATES = {
    template.template_id: template for template in TRAIN_TEMPLATES + HELDOUT_TEMPLATES
}


def get_template(template_id: str) -> DomainTemplate:
    """Return a committed template by ID."""

    try:
        return _TEMPLATES[template_id]
    except KeyError as error:
        raise KeyError(f"unknown domain template: {template_id}") from error
