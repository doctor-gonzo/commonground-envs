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
    planted_items: tuple[dict[str, str], ...]
    distractors: tuple[dict[str, str], ...]


TRAIN_TEMPLATES = (
    DomainTemplate(
        template_id="customer-support-handbook",
        template_set="train",
        sector="subscription support",
        organization_names=("Lantern Support Cooperative", "Harbor Desk Guild", "Juniper Service Works"),
        factions=(
            {
                "faction_id": "support",
                "name": "Customer support",
                "summary": "Wants latitude to resolve customer harm quickly.",
                "priors": {"credit-latitude": 0.9, "refund-window": 0.8, "escalation-gap": 0.7},
            },
            {
                "faction_id": "finance",
                "name": "Finance controls",
                "summary": "Wants predictable approvals and bounded concessions.",
                "priors": {"credit-latitude": -0.8, "refund-window": -0.9, "escalation-gap": 0.0},
            },
            {
                "faction_id": "risk",
                "name": "Risk review",
                "summary": "Wants reviewable exceptions and explicit escalation ownership.",
                "priors": {"credit-latitude": 0.0, "refund-window": -0.6, "escalation-gap": -0.9},
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
                "canonical_question": "What credit amount and evidence qualify as small and materially inconvenient?",
                "target_dimension": "credit-latitude",
            },
            {
                "plant_id": "refund-window-conflict",
                "doc_id": "refund-guide",
                "anchor_quote": "Subscription refunds are available until the end of the current billing cycle.",
                "type": "contradiction",
                "canonical_question": "Does the billing-cycle window or the support macro's weekly-review deadline control?",
                "target_dimension": "refund-window",
            },
            {
                "plant_id": "owner-unavailable-gap",
                "doc_id": "support-macro",
                "anchor_quote": "Escalate account access failures to the security queue.",
                "type": "gap",
                "canonical_question": "Who may authorize recovery when the account owner cannot be reached?",
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
        organization_names=("Mosaic Learning Circle", "Northstar Lesson Studio", "Cedar Course Cooperative"),
        factions=(
            {
                "faction_id": "instructors",
                "name": "Instructors",
                "summary": "Want discretion over feedback and revision pacing.",
                "priors": {"feedback-depth": 0.9, "revision-limit": -0.7, "access-gap": 0.6},
            },
            {
                "faction_id": "learner-success",
                "name": "Learner success",
                "summary": "Wants generous revision and accessibility support.",
                "priors": {"feedback-depth": 0.6, "revision-limit": 0.9, "access-gap": 0.9},
            },
            {
                "faction_id": "quality",
                "name": "Quality assurance",
                "summary": "Wants consistent workload and documented accommodations.",
                "priors": {"feedback-depth": -0.8, "revision-limit": -0.8, "access-gap": -0.7},
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
                "canonical_question": "What evidence makes feedback substantive enough to satisfy the standard?",
                "target_dimension": "feedback-depth",
            },
            {
                "plant_id": "revision-conflict",
                "doc_id": "revision-faq",
                "anchor_quote": "A learner may submit one revision after grading.",
                "type": "contradiction",
                "canonical_question": "Is revision limited after grading or allowed until the rubric is met?",
                "target_dimension": "revision-limit",
            },
            {
                "plant_id": "live-access-gap",
                "doc_id": "coach-playbook",
                "anchor_quote": "Accessibility requests are routed to learner success.",
                "type": "gap",
                "canonical_question": "What accommodation applies when a live session has already started?",
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
        organization_names=("Maple Civic Lab", "Open Borough Studio", "Riverbend Service Office"),
        factions=(
            {
                "faction_id": "caseworkers",
                "name": "Case workers",
                "summary": "Want flexible triage and direct resolution.",
                "priors": {"urgent-triage": 0.9, "appeal-route": 0.8, "offline-gap": 0.7},
            },
            {
                "faction_id": "auditors",
                "name": "Program auditors",
                "summary": "Want uniform queues and complete records.",
                "priors": {"urgent-triage": -0.8, "appeal-route": -0.9, "offline-gap": -0.7},
            },
            {
                "faction_id": "residents",
                "name": "Resident advocates",
                "summary": "Want accessible channels and rapid review.",
                "priors": {"urgent-triage": 0.7, "appeal-route": 0.9, "offline-gap": 0.9},
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
                "canonical_question": "Which conditions make a request urgent enough to move ahead?",
                "target_dimension": "urgent-triage",
            },
            {
                "plant_id": "appeal-channel-conflict",
                "doc_id": "appeals-page",
                "anchor_quote": "Residents submit an appeal through the digital portal.",
                "type": "contradiction",
                "canonical_question": "Is the digital portal mandatory or is a counter form equally official?",
                "target_dimension": "appeal-route",
            },
            {
                "plant_id": "no-address-gap",
                "doc_id": "counter-script",
                "anchor_quote": "Mail notices to the verified address on file.",
                "type": "gap",
                "canonical_question": "How is notice delivered when a resident has no verified address?",
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
        organization_names=("Willow People Collective", "Brightpath Benefits Studio", "Commonleaf Workplace Guild"),
        factions=(
            {
                "faction_id": "people-ops",
                "name": "People operations",
                "summary": "Want accessible benefits and flexible administration.",
                "priors": {"notice-window": 0.8, "stipend-approval": 0.7, "contractor-gap": 0.9},
            },
            {
                "faction_id": "managers",
                "name": "Team managers",
                "summary": "Want local discretion and fast decisions.",
                "priors": {"notice-window": 0.9, "stipend-approval": 0.8, "contractor-gap": 0.0},
            },
            {
                "faction_id": "payroll",
                "name": "Payroll controls",
                "summary": "Want advance approval and clear eligibility boundaries.",
                "priors": {"notice-window": -0.8, "stipend-approval": -0.9, "contractor-gap": -0.8},
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
                "canonical_question": "What notice period is reasonable for each kind of planned leave?",
                "target_dimension": "notice-window",
            },
            {
                "plant_id": "stipend-approval-conflict",
                "doc_id": "stipend-policy",
                "anchor_quote": "Only learning purchases approved before purchase are reimbursed during the active benefit cycle.",
                "type": "contradiction",
                "canonical_question": "Must approval precede purchase or may a manager approve after receiving a receipt?",
                "target_dimension": "stipend-approval",
            },
            {
                "plant_id": "contractor-benefit-gap",
                "doc_id": "manager-checklist",
                "anchor_quote": "Contractors ask their manager about benefit access.",
                "type": "gap",
                "canonical_question": "Which contractor categories are eligible for learning benefits?",
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


HELDOUT_TEMPLATES = (
    DomainTemplate(
        template_id="freight-cooperative-manual",
        template_set="heldout",
        sector="freight logistics",
        organization_names=("Blue Heron Freight Cooperative", "Granite Route Guild", "Prairie Relay Works"),
        factions=(
            {
                "faction_id": "dispatch",
                "name": "Dispatchers",
                "summary": "Want routing discretion and fast exception handling.",
                "priors": {"weather-threshold": 0.9, "handoff-authority": 0.8, "silent-driver-gap": 0.7},
            },
            {
                "faction_id": "safety",
                "name": "Safety officers",
                "summary": "Want conservative stops and explicit handoff authority.",
                "priors": {"weather-threshold": -0.9, "handoff-authority": -0.8, "silent-driver-gap": -0.9},
            },
            {
                "faction_id": "customers",
                "name": "Customer coordinators",
                "summary": "Want continuity and timely delivery updates.",
                "priors": {"weather-threshold": 0.6, "handoff-authority": 0.9, "silent-driver-gap": 0.8},
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
                "canonical_question": "Which observable conditions require a route pause?",
                "target_dimension": "weather-threshold",
            },
            {
                "plant_id": "handoff-authority-conflict",
                "doc_id": "radio-script",
                "anchor_quote": "The duty coordinator may transfer a delayed load to an available route.",
                "type": "contradiction",
                "canonical_question": "May the duty coordinator transfer a load after hours without the assigned dispatcher?",
                "target_dimension": "handoff-authority",
            },
            {
                "plant_id": "unreachable-driver-gap",
                "doc_id": "radio-script",
                "anchor_quote": "Contact the assigned driver before changing the route.",
                "type": "gap",
                "canonical_question": "What happens when the assigned driver cannot be reached?",
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
        organization_names=("Copper Kite Marketplace", "Fieldnote Creator Exchange", "Silver Loom Cooperative"),
        factions=(
            {
                "faction_id": "creators",
                "name": "Creators",
                "summary": "Want flexible remedies and control over listings.",
                "priors": {"timely-response": 0.9, "removal-authority": -0.8, "appeal-gap": 0.9},
            },
            {
                "faction_id": "trust",
                "name": "Trust and safety",
                "summary": "Want rapid enforcement and reviewable appeals.",
                "priors": {"timely-response": -0.8, "removal-authority": 0.9, "appeal-gap": 0.8},
            },
            {
                "faction_id": "buyers",
                "name": "Buyer advocates",
                "summary": "Want clear deadlines and stable remedies.",
                "priors": {"timely-response": -0.7, "removal-authority": 0.7, "appeal-gap": -0.8},
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
                "canonical_question": "What response window counts as timely for each dispute class?",
                "target_dimension": "timely-response",
            },
            {
                "plant_id": "listing-removal-conflict",
                "doc_id": "listing-rules",
                "anchor_quote": "Trust reviewers may remove a misleading listing immediately.",
                "type": "contradiction",
                "canonical_question": "Does immediate trust removal override the creator's keep-active rule during review?",
                "target_dimension": "removal-authority",
            },
            {
                "plant_id": "dashboard-access-gap",
                "doc_id": "creator-playbook",
                "anchor_quote": "Appeals are submitted from the listing dashboard.",
                "type": "gap",
                "canonical_question": "How can a suspended creator appeal without dashboard access?",
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


_TEMPLATES = {template.template_id: template for template in TRAIN_TEMPLATES + HELDOUT_TEMPLATES}


def get_template(template_id: str) -> DomainTemplate:
    """Return a committed template by ID."""

    try:
        return _TEMPLATES[template_id]
    except KeyError as error:
        raise KeyError(f"unknown domain template: {template_id}") from error
