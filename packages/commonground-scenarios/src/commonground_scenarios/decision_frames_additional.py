"""Authored decision frames for the additional held-out policy templates.

These frames state the semantic decision represented by each planted issue.
They are deliberately authored from the policy rules and canonical question,
not synthesized from token positions or lexical overlap.  ``anchor_outcome``
always preserves the primary planted passage; ``alternative_outcome`` states
the clarification, fallback, or conflicting rule under consideration.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

DecisionFrame = Mapping[str, str]
TemplateDecisionFrames = Mapping[str, DecisionFrame]

_FRAME_FIELDS: Final = frozenset(
    {
        "actor",
        "action",
        "condition",
        "anchor_outcome",
        "alternative_outcome",
    }
)


def _frame(
    *,
    actor: str,
    action: str,
    condition: str,
    anchor_outcome: str,
    alternative_outcome: str,
) -> DecisionFrame:
    """Freeze one complete, non-empty five-field decision frame."""

    frame = {
        "actor": actor,
        "action": action,
        "condition": condition,
        "anchor_outcome": anchor_outcome,
        "alternative_outcome": alternative_outcome,
    }
    if set(frame) != _FRAME_FIELDS or any(
        not value.strip() for value in frame.values()
    ):
        raise ValueError("decision frames require five non-empty fields")
    return MappingProxyType(frame)


def _template(
    *,
    scope_threshold: DecisionFrame,
    authority_conflict: DecisionFrame,
    uncovered_exception: DecisionFrame,
) -> TemplateDecisionFrames:
    """Freeze the three expected plant frames for one held-out template."""

    return MappingProxyType(
        {
            "scope-threshold": scope_threshold,
            "authority-conflict": authority_conflict,
            "uncovered-exception": uncovered_exception,
        }
    )


ADDITIONAL_HELDOUT_DECISION_FRAMES: Final[Mapping[str, TemplateDecisionFrames]] = (
    MappingProxyType(
        {
            "community-clinic-scheduling": _template(
                scope_threshold=_frame(
                    actor="clinic schedulers",
                    action="determine early appointment eligibility",
                    condition="a patient's condition may warrant priority scheduling",
                    anchor_outcome="priority patients are offered an early appointment",
                    alternative_outcome=(
                        "schedulers define which conditions qualify for an early appointment"
                    ),
                ),
                authority_conflict=_frame(
                    actor="telehealth schedulers",
                    action="book an urgent telehealth visit",
                    condition="identity confirmation is not yet complete",
                    anchor_outcome="identity is confirmed before the visit is booked",
                    alternative_outcome=(
                        "an urgent telehealth slot is booked before identity confirmation"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="clinic staff",
                    action="deliver preparation instructions",
                    condition="a patient cannot access the patient portal",
                    anchor_outcome="preparation instructions are sent through the portal",
                    alternative_outcome="staff send the instructions through another channel",
                ),
            ),
            "cooperative-housing-maintenance": _template(
                scope_threshold=_frame(
                    actor="maintenance coordinators",
                    action="classify urgent repairs and set response times",
                    condition="a repair may require prompt handling",
                    anchor_outcome="urgent repairs are handled promptly",
                    alternative_outcome=(
                        "coordinators define which repairs are urgent and how quickly to respond"
                    ),
                ),
                authority_conflict=_frame(
                    actor="emergency repair crews",
                    action="enter an occupied unit",
                    condition="active damage must be stopped immediately",
                    anchor_outcome="a manager approves entry before the crew enters",
                    alternative_outcome=(
                        "the emergency crew enters immediately without manager approval"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="housing staff",
                    action="deliver repair notices",
                    condition="a resident cannot access the resident application",
                    anchor_outcome="repair notices are posted in the resident application",
                    alternative_outcome="the resident receives the notice another way",
                ),
            ),
            "library-digitization-rules": _template(
                scope_threshold=_frame(
                    actor="archivists",
                    action="classify sensitive materials and set access restrictions",
                    condition="digitized material may be culturally sensitive",
                    anchor_outcome="sensitive materials receive appropriate access restrictions",
                    alternative_outcome=(
                        "archivists define sensitivity and choose the applicable restrictions"
                    ),
                ),
                authority_conflict=_frame(
                    actor="library access staff",
                    action="provide preservation scan images to researchers",
                    condition="a researcher requests full-resolution preservation scans",
                    anchor_outcome="researchers may download full-resolution preservation scans",
                    alternative_outcome="researchers receive excerpt images only",
                ),
                uncovered_exception=_frame(
                    actor="library staff",
                    action="accept a takedown request",
                    condition="the requester cannot access a catalog account",
                    anchor_outcome="the request is submitted through the catalog account",
                    alternative_outcome="the requester uses another takedown route",
                ),
            ),
            "food-bank-distribution": _template(
                scope_threshold=_frame(
                    actor="intake volunteers",
                    action="assess household need and assign additional staples",
                    condition="a household may have high need",
                    anchor_outcome="high-need households receive additional staple items",
                    alternative_outcome=(
                        "volunteers define high need and decide how many staples to add"
                    ),
                ),
                authority_conflict=_frame(
                    actor="food distribution staff",
                    action="reassign an unclaimed reserved food box",
                    condition="thirty minutes have passed since scheduled pickup",
                    anchor_outcome=(
                        "the reserved box remains assigned until the distribution window closes"
                    ),
                    alternative_outcome=(
                        "staff reassign the unclaimed box after thirty minutes"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="food bank intake staff",
                    action="allow a proxy to collect food",
                    condition="the household identification card is unavailable",
                    anchor_outcome="the proxy presents the household identification card",
                    alternative_outcome="the proxy collects food without the identification card",
                ),
            ),
            "renewable-microgrid-operations": _template(
                scope_threshold=_frame(
                    actor="microgrid operators",
                    action="set the overnight battery reserve level",
                    condition="the microgrid must maintain an adequate overnight reserve",
                    anchor_outcome="operators maintain an adequate battery reserve overnight",
                    alternative_outcome=(
                        "operators define the battery level that counts as adequate"
                    ),
                ),
                authority_conflict=_frame(
                    actor="peak-response operators",
                    action="discharge the battery below its safety reserve",
                    condition="a feeder trip must be prevented",
                    anchor_outcome="battery discharge below the safety reserve is prohibited",
                    alternative_outcome=(
                        "operators discharge below the reserve to prevent the feeder trip"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="microgrid operators",
                    action="enter island mode",
                    condition="the site lead cannot be reached",
                    anchor_outcome="operators contact the site lead before entering island mode",
                    alternative_outcome="operators enter island mode without reaching the site lead",
                ),
            ),
            "museum-loan-handling": _template(
                scope_threshold=_frame(
                    actor="museum registrars",
                    action="classify loan damage for immediate conservation review",
                    condition="a loan arrives with possible significant damage",
                    anchor_outcome="significantly damaged loans receive immediate review",
                    alternative_outcome=(
                        "registrars define which damage is significant enough for immediate review"
                    ),
                ),
                authority_conflict=_frame(
                    actor="approved couriers",
                    action="transfer a delayed museum loan",
                    condition="the assigned courier is delayed",
                    anchor_outcome="only the assigned registrar authorizes the transfer",
                    alternative_outcome=(
                        "a courier transfers the delayed loan to another approved courier"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="museum loan staff",
                    action="select a return route for the borrowed object",
                    condition="the designated border crossing is closed",
                    anchor_outcome="the borrower uses the designated border crossing",
                    alternative_outcome="the borrower returns the object by another route",
                ),
            ),
            "open-source-grant-program": _template(
                scope_threshold=_frame(
                    actor="grant reviewers",
                    action="assess community benefit for priority review",
                    condition="an open-source project claims substantial community benefit",
                    anchor_outcome="projects with substantial community benefit receive priority",
                    alternative_outcome=(
                        "reviewers define what counts as substantial community benefit"
                    ),
                ),
                authority_conflict=_frame(
                    actor="the grant steering group",
                    action="ratify an essential expense after purchase",
                    condition="the expense was not approved before purchase",
                    anchor_outcome="only preapproved expenses are reimbursed",
                    alternative_outcome=(
                        "the steering group retrospectively ratifies and reimburses the expense"
                    ),
                ),
                uncovered_exception=_frame(
                    actor="another project maintainer",
                    action="sign the final grant report",
                    condition="the lead maintainer is unavailable",
                    anchor_outcome="the lead maintainer signs the final report",
                    alternative_outcome="another maintainer signs the final report",
                ),
            ),
            "agricultural-water-allocation": _template(
                scope_threshold=_frame(
                    actor="water allocation coordinators",
                    action="determine whether measurements show a severe shortage",
                    condition="irrigation supply measurements indicate a possible shortage",
                    anchor_outcome="severe shortages trigger reduced irrigation allocations",
                    alternative_outcome=(
                        "coordinators define the measurements that constitute a severe shortage"
                    ),
                ),
                authority_conflict=_frame(
                    actor="the emergency water desk",
                    action="adjust irrigation allocations daily",
                    condition="a canal failure disrupts the published weekly schedule",
                    anchor_outcome="weekly allocations remain fixed until the next schedule",
                    alternative_outcome="the emergency desk adjusts allocations each day",
                ),
                uncovered_exception=_frame(
                    actor="water allocation coordinators",
                    action="allocate field water",
                    condition="the upstream flow sensor is offline",
                    anchor_outcome="field allocations rely on the upstream flow sensor",
                    alternative_outcome="coordinators allocate water without the sensor reading",
                ),
            ),
            "disaster-shelter-intake": _template(
                scope_threshold=_frame(
                    actor="shelter intake workers",
                    action="assess vulnerability for bed priority",
                    condition="a resident seeks priority for an available shelter bed",
                    anchor_outcome="vulnerable residents receive priority for available beds",
                    alternative_outcome=(
                        "intake workers define which circumstances make a resident vulnerable"
                    ),
                ),
                authority_conflict=_frame(
                    actor="crisis intake staff",
                    action="reserve a shelter bed",
                    condition="a resident lacks identification during crisis intake",
                    anchor_outcome="a bed is held only after identification is presented",
                    alternative_outcome="crisis intake reserves a bed before identification",
                ),
                uncovered_exception=_frame(
                    actor="shelter assignment staff",
                    action="reunite family members in bed assignments",
                    condition="family members arrive separately after assignments are complete",
                    anchor_outcome="family members are assigned together when they arrive together",
                    alternative_outcome="staff reunite separately arriving family members",
                ),
            ),
            "translation-quality-service": _template(
                scope_threshold=_frame(
                    actor="translation coordinators",
                    action="classify sensitive content and select a senior reviewer",
                    condition="a translation request may contain sensitive content",
                    anchor_outcome="sensitive content receives a senior language review",
                    alternative_outcome=(
                        "coordinators define sensitivity and the required reviewer seniority"
                    ),
                ),
                authority_conflict=_frame(
                    actor="translation publication staff",
                    action="publish an urgent safety notice",
                    condition="senior review is not yet complete",
                    anchor_outcome="no translation is published before senior review is complete",
                    alternative_outcome="the urgent safety notice is published before senior review",
                ),
                uncovered_exception=_frame(
                    actor="translation coordinators",
                    action="route an unlisted language request",
                    condition="the requested language is absent from the reviewer roster",
                    anchor_outcome="requests are assigned to a language on the reviewer roster",
                    alternative_outcome="the request is routed outside the reviewer roster",
                ),
            ),
            "research-computing-queue": _template(
                scope_threshold=_frame(
                    actor="compute queue operators",
                    action="classify a research run for expedited placement",
                    condition="a run may have high research impact",
                    anchor_outcome="high-impact runs receive expedited queue placement",
                    alternative_outcome="operators define which research runs are high impact",
                ),
                authority_conflict=_frame(
                    actor="the night compute operator",
                    action="transfer idle compute capacity between projects",
                    condition="another project is delayed while capacity is idle",
                    anchor_outcome="compute allocations are not transferred between projects",
                    alternative_outcome="the night operator transfers idle capacity",
                ),
                uncovered_exception=_frame(
                    actor="compute queue operators",
                    action="make an emergency queue change",
                    condition="the principal investigator is unreachable",
                    anchor_outcome="the principal investigator confirms every emergency change",
                    alternative_outcome="the emergency change proceeds without confirmation",
                ),
            ),
            "makerspace-tool-safety": _template(
                scope_threshold=_frame(
                    actor="makerspace floor supervisors",
                    action="classify dangerous tools for advanced orientation",
                    condition="a tool may pose enough risk to require extra training",
                    anchor_outcome="dangerous tools require advanced orientation",
                    alternative_outcome=(
                        "floor supervisors define which tools require advanced orientation"
                    ),
                ),
                authority_conflict=_frame(
                    actor="event tool supervisors",
                    action="permit a guest to use a powered tool",
                    condition="the guest has direct supervision but no completed orientation",
                    anchor_outcome="guests complete orientation before using powered tools",
                    alternative_outcome="a directly supervised event guest uses the powered tool",
                ),
                uncovered_exception=_frame(
                    actor="makerspace supervisors",
                    action="stop powered-tool use",
                    condition="the designated supervisor must leave unexpectedly",
                    anchor_outcome="the designated supervisor remains present throughout tool use",
                    alternative_outcome="powered-tool use stops when the supervisor leaves",
                ),
            ),
            "transit-accessibility-service": _template(
                scope_threshold=_frame(
                    actor="transit operators",
                    action="determine reasonable boarding assistance",
                    condition="a rider requests boarding assistance",
                    anchor_outcome="operators provide reasonable boarding assistance",
                    alternative_outcome=(
                        "operators define which assistance is reasonable for the situation"
                    ),
                ),
                authority_conflict=_frame(
                    actor="transit operators",
                    action="board a mobility-device user without a reservation",
                    condition="safe space is available when the rider is waiting",
                    anchor_outcome="large mobility devices require a boarding reservation",
                    alternative_outcome="the waiting rider boards when safe space is available",
                ),
                uncovered_exception=_frame(
                    actor="transit operators",
                    action="provide step-free boarding",
                    condition="the station lift is out of service",
                    anchor_outcome="riders use the station lift for step-free boarding",
                    alternative_outcome="operators provide another boarding method",
                ),
            ),
            "fisheries-cooperative-landing": _template(
                scope_threshold=_frame(
                    actor="vessel skippers",
                    action="decide whether poor weather requires an early return",
                    condition="observed weather may make continued fishing unsafe",
                    anchor_outcome="vessels return early when weather becomes poor",
                    alternative_outcome=(
                        "skippers define the weather conditions that require an early return"
                    ),
                ),
                authority_conflict=_frame(
                    actor="vessel skippers",
                    action="transfer a catch at sea",
                    condition="a delay threatens preservation of the catch",
                    anchor_outcome="only the dock officer authorizes a transfer between vessels",
                    alternative_outcome="a skipper transfers the catch at sea to preserve it",
                ),
                uncovered_exception=_frame(
                    actor="vessel skippers",
                    action="change the vessel's landing destination",
                    condition="communications with the landing desk are lost",
                    anchor_outcome="the vessel contacts the landing desk before changing destination",
                    alternative_outcome="the vessel changes destination without desk contact",
                ),
            ),
            "regional-archives-access": _template(
                scope_threshold=_frame(
                    actor="archivists",
                    action="classify sensitive files and set reading-room restrictions",
                    condition="an archival file may contain sensitive material",
                    anchor_outcome="sensitive files receive suitable reading-room restrictions",
                    alternative_outcome=(
                        "archivists define sensitivity and choose suitable restrictions"
                    ),
                ),
                authority_conflict=_frame(
                    actor="reading-room staff",
                    action="allow a researcher to photograph cited pages",
                    condition="the researcher guide allows photographs in the restricted room",
                    anchor_outcome="researchers may not make copies in the restricted reading room",
                    alternative_outcome="researchers take phone photographs of cited pages",
                ),
                uncovered_exception=_frame(
                    actor="archives permission staff",
                    action="select an authority to decide a permission request",
                    condition="the record's author has died",
                    anchor_outcome="the permission request goes to the record's living author",
                    alternative_outcome="another authority decides the permission request",
                ),
            ),
            "rural-broadband-outages": _template(
                scope_threshold=_frame(
                    actor="broadband dispatchers",
                    action="classify a widespread outage for accelerated response",
                    condition="multiple customer connections are affected",
                    anchor_outcome="widespread outages receive an accelerated field response",
                    alternative_outcome=(
                        "dispatchers define how many affected connections count as widespread"
                    ),
                ),
                authority_conflict=_frame(
                    actor="the overnight duty lead",
                    action="dispatch a field engineer",
                    condition="an outage occurs overnight",
                    anchor_outcome="only the network operations center dispatches an engineer",
                    alternative_outcome="the duty lead dispatches an engineer directly",
                ),
                uncovered_exception=_frame(
                    actor="broadband support staff",
                    action="deliver restoration instructions",
                    condition="a disconnected customer cannot rely on the normal text message",
                    anchor_outcome="customers receive restoration instructions by text message",
                    alternative_outcome="the customer receives instructions through another channel",
                ),
            ),
            "school-meal-accommodations": _template(
                scope_threshold=_frame(
                    actor="school meal coordinators",
                    action="classify dietary needs for a customized meal plan",
                    condition="a student's dietary needs may be exceptional",
                    anchor_outcome="exceptional dietary needs receive a customized meal plan",
                    alternative_outcome=(
                        "meal coordinators define which dietary needs are exceptional"
                    ),
                ),
                authority_conflict=_frame(
                    actor="school nurses",
                    action="approve a dietary accommodation",
                    condition="the monthly menu has already been finalized",
                    anchor_outcome="accommodation forms arrive before menu finalization",
                    alternative_outcome="a school nurse approves the accommodation afterward",
                ),
                uncovered_exception=_frame(
                    actor="school meal staff",
                    action="make a safety-related meal-plan change",
                    condition="the student's guardian cannot be reached",
                    anchor_outcome="a guardian confirms every meal-plan change",
                    alternative_outcome="the safety-related change proceeds without confirmation",
                ),
            ),
            "water-quality-laboratory": _template(
                scope_threshold=_frame(
                    actor="laboratory analysts",
                    action="classify an anomalous reading for additional review",
                    condition="a reading deviates from expected measurements",
                    anchor_outcome="anomalous readings receive additional laboratory review",
                    alternative_outcome=(
                        "analysts define the deviation that makes a reading anomalous"
                    ),
                ),
                authority_conflict=_frame(
                    actor="laboratory incident staff",
                    action="release a potential contamination result",
                    condition="a duplicate test has not yet confirmed the result",
                    anchor_outcome="results are released only after duplicate-test confirmation",
                    alternative_outcome="the potential contamination result is released immediately",
                ),
                uncovered_exception=_frame(
                    actor="laboratory release staff",
                    action="withhold a result",
                    condition="the current instrument calibration record is missing",
                    anchor_outcome="every result cites the current calibration record",
                    alternative_outcome="the result is withheld until the record is available",
                ),
            ),
        }
    )
)


def decision_frame_for(template_id: str, plant_id: str) -> DecisionFrame:
    """Return the immutable authored frame for one additional held-out plant."""

    try:
        return ADDITIONAL_HELDOUT_DECISION_FRAMES[template_id][plant_id]
    except KeyError as error:
        raise KeyError(
            f"no additional held-out decision frame for {template_id!r}/{plant_id!r}"
        ) from error
