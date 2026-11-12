"""What the assessment workflow carries from one node to the next.

LangGraph checkpoints this dictionary after every node, which is what lets a run paused at the
Clearance Controller's gate survive a restart. Every key is replaced by whichever node writes it,
except llm_call_count, which adds up: each agent node returns 1 and the reducer sums them.
"""

import operator
from typing import Annotated, TypedDict


class ClearanceAssessmentState(TypedDict):
    """The workflow's state: the request, the facts the tools found and the agents' replies."""

    campaign_reference: str
    executive_request: str
    executive_id: str
    thread_id: str
    run_id: str
    assessment_id: str
    campaign_profile: dict
    campaign_spots: list
    campaign_notes: list
    amendment_facts: dict
    issue_facts: dict
    schedule_facts: dict
    assessment_plan: dict
    copy_assessment: dict
    issue_analysis: dict
    schedule_advice: dict
    controller_decision: dict
    clearance_outcome: dict
    restriction_notice: dict
    clearance_certificate: dict
    llm_call_count: Annotated[int, operator.add]
