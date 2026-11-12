"""The order the nodes run in, the Clearance Controller's gate, the branch and the checkpoint.

    load_campaign_context -> plan_assessment -> assess_copy_risk -> identify_issues
      -> check_spot_schedule -> compose_clearance_assessment -> request_controller_decision (gate)
    after the gate, by the decision:
      CLEAR                   -> draft_clearance_certificate -> record_clearance_outcome
      CLEAR WITH RESTRICTIONS -> draft_restriction_notice -> draft_clearance_certificate -> record
      REJECT                  -> record_clearance_outcome
    record_clearance_outcome -> END

The SQLite checkpointer saves the state after every node, so a run paused at the gate can be
resumed by any process that opens the same file, even after the machine was switched off.
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic.nodes import (
    assess_copy_risk_node,
    check_spot_schedule_node,
    compose_clearance_assessment_node,
    draft_clearance_certificate_node,
    draft_restriction_notice_node,
    identify_issues_node,
    load_campaign_context_node,
    plan_assessment_node,
    record_clearance_outcome_node,
    request_controller_decision_node,
    route_controller_decision,
)
from agentic.state import ClearanceAssessmentState
from config import config

ASSESSING_STEPS = [
    ("load_campaign_context", load_campaign_context_node),
    ("plan_assessment", plan_assessment_node),
    ("assess_copy_risk", assess_copy_risk_node),
    ("identify_issues", identify_issues_node),
    ("check_spot_schedule", check_spot_schedule_node),
    ("compose_clearance_assessment", compose_clearance_assessment_node),
    ("request_controller_decision", request_controller_decision_node),
]

DECISION_PATHS = {
    config.DECISION_CLEAR: "draft_clearance_certificate",
    config.DECISION_CLEAR_WITH_RESTRICTIONS: "draft_restriction_notice",
    config.DECISION_REJECT: "record_clearance_outcome",
}


def build_workflow(checkpoint_path: str) -> CompiledStateGraph:
    """Assemble the ten nodes into a workflow that can be run, paused at the gate and resumed."""
    workflow = StateGraph(ClearanceAssessmentState)
    for node_name, node in ASSESSING_STEPS:
        workflow.add_node(node_name, node)
    workflow.add_node("draft_restriction_notice", draft_restriction_notice_node)
    workflow.add_node("draft_clearance_certificate", draft_clearance_certificate_node)
    workflow.add_node("record_clearance_outcome", record_clearance_outcome_node)

    workflow.add_edge(START, ASSESSING_STEPS[0][0])
    for (earlier, _), (later, _) in zip(ASSESSING_STEPS, ASSESSING_STEPS[1:]):
        workflow.add_edge(earlier, later)
    workflow.add_conditional_edges(
        "request_controller_decision", route_controller_decision, DECISION_PATHS
    )
    workflow.add_edge("draft_restriction_notice", "draft_clearance_certificate")
    workflow.add_edge("draft_clearance_certificate", "record_clearance_outcome")
    workflow.add_edge("record_clearance_outcome", END)

    folder = os.path.dirname(checkpoint_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(connection))
