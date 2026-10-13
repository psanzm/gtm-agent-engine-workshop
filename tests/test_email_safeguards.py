import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "LEAD-12853",
            "name": "Omar Okafor",
            "email": "omar.okafor@lakesideanalytics.com",
        }
        self.from_rep = {"name": "Ava Mills", "email": "rep@example.com"}

    def send(self, prospect=None):
        return send_prospect_email.func(
            prospect or self.prospect, "Subject", "Body", self.runtime, self.from_rep
        )

    def test_disqualified_prospect_is_blocked_without_sending(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Priya Nair",
            "email": "priya.nair@brightwaveapps.com",
        }
        with patch.object(agent_module.uuid, "uuid4") as make_uuid:
            result = self.send(prospect)

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        make_uuid.assert_not_called()

    def test_eligible_prospect_is_sent(self):
        result = self.send()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.prospect["email"])

    def test_missing_source_record_preserves_send_behavior(self):
        prospect = {**self.prospect, "prospect_id": "LEAD-MISSING"}
        result = self.send(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.prospect["email"])


class RunAgentEmailTests(unittest.TestCase):
    def test_blocked_email_is_reported_without_claiming_success(self):
        messages = [
            ToolMessage(
                content=json.dumps(
                    {"status": "blocked", "error": "Prospect is disqualified; email not sent."}
                ),
                name="send_prospect_email",
                tool_call_id="send-1",
            ),
            AIMessage(content="Email sent successfully."),
        ]
        with patch.object(agent_module.gtm_agent, "invoke", return_value={"messages": messages}):
            result = run_agent("Email LEAD-50001", user_id="rep_amills")

        self.assertIn("disqualified", result["reply"])
        self.assertIn("did not send", result["reply"])
        self.assertNotIn("sent successfully", result["reply"])
        self.assertEqual(result["messages"][-1].content, result["reply"])


if __name__ == "__main__":
    unittest.main()
