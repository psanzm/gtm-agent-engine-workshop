import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import gtm_agent
from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_allowed_contact_fields(self):
        source = data_service.get_prospect_record("LEAD-15229")

        result = gtm_agent.get_prospect.func("LEAD-15229")

        expected_contact = {
            "prospect_id": "LEAD-15229",
            **{
                field: source[field]
                for field in ("name", "email", "annual_revenue", "disqualified", "enrichment_source")
            },
        }
        self.assertEqual(result["prospect"], expected_contact)
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_profile_cache_miss_saves_only_allowed_fields(self):
        source = {
            "name": "Example Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 12000000,
            "billing_qualification": {"tax_id": "private"},
        }
        expected_profile = {
            "prospect_id": "LEAD-15229",
            "annual_revenue": 12000000,
            "engagement_history": ["discovery call"],
            "account_details": {"industry": "software"},
            "tech_stack": ["CRM"],
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=source),
            patch.object(data_service, "fetch_engagement_history", return_value=expected_profile["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=expected_profile["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=expected_profile["tech_stack"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-15229")

        self.assertEqual(result["prospect_profile"], expected_profile)
        self.assertEqual(save_profile.call_args.args, ("LEAD-15229", expected_profile))
        self.assertFalse({"name", "email", "billing_qualification"} & set(save_profile.call_args.args[1]))

    def test_cached_profile_is_sanitized_before_return_and_save(self):
        cached_profile = {
            "prospect_id": "LEAD-15229",
            "name": "Example Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 12000000,
            "billing_qualification": {"card_on_file": "private"},
            "account_details": {"industry": "software"},
            "tech_stack": ["CRM"],
            "unexpected_field": "private",
        }
        expected_profile = {
            "prospect_id": "LEAD-15229",
            "annual_revenue": 12000000,
            "account_details": {"industry": "software"},
            "tech_stack": ["CRM"],
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": cached_profile}),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-15229")

        self.assertEqual(result["prospect_profile"], expected_profile)
        self.assertEqual(save_profile.call_args.args, ("LEAD-15229", expected_profile))

    def test_profile_cache_reader_sanitizes_before_return(self):
        original = data_service._PROFILES.get("LEAD-15229")
        cached_profile = {
            "prospect_id": "LEAD-15229",
            "email": "prospect@example.com",
            "billing_qualification": {"tax_id": "private"},
            "annual_revenue": 12000000,
            "tech_stack": ["CRM"],
        }
        data_service._PROFILES["LEAD-15229"] = cached_profile
        try:
            result = data_service.get_profile_from_db.__wrapped__("LEAD-15229")
        finally:
            if original is None:
                data_service._PROFILES.pop("LEAD-15229", None)
            else:
                data_service._PROFILES["LEAD-15229"] = original

        self.assertEqual(
            result["prospect_profile"],
            {
                "prospect_id": "LEAD-15229",
                "annual_revenue": 12000000,
                "tech_stack": ["CRM"],
            },
        )

    def test_scoring_message_contains_only_scoring_inputs(self):
        response = Mock()
        response.model_dump.return_value = {"score": 80}
        profile = {
            "prospect_id": "LEAD-15229",
            "name": "Example Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 12000000,
            "billing_qualification": {"tax_id": "private"},
            "account_details": {"industry": "software"},
            "tech_stack": ["Old CRM"],
            "segment": "mid-market",
            "engagement_history": ["private note"],
        }
        offering = {
            "description": "CRM offering",
            "required_tech_stack": ["CRM"],
            "min_annual_revenue": 1000000,
        }
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(gtm_agent, "_scoring_llm", Mock(invoke=Mock(return_value=response))) as scoring_llm,
        ):
            gtm_agent.score_prospect.func(profile, offering)

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        serialized_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            serialized_profile,
            {
                "annual_revenue": 12000000,
                "tech_stack": ["CRM"],
                "account_details": {"industry": "software"},
                "segment": "mid-market",
            },
        )
        self.assertNotIn("billing_qualification", user_message)
        self.assertNotIn("name", serialized_profile)
        self.assertNotIn("email", serialized_profile)


if __name__ == "__main__":
    unittest.main()
