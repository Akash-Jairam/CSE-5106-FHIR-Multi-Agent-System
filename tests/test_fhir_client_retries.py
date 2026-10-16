"""FHIR reads retry dropped connections and throttling; nothing else is retried."""
import unittest
from unittest.mock import Mock

import requests

from fhir_client import FHIRClient, FHIR_GET_RETRY


class FhirReadRetryTests(unittest.TestCase):
    def test_session_gets_a_get_only_retrying_adapter(self):
        client = FHIRClient("p", "l", "d", "s", creds=requests.Session())
        retry = client.session.get_adapter("https://healthcare.googleapis.com/v1").max_retries
        self.assertIs(retry, FHIR_GET_RETRY)
        self.assertEqual(retry.allowed_methods, frozenset({"GET"}))
        self.assertGreater(retry.connect, 0)
        self.assertIn(503, retry.status_forcelist)

    def test_session_without_adapters_is_left_alone(self):
        session = Mock(spec=["get"])
        self.assertIs(FHIRClient("p", "l", "d", "s", creds=session).session, session)


if __name__ == "__main__":
    unittest.main()
