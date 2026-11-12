from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.math_verification.main import app


class ApiTests(unittest.TestCase):
    def test_health_does_not_require_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            response = TestClient(app).get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["provider_configured"])

    def test_rlc_example_is_concrete(self) -> None:
        response = TestClient(app).get("/api/examples/rlc")
        self.assertEqual(response.status_code, 200)
        payload = response.json()["request"]
        self.assertEqual(payload["discipline"], "Electrical Engineering")
        self.assertIn("L*d2q/dt2", payload["problem"])
        self.assertIn("100 microfarad", payload["problem"])

    def test_optional_backend_key_is_enforced(self) -> None:
        with patch.dict(os.environ, {"BACKEND_API_KEY": "test-secret"}, clear=True):
            unauthorized = TestClient(app).get("/api/examples/rlc")
            authorized = TestClient(app).get(
                "/api/examples/rlc", headers={"X-API-Key": "test-secret"}
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)


if __name__ == "__main__":
    unittest.main()
