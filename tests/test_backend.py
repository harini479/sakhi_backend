"""Automated verification tests for Sakhi Python Backend (Cloud Run readiness & security)."""

import asyncio
import json
import os
import re
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import auth, chat_engine, config, firebase_client, guardrails, llm, store
import dev_server


class TestCloudRunConfigAndImports(unittest.TestCase):
    def test_default_settings_and_production_cors_wildcard_blocked(self):
        old_env = dict(os.environ)
        try:
            os.environ["APP_ENV"] = "production"
            os.environ["CORS_ORIGINS"] = "*, https://sakhi-frontend.run.app/"
            os.environ["GEMINI_MODEL"] = "gemini-4-hydrogen"
            settings = config.load_settings()
            self.assertTrue(settings.is_production)
            self.assertTrue(settings.require_firebase_auth)
            self.assertEqual(settings.cors_origins, ["https://sakhi-frontend.run.app"])
            self.assertEqual(settings.gemini_model, "gemini-4-hydrogen")
            self.assertEqual(settings.port, 8080)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_firebase_status_reports_adc_mode(self):
        status = firebase_client.get_firebase_status()
        self.assertEqual(status["authMode"], "application_default_credentials")
        self.assertIn("initialized", status)

    def test_curated_knowledge_and_patients_loaded(self):
        self.assertGreaterEqual(len(store.medical_knowledge), 50)
        self.assertEqual(len(store.clinical_rules), 24)
        self.assertEqual(store.get_patient_by_phone("+918888888882")["name"], "Ananya Reddy")
        preview = store.get_predefined_care_preview()
        self.assertGreaterEqual(len(preview["pre_care"]["topics"]), 8)
        self.assertGreaterEqual(len(preview["post_care"]["patients"]), 4)


class TestMedicalSafetyAndEscalations(unittest.TestCase):
    def test_out_of_scope_guardrail(self):
        self.assertIsNotNone(guardrails.get_redirect_for_out_of_scope("Who won the IPL cricket match?"))
        self.assertIsNone(guardrails.get_redirect_for_out_of_scope("What is IVF?"))

    def test_missing_data_escalates_to_verified_doctor(self):
        res = asyncio.run(
            chat_engine.process_sakhi_chat(
                phone_number="+911111111111",
                message="Can I take ashwagandha with warfarin for lupus?",
                language="en",
            )
        )
        self.assertEqual(res["mode"], "doctor_escalation")
        self.assertIn("verified doctor", res["reply"].lower())

    def test_emergency_instructs_contact_caretaker_staff(self):
        res = asyncio.run(
            chat_engine.process_sakhi_chat(
                phone_number="+918888888881",
                message="I am having chest pain",
                language="en",
            )
        )
        self.assertEqual(res["mode"], "emergency")
        self.assertIn("caretaker staff", res["reply"].lower())

    def test_missed_ivf_injection_triggers_emergency(self):
        res = asyncio.run(
            chat_engine.process_sakhi_chat(
                phone_number="+918888888882",
                message="I missed my stimulation injection dose today",
                language="en",
            )
        )
        self.assertEqual(res["mode"], "emergency")
        self.assertIn("caretaker staff", res["reply"].lower())


class TestFirebaseAuthEnforcement(unittest.TestCase):
    def test_production_requires_bearer_token(self):
        old_env = dict(os.environ)
        try:
            os.environ["APP_ENV"] = "production"
            with self.assertRaises(auth.AuthError) as ctx:
                auth.enforce_user_authorization("uid-123", None)
            self.assertEqual(ctx.exception.status_code, 401)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_cross_user_access_forbidden(self):
        fake_verifier = lambda _tok: {"uid": "uid-alice", "email": "alice@example.com"}
        with self.assertRaises(auth.AuthError) as ctx:
            auth.enforce_user_authorization(
                "uid-bob",
                "Bearer valid-token",
                token_verifier=fake_verifier,
            )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_same_user_token_allowed(self):
        fake_verifier = lambda _tok: {"uid": "uid-alice", "email": "alice@example.com"}
        identity = auth.enforce_user_authorization(
            "uid-alice",
            "Bearer valid-token",
            token_verifier=fake_verifier,
        )
        self.assertIsNotNone(identity)
        self.assertEqual(identity.uid, "uid-alice")

    def test_webhook_secret_enforced_when_configured(self):
        old_env = dict(os.environ)
        try:
            os.environ["WEBHOOK_SECRET"] = "test-secret-value"
            with self.assertRaises(auth.AuthError) as ctx:
                auth.verify_webhook_secret("wrong-secret")
            self.assertEqual(ctx.exception.status_code, 401)
            auth.verify_webhook_secret("test-secret-value")
        finally:
            os.environ.clear()
            os.environ.update(old_env)


class TestHttpServerOnPort8080(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 8080), dev_server.SakhiRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str):
        with urlopen(f"http://127.0.0.1:8080{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, body: dict, headers: dict | None = None):
        req_headers = {"Content-Type": "application/json", **(headers or {})}
        req = Request(
            f"http://127.0.0.1:8080{path}",
            data=json.dumps(body).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def test_health_endpoint_responds_without_gemini(self):
        status, data = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data, {"status": "ok", "service": "backend-service"})

    def test_api_health_and_care_preview(self):
        status, data = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("firebase", data)
        self.assertIn("providers", data)

        p_status, preview = self._get("/api/care-preview")
        self.assertEqual(p_status, 200)
        self.assertIn("pre_care", preview)

    def test_sensitive_route_blocks_unauthenticated_in_production(self):
        old_env = dict(os.environ)
        try:
            os.environ["APP_ENV"] = "production"
            status, data = self._post(
                "/user/relation",
                {"user_id": "u-venkatesh", "relation": "partner"},
            )
            self.assertEqual(status, 401)
            self.assertIn("detail", data)
        finally:
            os.environ.clear()
            os.environ.update(old_env)


class TestDeploymentHygiene(unittest.TestCase):
    def test_no_hardcoded_secrets_in_deployment_files(self):
        secret_patterns = [
            re.compile(r"sk-proj-[A-Za-z0-9_-]{10,}"),
            re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"),
            re.compile(r"-----BEGIN PRIVATE KEY-----"),
            re.compile(r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[A-Za-z0-9_-]{10,}"),
        ]
        for file_path in BACKEND_ROOT.rglob("*"):
            if not file_path.is_file() or "__pycache__" in file_path.parts:
                continue
            if file_path.name == "test_backend.py":
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pat in secret_patterns:
                self.assertIsNone(
                    pat.search(content),
                    f"Potential hardcoded secret detected in {file_path}",
                )

    def test_dockerfile_valid_cloud_run_directives(self):
        dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.11-slim", dockerfile)
        self.assertIn("USER appuser", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("uvicorn main:app", dockerfile)
        self.assertIn("0.0.0.0", dockerfile)
        self.assertIn("${PORT:-8080}", dockerfile)


if __name__ == "__main__":
    unittest.main()
