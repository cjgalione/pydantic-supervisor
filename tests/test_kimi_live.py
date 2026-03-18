"""Live integration test for Kimi model via deployed interactive endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
import urllib.error
import urllib.request
import httpx

KIMI_MODEL_NAME = "moonshotai/Kimi-K2.5"


class KimiLiveIntegrationTest(unittest.TestCase):
    def test_kimi_request_succeeds_via_interactive_endpoint(self) -> None:
        run_live = (os.getenv("RUN_LIVE_KIMI_TEST") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not run_live:
            self.skipTest("Set RUN_LIVE_KIMI_TEST=1 to run live Kimi integration test.")

        base_url = (os.getenv("PYDANTIC_SUPERVISOR_INTERACTIVE_URL") or "").strip().rstrip("/")
        if not base_url:
            self.skipTest(
                "Set PYDANTIC_SUPERVISOR_INTERACTIVE_URL to the deployed interactive base URL."
            )

        payload = {
            "query": "Compute 12*9 and return only the number.",
            "workflow_name": "kimi-live-test",
            "supervisor_model": KIMI_MODEL_NAME,
            "research_model": KIMI_MODEL_NAME,
            "math_model": KIMI_MODEL_NAME,
        }
        request = urllib.request.Request(
            url=f"{base_url}/interactive/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )

        status_code: int | None = None
        raw_body = ""
        urlopen_error: Exception | None = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status_code = response.status
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            self.fail(f"HTTP {exc.code} from interactive endpoint: {error_body}")
        except urllib.error.URLError as exc:
            urlopen_error = exc

        if status_code is None:
            httpx_error: Exception | None = None
            try:
                with httpx.Client(timeout=120) as client:
                    response = client.post(
                        f"{base_url}/interactive/query",
                        headers={"content-type": "application/json"},
                        json=payload,
                    )
                status_code = int(response.status_code)
                raw_body = response.text
            except Exception as fallback_exc:
                httpx_error = fallback_exc

            if status_code is None:
                curl_cmd = [
                    "curl",
                    "-sS",
                    "-w",
                    "\n%{http_code}",
                    "-X",
                    "POST",
                    f"{base_url}/interactive/query",
                    "-H",
                    "content-type: application/json",
                    "-d",
                    json.dumps(payload),
                ]
                try:
                    curl_proc = subprocess.run(
                        curl_cmd,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=180,
                    )
                except Exception as curl_exc:
                    self.fail(
                        "Failed to reach interactive endpoint with urllib/httpx/curl. "
                        f"urllib error: {urlopen_error!r}; httpx error: {httpx_error!r}; "
                        f"curl error: {curl_exc!r}"
                    )

                if curl_proc.returncode != 0:
                    if curl_proc.returncode == 6:
                        self.skipTest(
                            "DNS resolution for interactive endpoint is unavailable in this environment."
                        )
                    self.fail(
                        "curl request to interactive endpoint failed. "
                        f"returncode={curl_proc.returncode}, stderr={curl_proc.stderr!r}"
                    )
                stdout = curl_proc.stdout
                if "\n" not in stdout:
                    self.fail(f"Unexpected curl output format: {stdout!r}")
                raw_body, status_text = stdout.rsplit("\n", 1)
                status_code = int(status_text.strip())

        self.assertEqual(status_code, 200, raw_body)
        parsed = json.loads(raw_body)
        self.assertEqual(
            parsed.get("resolved_models", {}).get("supervisor_model"),
            KIMI_MODEL_NAME,
            raw_body,
        )
        final_output = str(parsed.get("final_output", "")).strip()
        self.assertTrue(final_output, f"Expected non-empty final_output. Response: {raw_body}")


if __name__ == "__main__":
    unittest.main()
