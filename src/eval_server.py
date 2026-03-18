"""Modal deployment for Braintrust remote eval dev server."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

import modal

CORS_ALLOWED_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Amz-Date",
    "X-Api-Key",
    "X-Amz-Security-Token",
    "x-bt-auth-token",
    "x-bt-parent",
    "x-bt-org-name",
    "x-bt-project-id",
    "x-bt-stream-fmt",
    "x-bt-use-cache",
    "x-bt-use-gateway",
    "x-stainless-os",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-arch",
]

CORS_ALLOWED_ORIGINS: list[str | re.Pattern[str]] = [
    "https://braintrust.dev",
    "https://www.braintrust.dev",
    "https://braintrustdata.com",
    "https://www.braintrustdata.com",
    re.compile(r"https://.*\.braintrust\.dev"),
    re.compile(r"https://.*\.braintrustdata\.com"),
    re.compile(r"https://.*\.preview\.braintrust\.dev"),
]

CORS_EXPOSED_HEADERS = [
    "x-bt-cursor",
    "x-bt-found-existing-experiment",
    "x-bt-span-id",
    "x-bt-span-export",
]


def _is_allowed_origin(origin: str) -> bool:
    if not origin:
        return False
    for allowed in CORS_ALLOWED_ORIGINS:
        if isinstance(allowed, str) and origin == allowed:
            return True
        if isinstance(allowed, re.Pattern) and allowed.match(origin):
            return True
    return False


def _with_playground_cors(inner_app: Any) -> Any:
    """Wrap the Braintrust devserver so browser preflights are handled explicitly."""

    def _with_path_alias(scope: dict[str, Any]) -> dict[str, Any]:
        # Newer playground clients may call /runs; Braintrust devserver still serves /eval.
        if scope.get("type") == "http" and scope.get("path") == "/runs":
            updated_scope = dict(scope)
            updated_scope["path"] = "/eval"
            updated_scope["raw_path"] = b"/eval"
            return updated_scope
        return scope

    async def wrapped(
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        scope = _with_path_alias(scope)
        if scope["type"] != "http":
            await inner_app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode("utf-8")
        if not origin or not _is_allowed_origin(origin):
            await inner_app(scope, receive, send)
            return

        cors_headers = [
            (b"access-control-allow-origin", origin.encode()),
            (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, PATCH"),
            (b"access-control-allow-headers", ", ".join(CORS_ALLOWED_HEADERS).encode()),
            (b"access-control-expose-headers", ", ".join(CORS_EXPOSED_HEADERS).encode()),
            (b"access-control-allow-credentials", b"true"),
            (b"access-control-max-age", b"86400"),
        ]

        if scope["method"] == "OPTIONS":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": cors_headers,
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                header_pairs = list(message.get("headers", []))
                existing = {key.lower() for key, _ in header_pairs}
                for key, value in cors_headers:
                    if key.lower() not in existing:
                        header_pairs.append((key, value))
                message["headers"] = header_pairs
            await send(message)

        await inner_app(scope, receive, send_with_cors)

    return wrapped


def _sync_braintrust_cors(bt_cors: Any) -> None:
    """Keep Braintrust devserver CORS in sync with the wrapper config."""

    def _has_origin(candidate: str | re.Pattern[str]) -> bool:
        for existing in bt_cors.ALLOWED_ORIGINS:
            if isinstance(candidate, str) and isinstance(existing, str) and candidate == existing:
                return True
            if (
                isinstance(candidate, re.Pattern)
                and isinstance(existing, re.Pattern)
                and candidate.pattern == existing.pattern
            ):
                return True
        return False

    for origin in CORS_ALLOWED_ORIGINS:
        if not _has_origin(origin):
            bt_cors.ALLOWED_ORIGINS.append(origin)

    for header in ("x-bt-use-gateway", "x-bt-project-id"):
        if header not in bt_cors.ALLOWED_HEADERS:
            bt_cors.ALLOWED_HEADERS.append(header)

# Create image with all dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
    .add_local_python_source("evals")
    .add_local_dir("datasets", "/root/datasets")
    .add_local_file("dataset.jsonl", "/root/dataset.jsonl")
)

app = modal.App(
    os.environ.get(
        "MODAL_APP_NAME",
        "curtis-pydantic-supervisor-eval-server",
    ),
    image=modal_image,
)

# Always read secrets from local .env and send them as a Secret
_secrets = [modal.Secret.from_dotenv()]


@app.function(
    secrets=_secrets,
    # Keep the server warm with at least 1 instance
    min_containers=1,
    # Timeout for long-running evals
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def braintrust_eval_server():
    """
    Run Braintrust remote eval dev server on Modal.

    This uses Braintrust's built-in create_app() function which handles
    all the routing, middleware, and ASGI app setup automatically.
    """
    from pathlib import Path

    # IMPORTANT: Apply the SDK patch BEFORE any Braintrust imports
    # This ensures the patched version is used when evaluators are loaded
    from evals.braintrust_parameter_patch import apply_parameter_patch
    from evals.braintrust_gateway_header_patch import apply_gateway_header_patch

    apply_parameter_patch()
    apply_gateway_header_patch()

    # Now import Braintrust components (they will use the patched version)
    from braintrust.cli.eval import EvaluatorState, FileHandle, update_evaluators
    from braintrust.devserver import cors as bt_cors
    from braintrust.devserver.server import create_app
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    import evals
    from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic
    from src.config import AgentConfig
    from src.tracing import configure_adk_tracing

    # Find all eval files in the evals directory
    # In Modal, the evals package is mounted and importable
    if hasattr(evals, "__path__") and evals.__path__:
        evals_dir = Path(evals.__path__[0])
    elif hasattr(evals, "__file__") and evals.__file__:
        evals_dir = Path(evals.__file__).parent
    else:
        raise RuntimeError("Could not locate evals package directory")

    print(f"Scanning for evaluators in {evals_dir}")

    # Find all eval_*.py files (matching braintrust CLI pattern)
    eval_files = list(evals_dir.glob("eval_*.py"))
    print(f"Found {len(eval_files)} eval file(s): {[f.name for f in eval_files]}")

    # Load evaluators using Braintrust's CLI loader
    handles = [FileHandle(in_file=str(eval_file)) for eval_file in eval_files]
    eval_state = EvaluatorState()
    update_evaluators(eval_state, handles, terminate_on_failure=True)
    evaluators = [e.evaluator for e in eval_state.evaluators]

    print(f"Loaded {len(evaluators)} evaluator(s): {[e.eval_name for e in evaluators]}")

    # Keep Braintrust's built-in CORS middleware aligned with this wrapper.
    _sync_braintrust_cors(bt_cors)

    app = create_app(evaluators, org_name=None)

    # Configure tracing profile for interactive requests.
    configure_adk_tracing(
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
        project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
        project_name=os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor"),
    )
    supervisor = get_supervisor(force_rebuild=True)

    async def interactive_page(_: Request) -> HTMLResponse:
        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Pydantic Supervisor</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; min-height: 90px; font-size: 16px; padding: 10px; }
    button { margin-top: 10px; font-size: 16px; padding: 10px 14px; }
    pre { background: #f6f8fa; padding: 12px; overflow: auto; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Pydantic Supervisor</h1>
  <p>Submit a query to run the multi-agent supervisor and log the trace to Braintrust.</p>
  <textarea id="query" placeholder="Ask something...">What is 2+2?</textarea>
  <br />
  <button onclick="runQuery()">Run Query</button>
  <p id="status"></p>
  <h3>Response</h3>
  <pre id="result">(no result yet)</pre>
  <script>
    async function runQuery() {
      const query = document.getElementById('query').value;
      const status = document.getElementById('status');
      const result = document.getElementById('result');
      status.textContent = 'Running...';
      try {
        const res = await fetch('/interactive/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        result.textContent = JSON.stringify(data, null, 2);
        status.textContent = res.ok ? 'Done' : 'Error';
      } catch (err) {
        status.textContent = 'Request failed';
        result.textContent = String(err);
      }
    }
  </script>
</body>
</html>
"""
        return HTMLResponse(html)

    async def interactive_query(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        query = str(payload.get("query", "")).strip()
        if not query:
            return JSONResponse({"error": "Missing non-empty `query`"}, status_code=400)

        workflow_name = str(payload.get("workflow_name", "pydantic-supervisor-interactive")).strip()
        supervisor_model = str(payload.get("supervisor_model", "")).strip()
        research_model = str(payload.get("research_model", "")).strip()
        math_model = str(payload.get("math_model", "")).strip()

        selected_supervisor = supervisor
        if supervisor_model or research_model or math_model:
            config = AgentConfig(
                supervisor_model=supervisor_model or AgentConfig.model_fields["supervisor_model"].default,
                research_model=research_model
                or supervisor_model
                or AgentConfig.model_fields["research_model"].default,
                math_model=math_model
                or supervisor_model
                or AgentConfig.model_fields["math_model"].default,
            )
            selected_supervisor = get_supervisor(config=config, force_rebuild=True)

        run_result = await run_supervisor_with_critic(
            supervisor=selected_supervisor,
            query=query,
            app_name=workflow_name or "pydantic-supervisor-interactive",
        )
        return JSONResponse(
            {
                "query": query,
                "workflow_name": workflow_name,
                "resolved_models": {
                    "supervisor_model": supervisor_model or AgentConfig.model_fields["supervisor_model"].default,
                    "research_model": research_model
                    or supervisor_model
                    or AgentConfig.model_fields["research_model"].default,
                    "math_model": math_model
                    or supervisor_model
                    or AgentConfig.model_fields["math_model"].default,
                },
                "final_output": run_result.get("final_output", ""),
                "messages": run_result.get("messages", []),
            }
        )

    app.router.routes.append(Route("/interactive", interactive_page, methods=["GET"]))
    app.router.routes.append(Route("/interactive/query", interactive_query, methods=["POST"]))
    return _with_playground_cors(app)


# Optional: Add a local entrypoint for testing
@app.local_entrypoint()
def test():
    """Test the deployment locally."""
    print("Testing Braintrust eval server deployment...")
    print("Deploy with: modal deploy src/eval_server.py")
    print("After deployment, you can connect to it from the Braintrust Playground")
