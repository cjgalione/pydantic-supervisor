"""Modal deployment for Braintrust remote eval dev server."""

import os

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

# Create image with all dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src")
    .add_local_python_source("evals")
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

    from braintrust.cli.eval import EvaluatorState, FileHandle, update_evaluators
    from braintrust.devserver import cors as bt_cors
    from braintrust.devserver.server import create_app
    from starlette.middleware.cors import CORSMiddleware

    import evals
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

    configure_adk_tracing(
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
        project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
        project_name=os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor"),
    )

    # Braintrust devserver has its own CORS middleware; ensure Playground's
    # `x-bt-use-gateway` preflight header is recognized there as well.
    if "x-bt-use-gateway" not in bt_cors.ALLOWED_HEADERS:
        bt_cors.ALLOWED_HEADERS.append("x-bt-use-gateway")

    # Use Braintrust's built-in create_app which handles all the setup.
    app = create_app(evaluators, org_name=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://www.braintrust.dev",
            "https://www.braintrustdata.com",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=CORS_ALLOWED_HEADERS,
        expose_headers=[
            "x-bt-cursor",
            "x-bt-found-existing-experiment",
            "x-bt-span-id",
            "x-bt-span-export",
        ],
        max_age=86400,
    )
    return app


# Optional: Add a local entrypoint for testing
@app.local_entrypoint()
def test():
    """Test the deployment locally."""
    print("Testing Braintrust eval server deployment...")
    print("Deploy with: modal deploy src/app.py")
    print("After deployment, you can connect to it from the Braintrust Playground")
