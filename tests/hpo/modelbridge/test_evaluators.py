from __future__ import annotations

from pathlib import Path
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from aiaccel.hpo.modelbridge.config import ObjectiveConfig
from aiaccel.hpo.modelbridge.evaluators import build_evaluator, command_objective
from aiaccel.hpo.modelbridge.types import TrialContext


def _context() -> TrialContext:
    return TrialContext(
        scenario="demo",
        phase="macro",
        trial_index=0,
        params={"x": 0.5},
        seed=42,
        output_dir=Path("."),
    )


def test_build_evaluator_python_callable() -> None:
    config = ObjectiveConfig(target="tests.hpo.modelbridge.sample_objective.objective")
    evaluator = build_evaluator(config, base_env={"FROM": "TEST"})
    result = evaluator(_context())
    assert result.metrics["mae"] == 0.5
    env_payload = result.payload["env"]
    assert env_payload["FROM"] == "TEST"
    assert env_payload["AIACCEL_SCENARIO"] == "demo"
    assert env_payload["AIACCEL_PHASE"] == "macro"


def test_command_objective_runs_subprocess() -> None:
    payload = {"objective": 1.23, "metrics": {"mae": 1.23}}
    command = [
        "python",
        "-c",
        ("import json, sys; payload = {'objective': 1.23, 'metrics': {'mae': 1.23}}; json.dump(payload, sys.stdout)"),
    ]
    context = _context()
    result = command_objective(context, command=command, timeout=5.0, base_env={})
    assert result.objective == payload["objective"]
    assert result.metrics == payload["metrics"]


def test_http_objective() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # type: ignore[override]
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps({"objective": float(len(payload["params"])), "metrics": {"mae": 1.0}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"

    config = ObjectiveConfig(target=url)
    evaluator = build_evaluator(config, base_env={"FROM": "HTTP"})
    result = evaluator(_context())
    assert result.objective == 1.0
    assert result.metrics["mae"] == 1.0
    server.shutdown()
