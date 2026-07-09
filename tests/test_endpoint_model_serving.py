from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from llmdbenchmark.utilities.endpoint import test_model_serving as verify_model_serving


@dataclass
class FakeResult:
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False


class FakeCommandExecutor:
    def __init__(self, execute_result: FakeResult | None = None, kube_result: FakeResult | None = None):
        self.execute_result = execute_result or FakeResult()
        self.kube_result = kube_result or FakeResult()
        self.execute_calls: list[str] = []
        self.kube_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.dry_run = False
        self.logger = MagicMock()

    def execute(self, cmd: str, check: bool = True, force: bool = False, **_: Any) -> FakeResult:
        self.execute_calls.append(cmd)
        return self.execute_result

    def kube(self, *args: str, **kwargs: Any) -> FakeResult:
        self.kube_calls.append((tuple(args), dict(kwargs)))
        return self.kube_result


class TestTestModelServing:
    def test_localhost_endpoint_uses_local_execute(self):
        cmd = FakeCommandExecutor(
            execute_result=FakeResult(
                success=True,
                stdout='{"object":"list","data":[{"id":"Qwen/Qwen3-0.6B"}]}',
            )
        )

        error = verify_model_serving(
            cmd,
            "bench-1",
            "localhost",
            18000,
            "Qwen/Qwen3-0.6B",
            plan_config=None,
            max_retries=1,
        )

        assert error is None
        assert len(cmd.execute_calls) == 1
        assert len(cmd.kube_calls) == 1
        assert cmd.kube_calls[0][0][:3] == ("get", "role", "default-role")
        assert "localhost:18000/v1/models" in cmd.execute_calls[0]

    def test_cluster_endpoint_injects_no_proxy_for_host_and_port(self):
        cmd = FakeCommandExecutor(
            kube_result=FakeResult(
                success=True,
                stdout='{"object":"list","data":[{"id":"Qwen/Qwen3-0.6B"}]}',
            )
        )

        error = verify_model_serving(
            cmd,
            "bench-1",
            "10.244.0.123",
            80,
            "Qwen/Qwen3-0.6B",
            plan_config=None,
            max_retries=1,
        )

        assert error is None
        assert cmd.execute_calls == []
        assert len(cmd.kube_calls) == 2
        args, _ = cmd.kube_calls[-1]
        joined = " ".join(args)
        assert "--request-timeout=180s" in joined
        assert "--pod-running-timeout=180s" in joined
        assert "--env=NO_PROXY=127.0.0.1,localhost,10.244.0.123,10.244.0.123:80" in joined
        assert "--env=no_proxy=127.0.0.1,localhost,10.244.0.123,10.244.0.123:80" in joined
