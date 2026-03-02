from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import CapabilitySet, NormalizeContext, NormalizedFinding, TaskInput
from .parsing import normalize_findings_from_text
from .shim import ShimAdapterBase


class ClaudeAdapter(ShimAdapterBase):
    """Claude Code adapter with support for model selection and cc environment switching."""

    def __init__(self) -> None:
        super().__init__(
            provider_id="claude",
            binary_name="claude",
            capability_set=CapabilitySet(
                tiers=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
                supports_native_async=False,
                supports_poll_endpoint=False,
                supports_resume_after_restart=True,
                supports_schema_enforcement=True,
                min_supported_version="2.1.59",
                tested_os=["macos", "linux"],
            ),
        )

    def _auth_check_command(self, binary: str) -> List[str]:
        return [binary, "auth", "status"]

    def supported_permission_keys(self) -> List[str]:
        return ["permission_mode", "model", "cc_env"]

    def _resolve_cc_env(self, cc_env: Optional[str]) -> Dict[str, str]:
        """Resolve cc environment file and return env vars to inject.

        cc (Claude Multi-Environment Manager) stores configs in ~/.claude/envs/<name>
        Each file exports ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY
        """
        if not cc_env:
            return {}

        env_file = Path.home() / ".claude" / "envs" / cc_env
        if not env_file.exists():
            return {}

        env_vars = {}
        content = env_file.read_text()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]  # Remove 'export '
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
                    env_vars[key] = value

        return env_vars

    def _build_command(self, input_task: TaskInput) -> List[str]:
        # Default values
        permission_mode = "plan"
        model = None
        cc_env = None

        # Extract from provider_permissions
        raw_permissions = input_task.metadata.get("provider_permissions")
        if isinstance(raw_permissions, dict):
            value = raw_permissions.get("permission_mode")
            if isinstance(value, str) and value.strip():
                permission_mode = value.strip()

            model = raw_permissions.get("model")
            if not isinstance(model, str) or not model.strip():
                model = None

            cc_env = raw_permissions.get("cc_env")
            if not isinstance(cc_env, str) or not cc_env.strip():
                cc_env = None

        # Build command
        cmd = ["claude", "-p", "--permission-mode", permission_mode, "--output-format", "text"]

        if model:
            cmd.extend(["--model", model])

        cmd.append(input_task.prompt)
        return cmd

    def _get_env_override(self, input_task: TaskInput) -> Optional[Dict[str, str]]:
        """Return environment variables to inject for this task.

        Called by shim.run() to create the subprocess environment.
        """
        raw_permissions = input_task.metadata.get("provider_permissions")
        if not isinstance(raw_permissions, dict):
            return None

        cc_env = raw_permissions.get("cc_env")
        if not isinstance(cc_env, str) or not cc_env.strip():
            return None

        env_vars = self._resolve_cc_env(cc_env)
        if not env_vars:
            return None

        # Merge with current environment
        merged = os.environ.copy()
        merged.update(env_vars)
        return merged

    def _build_command_for_record(self) -> List[str]:
        return ["claude", "-p", "--permission-mode", "plan", "--model", "<model>", "--output-format", "text", "<prompt>"]

    def _is_success(self, return_code: int, stdout_text: str, stderr_text: str) -> bool:
        if return_code != 0:
            return False
        text = f"{stdout_text}\n{stderr_text}".lower()
        return "api error" not in text

    def normalize(self, raw: Any, ctx: NormalizeContext) -> List[NormalizedFinding]:
        text = raw if isinstance(raw, str) else ""
        return normalize_findings_from_text(text, ctx, "claude")
