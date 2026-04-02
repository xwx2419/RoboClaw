# Data Collection Guardrails

Stable EAP collection comes from repeatable reset behavior and traceable logs, not from trying to fix everything with prompt changes.

## 1) Always prefer deterministic stop/reset

- Route every rollout through `$monitored-subtask-execution` so timeout, failure, and uncertain outcomes all follow the same stop -> reset path.
- If behavior becomes unsafe or hard to explain, stop and reset first. Diagnose later.

## 2) Treat prompts as versioned inputs

- Reuse validated prompts from `references/eap-prompt-pairs.md` whenever possible to reduce drift.
- When a prompt changes, treat it as a new version and record that version in the dataset metadata.

## 3) Make the starting state explicit

- The key requirement is returning to a reusable starting state at the end of each round.
- Check the starting-state conditions after each round, for example object placement, drawer state, and robot pose. Stop collection and reset before continuing if the check fails.

## 4) Log the minimum fields needed for replay

For every collection round, log at least:

- `task_name`, `round_id`, `direction`
- `prompt`, `prompt_version` or prompt hash
- `policy host/port`, `step_interval`, `timeout_s`, `poll_interval_s`
- `status_text`, `success`, `failure_reason`
- trajectory indices or paths for images, joint states, and action logs

## 5) Pause after repeated failures

- If failures exceed the threshold, for example 2 or 3 consecutive failures, pause automation and require a human to restore the environment to a safe starting state before continuing.
