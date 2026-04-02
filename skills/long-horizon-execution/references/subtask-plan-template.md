# Subtask Plan Template

Use this file when building or revising the plan for a long-horizon task.

## Minimal Schema

Each subtask should have:

- `subtask_id`: stable identifier
- `objective`: short natural-language statement of what this step achieves
- `prompt`: the policy instruction passed to `$monitored-subtask-execution`
- `success_check`: observable condition that can be verified after execution
- `reset_after`: `true` when the rollout should end with a reset, otherwise `false`
- `max_retries`: maximum retry count before escalation
- `notes`: optional risk or dependency notes

## Progress Log Fields

For each attempt, log at least:

- `subtask_id`
- `attempt_index`
- `prompt`
- `policy_host/port`
- `step_interval`
- `timeout_s`
- `poll_interval_s`
- `reset_after`
- `status_text`
- `verification_result`
- `failure_reason`
- `started_at`
- `finished_at`

## Example Plan

`global_goal`: organize the vanity table

1. `subtask_id = place-primer`
   `prompt = Put the primer into the drawer labeled PRIMER and close it.`
   `success_check = primer is inside the PRIMER drawer and the drawer is closed`
   `reset_after = false`
   `max_retries = 2`

2. `subtask_id = place-lipstick`
   `prompt = Use your right arm to put the white YSL lipstick into the "LIPSTICK" holder.`
   `success_check = lipstick is seated in the LIPSTICK holder`
   `reset_after = false`
   `max_retries = 2`

3. `subtask_id = place-lotion`
   `prompt = Put the lotion into the 'LOTION' compartment.`
   `success_check = lotion is inside the LOTION compartment`
   `reset_after = false`
   `max_retries = 2`

Use separate subtasks whenever the success check or recovery path changes.
