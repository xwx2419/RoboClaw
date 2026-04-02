# EAP Prompt Pairs

These prompt pairs are copied from the repo’s `_ROLLOUT_TASK_SEQUENCES` and are intended to be used *verbatim* to reduce prompt drift.

Usage:

- **Data collection (forward + reverse)**: use both prompts as a self-resetting pair.

Source:

- `src/agent_demo/agent_layer/agent_components/agent_tools/agent_tools.py`

## `primer`

- forward: `Put the primer into the drawer labeled PRIMER and close it.`
- reverse: `Open the drawer labeled 'PRIMER'. Take out the primer and place it on the table.`

## `lipstick`

- forward: `Use your right arm to put the white YSL lipstick into the "LIPSTICK" holder.`
- reverse: `Using your right arm to remove the white YSL lipstick from the "LIPSTICK" holder and place it on the table.`

## `lotion`

- forward: `Put the lotion into the 'LOTION' compartment.`
- reverse: `Use your right arm to take the white body lotion bottle out of the 'LOTION' compartment and place it on the desk.`

## `wipe`

- forward: `Use your left arm to pull a tissue from a pack of tissues and wipe the toning water spilled on the table.`
- reverse: `Use your left arm to pick up the toning water bottle and invert it to let the liquid drip onto the table.`
