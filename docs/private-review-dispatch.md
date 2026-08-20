# Private upstream RIFT review dispatch

This workflow is dispatch-only. It never checks out or executes pull-request
content. It is eligible only when all three conditions are true:

- base repository is `oshaughn/research-projects-RIT` (enforced by workflow
  location and exact OIDC repository identity);
- PR author and head repository owner are `oshaughnessy-junior`;
- base branch is `rift_O4c`, `master`, or `rift_O4d`.

The coordinator must independently re-read the PR and enforce the same author,
head-owner, and base-branch allowlists. Workflow fields are only routing hints.

Before merge, an upstream owner must create GitHub environment
`private-review-dispatch-rift-upstream`; configure repository-specific
Tailscale WIF variables; install the reviewer App only on this repository with
metadata-read and pull-request-write; and complete the negative WIF/ACL tests.
The coordinator uses a separate ledger, OIDC audience, WIF credential, and
tailnet endpoint from the junior-fork review service.

Approval and automatic merge are disabled. A successful COMMENT review asks
OpenClaw/main to alert Richard for manual merge.
