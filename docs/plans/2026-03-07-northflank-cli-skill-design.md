# Northflank CLI Skill Design

**Date:** 2026-03-07
**Status:** Approved

## Purpose

A Claude Code skill that acts as a smart Northflank copilot — translating natural language requests into correct `northflank` CLI commands for full infrastructure management.

## Requirements

- **Full infrastructure management:** CRUD for all resource types (projects, services, jobs, addons, secrets, volumes, domains, pipelines, templates, clusters, integrations, log-sinks, registries)
- **Smart routing:** Natural language input → correct CLI commands with proper flags
- **Multi-project context awareness:** Detect current context, help switch between projects
- **Pre-authenticated:** Assumes `northflank login` already done
- **Safety guards:** Confirmation before destructive operations (delete, force-push, restart)

## Design Decisions

1. **Single SKILL.md** — all routing logic and command reference in one file, no external dependencies
2. **Allowed tools: Bash, Read, Write, Glob** — Bash for CLI execution, Read/Write for JSON templates, Glob for finding template files
3. **User-invocable** — callable via `/northflank`
4. **Argument-hint** — accepts natural language or explicit subcommands

## Architecture

### Command Routing

The skill receives natural language input via `$ARGUMENTS` and:
1. Checks current Northflank context (`northflank context ls`)
2. Maps the request to the appropriate command tree
3. Builds the command with correct flags
4. Shows the command before executing (for destructive ops, requires confirmation)
5. Formats output using `--output json` when parsing is needed

### Resource Coverage

| Resource | Commands | Notes |
|----------|----------|-------|
| Projects | list, create, get, update, delete | Context switching |
| Services | list, create, get, update, delete, pause, resume, restart, logs, metrics, builds, exec, forward | Full lifecycle |
| Jobs | list, create, get, update, delete, pause, resume, runs, logs, metrics, builds, exec | Manual + cron |
| Addons | list, create, get, update, delete, pause, resume, restart, backup, restore, import, credentials, logs, metrics | Database management |
| Secrets | list, create, get, update, delete, link/unlink | Env var management |
| Volumes | list, create, get, update, delete, attach, detach | Storage |
| Domains | list, create, get, delete, verify, subdomains, assign | DNS management |
| Templates | list, get, delete, run | IaC |
| Pipelines | list, get, release-flows | CI/CD |
| Clusters | list, create, get, delete | BYOC |
| Integrations | list, create, get, delete | Cloud providers |
| Log Sinks | list, create, get, update, delete, pause, resume | Observability |
| Registries | list, add, get, update, delete | Container registries |

### Safety Model

- **Safe (auto-execute):** list, get, logs, metrics, context ls
- **Confirm first:** create, update, pause, resume, restart, start, run
- **Explicit confirmation required:** delete, abort, force operations
