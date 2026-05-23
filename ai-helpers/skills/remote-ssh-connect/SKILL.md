---
name: remote-ssh-connect
description: >
  Open a new VSCode window connected to the EC2 via Remote SSH extension using
  a .pem key. Verifies infra is running (invokes /provision-aws-lab if not).
  Persists PEM_PATH inside this file. Trigger: remote ssh, vscode ec2, abrir
  vscode remoto, conectar ssh, remote connect.
---

# Remote SSH Connect — Skill

Open a new VS Code window connected to the narwhal-extractor EC2 via Remote SSH.

**Stored PEM_PATH:** `C:\Users\jooju\.aws\narwhal-keypair.pem`

---

## Step 0 — Resolve PEM_PATH

### 0a. Check if the user provided a PEM_PATH argument

If the user passed a path as argument to this skill (e.g., `/remote-ssh-connect C:\path\to\key.pem`), use that value AND update the **Stored PEM_PATH** line above in this file. Use the Edit tool to replace the old path with the new one. The line to edit is:

```
**Stored PEM_PATH:** `<old_path>`
```

Replace `<old_path>` with the new path. This persists the default for future invocations.

### 0b. No argument provided — use stored default

If no argument was provided, read the **Stored PEM_PATH** line above. Extract the path between the backticks.

### 0c. No stored default exists

If the stored path is `NONE` or empty, ask the user for the PEM file path using AskUserQuestion. Once provided, update the stored path (same edit as 0a).

### 0d. Validate the PEM file exists

```powershell
Test-Path "<PEM_PATH>"
```

If the file does not exist, tell the user and **stop**.

---

## Step 1 — Ensure Remote SSH Extension is Installed

```powershell
code --list-extensions
```

Check if `ms-vscode-remote.remote-ssh` is in the list. If NOT:

```powershell
code --install-extension ms-vscode-remote.remote-ssh
```

Wait for installation to complete.

---

## Step 2 — Verify Infrastructure is Running

### 2a. Read infra config

Read `C:\Users\jooju\OneDrive\Área de Trabalho\data-ingestion\terraform\variables.tf` to extract:
- `aws_region` (default: us-east-1)
- `ec2_key_pair_name` (default: narwhal-keypair)

The EC2 tag name is `narwhal-extractor` (from ec2.tf).

### 2b. Check EC2 state

Clear stale env vars and query:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instances --region us-east-1 --filters "Name=tag:Name,Values=narwhal-extractor" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]" --output text
```

### 2c. Decision tree

| Result | Action |
|---|---|
| `running` + has IP | Record `$IP` → go to Step 3 |
| `stopped` | Start it, wait for running + IP → go to Step 3 |
| `pending` / `stopping` | Wait 30s, re-check |
| **No instance / AWS auth fails** | Invoke `/provision-aws-lab` skill, then re-check for IP |

When invoking `/provision-aws-lab`, tell the user the infrastructure needs to come up first. After it completes, extract the IP from its output.

### 2d. Start a stopped instance

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 start-instances --instance-ids $ID --region us-east-1
```

Poll every 15s until `running` with a public IP (max 10 attempts):

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instances --instance-ids $ID --region us-east-1 --query "Reservations[0].Instances[0].[State.Name,PublicIpAddress]" --output text
```

---

## Step 3 — Configure SSH

### 3a. Ensure .ssh directory exists

```powershell
if (-not (Test-Path "$env:USERPROFILE\.ssh")) { New-Item -ItemType Directory "$env:USERPROFILE\.ssh" }
```

### 3b. Write or update SSH config

Read `$env:USERPROFILE\.ssh\config` (create if it doesn't exist). Look for a `Host narwhal-extractor` block.

**If the block exists**, update `HostName` to the current `$IP` and `IdentityFile` to the current `$PEM_PATH`.

**If no block exists**, append this block to the file:

```
Host narwhal-extractor
    HostName $IP
    User ec2-user
    IdentityFile $PEM_PATH
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

`StrictHostKeyChecking no` is needed because the EC2 IP changes each start and the host key won't match.

### 3c. Fix PEM file permissions (Windows)

The PEM file must not be world-readable. Run:

```powershell
icacls "$PEM_PATH" /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

This removes inherited permissions and grants read-only to the current user — equivalent to `chmod 400` on Linux.

---

## Step 4 — Open VS Code with Remote SSH

```powershell
code --remote ssh-remote+narwhal-extractor /home/ec2-user
```

This opens a new VS Code window connected to the EC2 at `/home/ec2-user`. The user will see the remote filesystem, can open terminals, and run notebooks with the EC2's Python/Jupyter kernel.

---

## Step 5 — Confirm to User

Print a summary:

```
============================================================
  VS Code Remote SSH — Connected
============================================================

  Host:          narwhal-extractor (ec2-user@$IP)
  PEM:           $PEM_PATH
  Remote Path:   /home/ec2-user
  SSH Config:    ~/.ssh/config (Host narwhal-extractor)

  A new VS Code window should be opening.
  If prompted for platform, select "Linux".

  Tips:
  - Open notebooks in /home/ec2-user/notebooks/
  - Terminal: Ctrl+` (runs bash on EC2)
  - Jupyter: select Python 3.11 kernel when opening .ipynb

============================================================
```

---

## Error Handling

| Error | Resolution |
|---|---|
| PEM file not found | Ask user for correct path. |
| Extension install fails | Tell user to install `ms-vscode-remote.remote-ssh` manually from VS Code marketplace. |
| AWS credentials expired | Invoke `/provision-aws-lab` which handles credential refresh. |
| SSH connection refused | Instance may still be booting. Wait 30s, retry `code --remote`. |
| VS Code prompts for password | PEM permissions are wrong. Re-run icacls fix (Step 3c). |
| "Host key verification failed" | `StrictHostKeyChecking no` in config should prevent this. If persists, delete the old entry from `~/.ssh/known_hosts`. |
