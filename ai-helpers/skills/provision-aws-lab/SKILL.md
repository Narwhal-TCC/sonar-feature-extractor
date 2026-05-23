---
name: provision-aws-lab
description: >
  Provision AWS Lab infrastructure for sonar-feature-extractor. Reads infra
  config from data-ingestion repo (source of truth), authenticates with
  provided credentials, runs Terraform to provision EC2+S3, waits for Jupyter,
  and returns the public URL. Trigger: provision, aws lab, subir infra, start
  ec2, iniciar lab, jupyter url.
---

# Provision AWS Lab — Skill

This skill provisions the full AWS Lab infrastructure for sonar-feature-extractor.
Follow every step in order. Do NOT skip steps.

---

## Step 0 — Read Infrastructure Config (Source of Truth)

The `data-ingestion` repository defines the infrastructure via Terraform. **Always read it first** — it may have changed since last session.

```
DATA_INGESTION_DIR = C:\Users\jooju\OneDrive\Área de Trabalho\data-ingestion
TERRAFORM_DIR      = $DATA_INGESTION_DIR\terraform
```

Read these files and extract the current values:

| File | Values to extract |
|---|---|
| `terraform/variables.tf` | `aws_region`, `data_bucket_name`, `ec2_instance_type`, `ec2_key_pair_name`, `jupyter_token`, `pipeline_workers` |
| `terraform/ec2.tf` | Instance tags (`Name`, `Project`), AMI filter, IAM profile, volume config |
| `terraform/s3.tf` | Bucket name, directory prefixes (raw/, features/, etc.) |
| `terraform/security_groups.tf` | SG name, ingress ports (22, 8888) |
| `terraform/outputs.tf` | Output format for jupyter_url, ssh_command |
| `terraform/main.tf` | Backend config, providers, Lambda definition |

Store the extracted values as variables for subsequent steps. Example expected defaults:

```
REGION          = us-east-1
BUCKET          = narwhal-data-293379721401
STATE_BUCKET    = narwhal-state-293379721401
INSTANCE_TYPE   = t3.medium
KEY_PAIR        = narwhal-keypair
JUPYTER_TOKEN   = narwhal-jupyter-2024
WORKERS         = 2
EC2_TAG_NAME    = narwhal-extractor
IAM_PROFILE     = LabInstanceProfile
```

**If any value differs from these defaults, use the value from the file.**

---

## Step 1 — Authenticate with AWS

### 1a. Locate the credentials file

The user provides the path to a credentials file. Common locations:
- `C:\Users\jooju\OneDrive\Área de Trabalho\data-ingestion\example_envs.txt`
- A file the user opens or references in their message

Read the file content. It must contain `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token` (AWS Academy uses temporary STS credentials with `ASIA` prefix).

### 1b. Write credentials

Read the current `C:\Users\jooju\.aws\credentials` file (required before Write), then overwrite it with the new credentials in INI format:

```ini
[default]
aws_access_key_id=<value>
aws_secret_access_key=<value>
aws_session_token=<value>
```

### 1c. Clear stale environment variables

AWS env vars override the credentials file. **Always clear them**.

Prepend this to EVERY `aws` and `terraform` CLI call in subsequent steps (PowerShell does not persist env var changes across tool calls):

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; <command>
```

### 1d. Validate

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws sts get-caller-identity --region $REGION
```

Must return a valid ARN. If it fails with `InvalidClientTokenId` or `InvalidToken`:
- Tell the user the credentials are expired
- Ask them to generate new ones from AWS Academy Learner Lab
- **Stop** — do not proceed with expired credentials

---

## Step 2 — Provision Infrastructure with Terraform

Terraform is the **single source of truth** for provisioning. All resources (EC2, S3, SG, Lambda) are defined in `$TERRAFORM_DIR`. Never recreate resources manually via AWS CLI — always use Terraform.

### 2a. Check for existing infrastructure

Before running Terraform, check if the EC2 instance already exists and just needs to be started:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instances --region $REGION --filters "Name=tag:Name,Values=$EC2_TAG_NAME" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]" --output text
```

**Decision tree:**

| Result | Action |
|---|---|
| Instance `running` | Record ID + IP → skip to Step 3 |
| Instance `stopped` | Start it → go to Step 2e |
| Instance `stopping` | Wait 30s, then start → go to Step 2e |
| Instance `pending` | Go to Step 2e (wait loop) |
| **No instance found** | Go to Step 2b (Terraform) |

### 2b. Ensure Terraform state bucket exists

The Terraform backend uses S3. The state bucket may not exist in a fresh lab:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws s3api head-bucket --bucket $STATE_BUCKET --region $REGION
```

If the bucket does NOT exist, create it:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws s3api create-bucket --bucket $STATE_BUCKET --region $REGION
```

### 2c. Terraform Init + Apply

Run from the Terraform directory. Use `-reconfigure` on init to handle backend changes between lab sessions.

**Init:**
```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; terraform -chdir="$TERRAFORM_DIR" init -reconfigure -backend-config="bucket=$STATE_BUCKET" -backend-config="key=terraform/narwhal/state.tfstate" -backend-config="region=$REGION"
```

If init fails with "state already locked" or backend errors, try adding `-migrate-state` instead of `-reconfigure`.

**Plan:**
```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; terraform -chdir="$TERRAFORM_DIR" plan -out=tfplan
```

Review the plan output. It should create: EC2 instance, S3 bucket (+ directory markers), security group, Lambda function. Confirm the resource count looks reasonable before applying.

**Apply:**
```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; terraform -chdir="$TERRAFORM_DIR" apply -auto-approve tfplan
```

Timeout: up to 5 minutes. Use `timeout: 600000` on this call.

### 2d. Extract outputs

After successful apply, read the Terraform outputs:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; terraform -chdir="$TERRAFORM_DIR" output -json
```

Extract from the JSON:
- `ec2_public_ip` → `$IP`
- `s3_data_bucket` → confirm matches `$BUCKET`
- `jupyter_url` → the final URL (sensitive output, may need `-raw`)

To get the instance ID (not in outputs), query by tag:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instances --region $REGION --filters "Name=tag:Name,Values=$EC2_TAG_NAME" "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text
```

### 2e. Wait for instance to be running (start or post-apply)

If the instance was started (not freshly created by Terraform), poll for running state:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 start-instances --instance-ids $ID --region $REGION
```

Then poll every 15 seconds, max 10 attempts:

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instances --instance-ids $ID --region $REGION --query "Reservations[0].Instances[0].[State.Name,PublicIpAddress]" --output text
```

Stop when state is `running` and PublicIpAddress is not `None`.

---

## Step 3 — Verify S3 Bucket

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws s3 ls s3://$BUCKET/ --region $REGION
```

Confirm the 4 prefixes exist (raw/, features/, pipelines/, notebooks/).

---

## Step 4 — Wait for JupyterLab to Be Ready

JupyterLab takes 2-5 minutes on first boot (user_data bootstrap installs packages). On subsequent starts, the systemd service restarts in ~30 seconds.

### 4a. Instance status checks

```powershell
$env:AWS_ACCESS_KEY_ID = $null; $env:AWS_SECRET_ACCESS_KEY = $null; $env:AWS_SESSION_TOKEN = $null; aws ec2 describe-instance-status --instance-ids $ID --region $REGION --query "InstanceStatuses[0].[InstanceStatus.Status,SystemStatus.Status]" --output text
```

Wait until both are `ok`. Poll every 20s, max 15 attempts.

### 4b. Port 8888 check

```powershell
try { $r = Invoke-WebRequest -Uri "http://${IP}:8888" -TimeoutSec 5 -UseBasicParsing; "ready ($($r.StatusCode))" } catch { "not ready" }
```

If status 200 or 302, JupyterLab is up. If not ready after instance checks pass, wait 30s and retry (up to 5 times). On first boot, JupyterLab may take up to 5 extra minutes while packages install.

---

## Step 5 — Output Results

Once everything is ready, print a clear summary:

```
============================================================
  AWS Lab Infrastructure — READY
============================================================

  EC2 Instance:  $ID ($INSTANCE_TYPE)
  Public IP:     $IP
  S3 Bucket:     $BUCKET
  Region:        $REGION

  JupyterLab:    http://$IP:8888/?token=$JUPYTER_TOKEN
  SSH:           ssh -i $KEY_PAIR.pem ec2-user@$IP

============================================================
```

The JupyterLab URL must be clickable. This is the primary deliverable of the skill.

---

## Step 6 — Update Instance ID in Memory

If the instance ID or IP changed from what is stored in memory (`project_aws_deployed.md`), update the memory file with the new values.

---

## Error Handling

| Error | Resolution |
|---|---|
| `InvalidClientTokenId` / `InvalidToken` | Credentials expired. Ask user to refresh from AWS Academy. **Stop.** |
| `AuthFailure` on EC2 but STS works | Stale env vars. Clear them (Step 1c) and retry. |
| `terraform init` fails (backend) | State bucket may not exist → create it (Step 2b). Or try `-migrate-state` instead of `-reconfigure`. |
| `terraform apply` fails (resource conflict) | Resource may exist outside Terraform state. Import it: `terraform import <resource> <id>`, then re-apply. |
| `terraform apply` fails (IAM/permissions) | Lab role may lack permissions. Check the error detail and report to user. |
| Key pair not found during apply | Tell user to create `$KEY_PAIR` in AWS Console → EC2 → Key Pairs. **Stop.** |
| Instance fails to start | Check instance status. If `terminated`, re-run `terraform apply`. |
| JupyterLab not responding after 5 min | SSH into instance and check: `sudo systemctl status jupyter`. On first boot, check `/var/log/cloud-init-output.log` for bootstrap progress. |
| Terraform state locked | Wait 60s and retry. If persistent, force unlock: `terraform force-unlock <LOCK_ID>` (with user confirmation). |
