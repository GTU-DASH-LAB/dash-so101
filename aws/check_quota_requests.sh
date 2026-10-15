#!/usr/bin/env bash
# ============================================================================
# Watches the H100 quota requests (us-east-1, 0 -> 16 vCPUs):
#   L-417A185B  Running On-Demand P instances
#   L-7212CCBC  All P Spot Instance Requests
#
# Runs from cron every 30 min. It does nothing (and costs nothing) until AWS's
# answer changes. On a change, a Claude agent (claude -p, no tools) explains
# what happened and, if AWS wants more information or said no, drafts the reply
# to paste into the support case. That report is emailed to you.
#
# Why the agent drafts instead of replying itself: this account has the Basic
# support plan, and the AWS Support API (the only way to post a case reply
# programmatically) needs a paid support plan. Replies go in by hand:
# https://support.console.aws.amazon.com/support/home#/case/history
#
#   ./check_quota_requests.sh            normal run (what cron calls)
#   ./check_quota_requests.sh --test     ask the agent about the current status
#                                        and print its report; no email, no state change
#
# When both requests are decided it stops checking (state/quota.done); remove the
# cron line then:  crontab -l | grep -v check_quota_requests | crontab -
# The `aws login` session expires after at most 12 h. When it does, you get one
# email asking you to run `aws login` again; AWS also emails the decision itself.
# ============================================================================
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/state"
LOG="$STATE/quota_check.log"
REGION=us-east-1
WANT=16
NOTIFY="$HOME/robotics/scripts/notify-progress.sh"
mkdir -p "$STATE"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
email() { "$NOTIFY" "$1" "$2" >> "$LOG" 2>&1 || log "email FAILED: $1"; }
TEST=0; [ "${1:-}" = "--test" ] && TEST=1

[ $TEST -eq 0 ] && [ -f "$STATE/quota.done" ] && exit 0

# --- 1. still logged in? -----------------------------------------------------------
if ! aws sts get-caller-identity --output text >/dev/null 2>"$STATE/sts.err"; then
  log "not logged in: $(head -c 200 "$STATE/sts.err")"
  if [ $TEST -eq 0 ] && [ ! -f "$STATE/login_expired.notified" ]; then
    email "AWS quota check paused: run 'aws login'" \
"The AWS login session on the workstation expired, so the H100 quota check can't run.
Run this in a terminal on the workstation:

    aws login

The check resumes by itself within 30 minutes. AWS also emails the account address when it decides."
    touch "$STATE/login_expired.notified"
  fi
  exit 0
fi
rm -f "$STATE/login_expired.notified"

# --- 2. current status: newest request per quota + the quota value in force --------
CUR="$STATE/current.json"
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region "$REGION" \
  --output json > "$STATE/history.json" 2>>"$LOG" || { log "history query failed"; exit 0; }
for c in L-417A185B L-7212CCBC; do
  v=$(aws service-quotas get-service-quota --service-code ec2 --quota-code "$c" --region "$REGION" \
        --query 'Quota.Value' --output text 2>>"$LOG") || v=unknown
  echo "$c $v"
done > "$STATE/values.txt"
WANT=$WANT python3 - "$STATE/history.json" "$STATE/values.txt" > "$CUR" <<'EOF' || { log "parse failed"; exit 0; }
import json, os, sys
hist = json.load(open(sys.argv[1]))["RequestedQuotas"]
vals = dict(l.split() for l in open(sys.argv[2]) if l.strip())
out = {}
for code in ("L-417A185B", "L-7212CCBC"):
    reqs = sorted((r for r in hist if r["QuotaCode"] == code), key=lambda r: r["Created"])
    r = reqs[-1] if reqs else {}
    out[code] = {"name": r.get("QuotaName"), "requested": r.get("DesiredValue"), "status": r.get("Status"),
                 "case_id": r.get("CaseId"), "quota_now": vals.get(code)}
print(json.dumps(out, indent=1, sort_keys=True))
EOF

PREV="$STATE/last.json"
if [ $TEST -eq 0 ] && [ -f "$PREV" ] && cmp -s "$CUR" "$PREV"; then
  log "no change: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(", ".join(f"{k}={v["status"]}/{v["quota_now"]}" for k,v in d.items()))' "$CUR")"
  exit 0
fi

# --- 3. something changed (or first run): let the agent explain it -----------------
decided=$(python3 - "$CUR" <<EOF
import json, sys
d = json.load(open(sys.argv[1]))
final = {"APPROVED", "DENIED", "NOT_APPROVED", "CASE_CLOSED", "INVALID_REQUEST"}
def done(v):
    try: granted = float(v["quota_now"]) >= $WANT
    except (TypeError, ValueError): granted = False
    return granted or v["status"] in final
print("yes" if all(done(v) for v in d.values()) else "no")
EOF
)
PROMPT="You monitor two AWS EC2 service-quota increase requests for a robotics researcher at Gebze Technical University (Turkey).
Goal: one p5.4xlarge (1x NVIDIA H100 80 GB, 16 vCPUs) in us-east-1 to fine-tune robot-learning models (pi0.5 vision-language-action
model; video world models such as VLA-JEPA / FLUX 3 Action) on their own 100-episode SO-101 robot-arm dataset, a few runs of
10-20 hours each, paid with the account's ~100 USD promotional credits. Both requests ask 0 -> 16 vCPUs.
The account has the Basic support plan, so any reply to an AWS support case must be pasted by the user in the Support Center.
Service Quotas statuses: PENDING (queued), CASE_OPENED (a support case exists; AWS may ask for details), APPROVED, DENIED,
CASE_CLOSED (case finished: check quota_now to see whether it was granted), NOT_APPROVED, INVALID_REQUEST.
quota_now is the limit actually in force now.

Previous status (empty on the first check):
$(cat "$PREV" 2>/dev/null)

Current status:
$(cat "$CUR")

Write a short plain-text email body (no markdown headings, no tables):
1. One or two sentences: what changed and what it means (can they launch the H100 now: yes/no, on-demand and/or spot).
2. What the user should do next, as 1-3 short steps.
3. Only if a support case is open, or a request was denied / not approved, or granted less than 16: a ready-to-paste reply
   for the support case, under 120 words, polite, factual, no exaggeration, asking for 16 vCPUs (one p5.4xlarge).
Do not invent facts beyond the above. Keep the whole email under 220 words."

REPORT=$(timeout 300 claude -p --model haiku --tools "" --no-session-persistence "$PROMPT" 2>>"$LOG")
if [ $? -ne 0 ] || [ -z "$REPORT" ]; then
  log "agent failed; sending raw status"
  REPORT="(The Claude agent did not answer; raw status below.)"
fi
BODY="$REPORT

---
Raw status (us-east-1):
$(cat "$CUR")
Support cases: https://support.console.aws.amazon.com/support/home#/case/history
Quotas: https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
[ "$decided" = yes ] && BODY="$BODY

Both requests are decided, so the check has stopped. Remove its cron line with:
crontab -l | grep -v check_quota_requests | crontab -"

if [ $TEST -eq 1 ]; then echo "$BODY"; exit 0; fi

SUMMARY=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(", ".join(f"{"On-demand" if k=="L-417A185B" else "Spot"} {v["status"]}" for k,v in d.items()))' "$CUR")
email "AWS H100 quota: $SUMMARY" "$BODY"
cp "$CUR" "$PREV"
log "change reported: $SUMMARY"
[ "$decided" = yes ] && { touch "$STATE/quota.done"; log "both decided; checking stopped"; }
exit 0
