#!/usr/bin/env bash
# ============================================================================
# One rented GPU machine on AWS, with guard rails. Settings: aws/gpu_box.env.
#
#   ./gpu_box.sh launch      start the machine (asks first; it costs money from then on)
#   ./gpu_box.sh status      is it running, for how long, what has it cost so far
#   ./gpu_box.sh ssh [cmd]   open a shell on it (or run one command)
#   ./gpu_box.sh push <local> <remote>   copy files/folders to it (rsync)
#   ./gpu_box.sh pull <remote> <local>   copy results back (rsync, resumable)
#   ./gpu_box.sh terminate   delete the machine (its disk goes with it - pull first)
#
# Guard rails, all on by default:
# * HARD TIME LIMIT: the machine powers itself off after MAX_HOURS, and powering off
#   TERMINATES it (no idle machine or disk keeps billing). Set in the boot script, so it
#   happens even if this PC is off.
# * SSH is open to THIS PC's public IP only.
# * Spot (MARKET=spot) is ~60% cheaper; AWS may reclaim it with 2 minutes' warning, so
#   training must checkpoint and results must be pulled regularly.
# * Only one machine at a time: launch refuses if one tagged by this script is running.
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gpu_box.env"
export PATH="$HOME/.local/bin:$PATH" AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
STATE="$HERE/state"; mkdir -p "$STATE"
KEY="$HOME/.ssh/${KEY_NAME}.pem"
die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo ":: $*"; }

aws sts get-caller-identity >/dev/null 2>&1 || die "Not logged in to AWS. Run: aws login"

instance() {   # id of the running/pending box, or empty
  aws ec2 describe-instances --filters "Name=tag:Name,Values=$BOX_NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[0].InstanceId' --output text | grep -v None | head -1
}
ip_of() { aws ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$STATE/known_hosts"
          -o ServerAliveInterval=30 -o ConnectTimeout=10)

ensure_key_and_sg() {
  if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
    [ -f "$KEY" ] && die "$KEY exists but AWS has no key '$KEY_NAME'. Move the file aside first."
    info "Creating SSH key '$KEY_NAME' -> $KEY" >&2
    ( umask 077; aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 \
        --query KeyMaterial --output text > "$KEY" ) || die "key creation failed"
  fi
  [ -f "$KEY" ] || die "AWS has key '$KEY_NAME' but $KEY is missing. Delete the AWS key and retry."
  local myip sg
  myip="$(curl -s https://checkip.amazonaws.com)/32"
  sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [ -z "$sg" ] || [ "$sg" = None ]; then
    info "Creating security group '$SG_NAME' (SSH from $myip only)" >&2
    sg=$(aws ec2 create-security-group --group-name "$SG_NAME" \
          --description "SSH from the lab workstation only" --query GroupId --output text) || die "SG failed"
  fi
  # (Re)allow this PC's current IP; it can change between days.
  aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$myip" >/dev/null 2>&1
  echo "$sg"
}

case "${1:-status}" in
launch)
  [ -n "$(instance)" ] && die "A machine is already running ($(instance)). ./gpu_box.sh status"
  ami=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text) || die "AMI lookup failed"
  echo "About to start: $INSTANCE_TYPE ($MARKET) in $REGION, ${DISK_GB} GB disk, image $ami"
  echo "It will delete itself after $MAX_HOURS h at the latest. Approx. cost: \$$PRICE_HINT/hour."
  read -r -p "Start it now? [y/N] " r; [[ "$r" =~ ^[Yy] ]] || exit 1
  sg=$(ensure_key_and_sg) || die "SSH key / security group setup failed"
  market=(); [ "$MARKET" = spot ] && market=(--instance-market-options \
    'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  userdata=$(printf '#!/bin/bash\nshutdown -h +%d "gpu_box.sh hard time limit"\n' "$((MAX_HOURS * 60))")
  id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$INSTANCE_TYPE" "${market[@]}" \
        --key-name "$KEY_NAME" --security-group-ids "$sg" \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options HttpTokens=required \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "$userdata" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$BOX_NAME}]" \
        --query 'Instances[0].InstanceId' --output text) || die "launch failed (quota? capacity? see above)"
  echo "$id" > "$STATE/instance_id"; date +%s > "$STATE/launched_at"
  info "Launched $id - waiting for it to boot"
  aws ec2 wait instance-running --instance-ids "$id" || die "did not reach running"
  ip=$(ip_of "$id")
  for _ in $(seq 60); do ssh "${SSH_OPTS[@]}" "ubuntu@$ip" true 2>/dev/null && break; sleep 5; done
  ssh "${SSH_OPTS[@]}" "ubuntu@$ip" nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    && echo "Ready: ./gpu_box.sh ssh   (ubuntu@$ip)"
  ;;
status)
  id=$(instance)
  [ -z "$id" ] && { echo "No machine running. Nothing is being billed by gpu_box.sh."; exit 0; }
  aws ec2 describe-instances --instance-ids "$id" --query \
    'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,InstanceLifecycle,PublicIpAddress,LaunchTime]' --output text
  if [ -f "$STATE/launched_at" ]; then
    h=$(awk -v s="$(cat "$STATE/launched_at")" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}')
    echo "Running ${h} h, about \$$(awk -v h="$h" -v p="$PRICE_HINT" 'BEGIN{printf "%.2f", h*p}') so far; hard limit ${MAX_HOURS} h."
  fi
  ;;
ssh)
  id=$(instance); [ -n "$id" ] || die "no machine running"
  shift; exec ssh "${SSH_OPTS[@]}" "ubuntu@$(ip_of "$id")" "$@"
  ;;
push|pull)
  id=$(instance); [ -n "$id" ] || die "no machine running"
  ip=$(ip_of "$id"); src="$2"; dst="$3"
  [ "$1" = push ] && dst="ubuntu@$ip:$dst" || src="ubuntu@$ip:$src"
  exec rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$src" "$dst"
  ;;
terminate)
  id=$(instance); [ -n "$id" ] || { echo "No machine running."; exit 0; }
  read -r -p "Terminate $id? Its disk is deleted too - pull results first. [y/N] " r; [[ "$r" =~ ^[Yy] ]] || exit 1
  aws ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[0].CurrentState.Name' --output text
  rm -f "$STATE/instance_id" "$STATE/launched_at"
  ;;
*) sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
