#!/usr/bin/env bash
# Exercise record_session.py's failure handling with fake hardware (no robot moves).
#   tests/run_recording_tests.sh [workdir]
set -uo pipefail
cd "$(dirname "$0")/.."
source ./config.env
PY="$VENV_BIN/python"
W="${1:-$(mktemp -d)}"
export HF_HUB_OFFLINE=1
PASS=0; FAIL=0

run() {  # run <scenario> <session> <episodes> <expected_rc>
  "$PY" -u tests/fake_hw_record.py "$1" "$W/$2" "$3" > "$W/$2.$1.out" 2>&1
  local rc=$?
  if [ "$rc" -eq "$4" ]; then echo "  ok   $1 -> exit $rc"; else echo "  FAIL $1 -> exit $rc (expected $4); see $W/$2.$1.out"; FAIL=$((FAIL+1)); return 1; fi
}
verify() {  # verify <session> <expected_episodes>
  if "$PY" tests/verify_session.py "$W/$1" "$2" > "$W/$1.verify.out" 2>&1; then
    echo "  ok   $1 has $2 demos, every frame decodes"; PASS=$((PASS+1))
  else
    echo "  FAIL $1: $(grep -E 'Error|assert' "$W/$1.verify.out" | tail -1)"; FAIL=$((FAIL+1))
  fi
}

echo "== normal session (3 demos)";            run normal s_normal 3 0 && verify s_normal 3
echo "== <- redo during demo 2";                run rerecord s_rerec 3 0 && verify s_rerec 3
echo "== accidental double press (0.3 s demo)"; run short s_short 3 0 && verify s_short 3
echo "== arm unplugged mid-demo 2";             run crash_record s_crash 3 3 && verify s_crash 1
echo "   ... plug back in, resume";             run normal s_crash 3 0 && verify s_crash 3
echo "== camera lost during reset after demo 2"; run crash_reset s_reset 3 3 && verify s_reset 2
echo "   ... resume";                           run normal s_reset 3 0 && verify s_reset 3
echo "== Esc during demo 2";                    run esc_record s_escr 3 2 && verify s_escr 1
echo "== Esc during reset after demo 2";        run esc_reset s_escs 3 2 && verify s_escs 2
echo "== Ctrl+C during demo 2";                 run ctrl_c s_ctrlc 3 130 && verify s_ctrlc 1
echo "== kill -9 during demo 2 (no cleanup at all)"; run hard_kill s_kill 3 137
echo
echo "RESULT: $PASS verified, $FAIL failed   (work dir: $W)"
[ "$FAIL" -eq 0 ]
