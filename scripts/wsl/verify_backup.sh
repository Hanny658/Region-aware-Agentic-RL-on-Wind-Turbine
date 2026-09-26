#!/bin/bash
# Content-level check of the archives before anything is deleted (2026-09-26).
#
# The backup script already proved each archive is complete and readable end to end. This proves it archived the
# right tree and that files come back out intact: it counts the classes of file the work actually depends on, and
# then extracts one wind field and one baseline and compares them byte for byte with the originals.
#
#   bash scripts/wsl/verify_backup.sh
set -u
D=/mnt/g/wind-n-openfast-backup/wsl
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0

echo "=== what the archives contain, against the live trees"
printf "%-34s %10s %10s\n" "file class" "archive" "live"

chk() {  # label  tar-grep-pattern  archive  live-find-cmd
  local lab=$1 pat=$2 arc=$3 live=$4
  local a l
  a=$(tar -tf "$D/$arc" | grep -c "$pat")
  l=$(eval "$live" | wc -l)
  printf "%-34s %10s %10s" "$lab" "$a" "$l"
  if [ "$a" -eq "$l" ] && [ "$a" -gt 0 ]; then echo "  ok"; else echo "  MISMATCH"; fail=1; fi
}

chk "wind600 .bts fields"        "wind600/.*\.bts$"      wtrl.tar           "ls \$HOME/wtrl/wind600/*.bts 2>/dev/null"
chk "wind .inp (low-turb recipe)" "wtrl/wind/.*\.inp$"   wtrl.tar           "ls \$HOME/wtrl/wind/*.inp 2>/dev/null"
chk "baselines .npz (~/wtrl)"    "baselines/openfast/.*\.npz$" wtrl.tar     "ls \$HOME/wtrl/baselines/openfast/*.npz 2>/dev/null"
chk "baselines .npz (~/wtrl600)" "baselines/openfast/.*\.npz$" wtrl600.tar  "ls \$HOME/wtrl600/baselines/openfast/*.npz 2>/dev/null"
chk "ckpt_best.pt"               "exp/.*/ckpt_best\.pt$" wtrl.tar           "find \$HOME/wtrl/exp -name ckpt_best.pt 2>/dev/null"
chk "eval_*.json"                "exp/.*eval_.*\.json$"  wtrl.tar           "find \$HOME/wtrl/exp -name 'eval_*.json' 2>/dev/null"
chk "decisions.jsonl"            "exp/.*/decisions\.jsonl$" wtrl.tar        "find \$HOME/wtrl/exp -name decisions.jsonl 2>/dev/null"
chk "llm_transcript.jsonl"       "exp/.*/llm_transcript\.jsonl$" wtrl.tar   "find \$HOME/wtrl/exp -name llm_transcript.jsonl 2>/dev/null"
# the archive lists every directory, so count the templates and their two data subdirectories alike
chk "OpenFAST template dirs"     "wtrl/runs/template_5mw.*/$" wtrl.tar      "find \$HOME/wtrl/runs -type d -name 'template_5mw*' -o -type d -path '*template_5mw*/*' 2>/dev/null"

echo
echo "=== extract two files and compare byte for byte with the originals"
for rel in "wtrl/wind600/U24_TIB_S10.bts" "wtrl600/baselines/openfast/U12_TIB_S7.npz"; do
  arc="wtrl.tar"; case "$rel" in wtrl600/*) arc="wtrl600.tar";; esac
  if tar -xf "$D/$arc" -C "$T" "$rel" 2>/dev/null; then
    if cmp -s "$T/$rel" "$HOME/$rel"; then
      printf "  %-46s identical (%s)\n" "$rel" "$(du -h "$HOME/$rel" | cut -f1)"
    else
      printf "  %-46s DIFFERS\n" "$rel"; fail=1
    fi
  else
    printf "  %-46s NOT IN ARCHIVE\n" "$rel"; fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then echo "CONTENT VERIFIED - the archives hold the right trees and restore byte-identical"
else echo "CONTENT CHECK FAILED - DO NOT DELETE ANYTHING"; fi
exit "$fail"
