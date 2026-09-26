#!/bin/bash
# What this project occupies inside WSL, by directory and by kind (2026-09-26).
# Written as a file because inline `wsl.exe -- bash -c` eats loop variables and command substitution here.
set -u
echo "=== project roots under \$HOME"
for d in "$HOME"/wtrl "$HOME"/wtrl600 "$HOME"/wtrl_rt "$HOME"/wtrl_rt_range "$HOME"/wtrl_mpc_range; do
  if [ -d "$d" ]; then printf "%-26s %8s\n" "${d/#$HOME/~}" "$(du -sh "$d" 2>/dev/null | cut -f1)"; fi
done

echo
echo "=== inside ~/wtrl"
for d in "$HOME"/wtrl/*/; do
  printf "  %-24s %8s\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

echo
echo "=== wind banks (irreplaceable: the canonical seed pairing must never be regenerated)"
for d in "$HOME"/wtrl/wind "$HOME"/wtrl/wind600; do
  if [ -d "$d" ]; then
    printf "  %-24s %8s  %s .bts\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)" \
      "$(ls "$d"/*.bts 2>/dev/null | wc -l)"
  fi
done

echo
echo "=== paired GSPI/base baselines (.npz, carry the raw channels every percentage divides by)"
for h in "$HOME"/wtrl "$HOME"/wtrl600 "$HOME"/wtrl_rt "$HOME"/wtrl_rt_range "$HOME"/wtrl_mpc_range; do
  d="$h/baselines/openfast"
  if [ -d "$d" ]; then
    printf "  %-24s %8s  %s .npz\n" "${h/#$HOME/~}" "$(du -sh "$d" 2>/dev/null | cut -f1)" \
      "$(ls "$d"/*.npz 2>/dev/null | wc -l)"
  fi
done

echo
echo "=== experiment runs"
printf "  %-24s %8s  %s run dirs\n" "~/wtrl/exp" "$(du -sh "$HOME"/wtrl/exp 2>/dev/null | cut -f1)" \
  "$(ls -d "$HOME"/wtrl/exp/*/ 2>/dev/null | wc -l)"
printf "  %-24s %8s  %s files\n" "  checkpoints ckpt_*.pt" \
  "$(du -ch "$HOME"/wtrl/exp/*/ckpt_*.pt 2>/dev/null | tail -1 | cut -f1)" \
  "$(ls "$HOME"/wtrl/exp/*/ckpt_*.pt 2>/dev/null | wc -l)"
printf "  %-24s %8s  %s files\n" "  of which ckpt_best.pt" \
  "$(du -ch "$HOME"/wtrl/exp/*/ckpt_best.pt 2>/dev/null | tail -1 | cut -f1)" \
  "$(ls "$HOME"/wtrl/exp/*/ckpt_best.pt 2>/dev/null | wc -l)"
printf "  %-24s %8s  %s files\n" "  resume_*.pt (disposable)" \
  "$(du -ch "$HOME"/wtrl/exp/*/resume_*.pt 2>/dev/null | tail -1 | cut -f1)" \
  "$(ls "$HOME"/wtrl/exp/*/resume_*.pt 2>/dev/null | wc -l)"
printf "  %-24s %8s  %s files\n" "  evaluations (json+csv)" \
  "$(du -ch "$HOME"/wtrl/exp/*/eval_*.json "$HOME"/wtrl/exp/*/eval_*.csv "$HOME"/wtrl/exp/*/*/eval_*.json "$HOME"/wtrl/exp/*/*/eval_*.csv 2>/dev/null | tail -1 | cut -f1)" \
  "$(ls "$HOME"/wtrl/exp/*/eval_*.json "$HOME"/wtrl/exp/*/eval_*.csv "$HOME"/wtrl/exp/*/*/eval_*.json "$HOME"/wtrl/exp/*/*/eval_*.csv 2>/dev/null | wc -l)"
printf "  %-24s %8s  %s files\n" "  logs_* (per-episode)" \
  "$(du -ch "$HOME"/wtrl/exp/*/logs_* 2>/dev/null | tail -1 | cut -f1)" \
  "$(ls -d "$HOME"/wtrl/exp/*/logs_* 2>/dev/null | wc -l)"

echo
echo "=== regenerable by scripts/wsl/bootstrap.sh"
for d in "$HOME"/wtrl/mamba "$HOME"/wtrl/src "$HOME"/wtrl/tmp; do
  if [ -d "$d" ]; then printf "  %-24s %8s\n" "${d/#$HOME/~}" "$(du -sh "$d" 2>/dev/null | cut -f1)"; fi
done

echo
echo "=== filesystem"
df -h / | tail -1
