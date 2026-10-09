#!/bin/bash
# Restore the project trees from the external drive into a fresh WSL (2026-10-09).
# Run as root the first time (it needs to edit /etc/wsl.conf); the trees are then owned by the normal user.
#
#   wsl -d Ubuntu-24.04 --user root -- bash /mnt/c/.../scripts/wsl/restore_from_g.sh
#   wsl --terminate Ubuntu-24.04      # so the default user takes effect
set -u
SRC=/mnt/g/wind-n-openfast-backup/wsl
U=hanny
H=/home/$U

# the stock Ubuntu image already ships a wsl.conf carrying a [boot] section, so append rather than create: writing
# the file outright would drop systemd, and testing "does the file exist" skips the edit entirely
if [ "$(id -u)" = "0" ] && ! grep -q '^\[user\]' /etc/wsl.conf 2>/dev/null; then
  printf '\n[user]\ndefault=%s\n' "$U" >> /etc/wsl.conf
  echo "added [user] default=$U to /etc/wsl.conf (effective after: wsl --terminate Ubuntu-24.04)"
fi

[ -d "$SRC" ] || { echo "backup not visible at $SRC - is G: connected?"; exit 1; }

for r in wtrl wtrl600 wtrl_rt wtrl_rt_range wtrl_mpc_range; do
  if [ -d "$H/$r" ]; then echo "skip $r (already present)"; continue; fi
  [ -f "$SRC/$r.tar" ] || { echo "missing archive $r.tar"; continue; }
  echo "=== $r $(date +%H:%M:%S)"
  tar -xf "$SRC/$r.tar" -C "$H" || { echo "    EXTRACT FAILED"; exit 1; }
  chown -R "$U:$U" "$H/$r" 2>/dev/null || true
  echo "    $(du -sh "$H/$r" | cut -f1)  $(find "$H/$r" | wc -l) entries"
done

echo
echo "=== what came back"
for r in wtrl wtrl600 wtrl_rt wtrl_rt_range wtrl_mpc_range; do
  [ -d "$H/$r" ] && printf "  %-18s %8s\n" "$r" "$(du -sh "$H/$r" | cut -f1)"
done
printf "  %-18s %8s\n" "wind600 .bts" "$(ls "$H"/wtrl/wind600/*.bts 2>/dev/null | wc -l)"
printf "  %-18s %8s\n" "run dirs" "$(ls -d "$H"/wtrl/exp/*/ 2>/dev/null | wc -l)"
df -h / | tail -1
echo
echo "then, as the normal user: ~/wtrl/run.sh python scripts/dev/integrity_check.py"
