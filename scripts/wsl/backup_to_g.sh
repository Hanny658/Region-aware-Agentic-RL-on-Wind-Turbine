#!/bin/bash
# Archive everything this project keeps inside WSL to the external drive (2026-09-26).
#
# The WSL side is tarred rather than copied, because the target is exFAT and would silently drop the Linux
# permissions and the symlinks inside the micromamba environment. One archive per root, no compression: the wind
# fields and the .npz baselines are already binary and the drive has 931 GB free, so the time is better spent on
# verification than on deflate.
#
# Each archive is then read back end to end (tar -t) and its entry count compared with the source, which is what
# catches a truncated or corrupted write, and a sha256 is recorded so the archive can be re-verified later.
#
#   bash scripts/wsl/backup_to_g.sh
set -u
DEST=/mnt/g/wind-n-openfast-backup/wsl
mkdir -p "$DEST"
cd "$HOME" || exit 1
ROOTS="wtrl wtrl600 wtrl_rt wtrl_rt_range wtrl_mpc_range"
MAN="$DEST/MANIFEST.txt"
: > "$MAN"
echo "WSL archives written $(date -Is) from $(hostname):$HOME" >> "$MAN"
echo >> "$MAN"

fail=0
for r in $ROOTS; do
  [ -d "$HOME/$r" ] || { echo "skip $r (absent)"; continue; }
  echo "=== $r $(date +%H:%M:%S)"
  # count what goes in, excluding the transient scratch directory
  n_src=$(find "$HOME/$r" -path "$HOME/wtrl/tmp" -prune -o -print 2>/dev/null | wc -l)
  echo "    source entries: $n_src"
  tar --exclude="wtrl/tmp/*" --exclude="wtrl/tmp" -cf "$DEST/$r.tar" "$r" || { echo "    TAR FAILED"; fail=1; continue; }
  sz=$(stat -c %s "$DEST/$r.tar")
  echo "    archive: $(numfmt --to=iec "$sz")B"
  n_arc=$(tar -tf "$DEST/$r.tar" | wc -l)
  echo "    archive entries: $n_arc"
  sha=$(sha256sum "$DEST/$r.tar" | cut -d' ' -f1)
  echo "    sha256: $sha"
  if [ "$n_arc" -ne "$n_src" ]; then
    echo "    MISMATCH: $n_src in, $n_arc out"; fail=1
  else
    echo "    verified"
  fi
  printf "%-16s %14s bytes  %8s entries  sha256 %s\n" "$r.tar" "$sz" "$n_arc" "$sha" >> "$MAN"
done

echo >> "$MAN"
echo "entry counts are 'find | wc -l' of the source tree against 'tar -tf | wc -l' of the archive;" >> "$MAN"
echo "~/wtrl/tmp is excluded by design (transient scratch)." >> "$MAN"
echo
echo "=== manifest"
cat "$MAN"
[ "$fail" -eq 0 ] && echo "ALL ARCHIVES VERIFIED" || echo "SOME ARCHIVES FAILED - DO NOT DELETE ANYTHING"
exit "$fail"
