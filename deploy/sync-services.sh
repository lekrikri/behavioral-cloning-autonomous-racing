#!/usr/bin/env bash
# sync-services.sh — installe / met à jour les units systemd robocar depuis CE repo,
# puis recharge systemd et relance les services.
#
# Le WorkingDirectory installé pointe sur le repo d'où ce script est lancé :
#   - prod : lancer depuis ~/robocar-Paris-PGE_MSC (clone sur main) → services sur main
#   - dev  : lancer depuis son clone de dev → services sur le code de dev (pour tester),
#            puis re-sync depuis le clone main pour revenir en prod.
#
# Remplace l'ancien robocar-update.sh (git pull au boot — supprimé). Le déploiement de CODE
# reste explicite (git pull manuel quand on décide) ; ce script ne touche QUE les services.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DEST=/etc/systemd/system
UNITS=(robocar-cam-hub.service robocar-core.service)

echo "[sync] repo courant = $REPO_ROOT"
for u in "${UNITS[@]}"; do
  echo "[sync] install $u -> $DEST/ (WorkingDirectory=$REPO_ROOT)"
  sed "s|^WorkingDirectory=.*|WorkingDirectory=$REPO_ROOT|" "$HERE/systemd/$u" \
    | sudo tee "$DEST/$u" >/dev/null
  sudo chmod 0644 "$DEST/$u"
done

sudo systemctl daemon-reload
sudo systemctl enable "${UNITS[@]}"
sudo systemctl restart "${UNITS[@]}"

echo "[sync] OK — état :"
systemctl --no-pager status "${UNITS[@]}" | grep -E 'Loaded|Active' || true
