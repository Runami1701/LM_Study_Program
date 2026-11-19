#!/usr/bin/env bash
# Set up the course on a Raspberry Pi (or any Debian/Ubuntu machine) as a service
# that starts on boot and can be opened from other devices on your home network.
#
#   cd LM_Study_Program && ./scripts/pi_setup.sh          (port 8000)
#   PORT=8080 ./scripts/pi_setup.sh                         (another port)
#
# Safe to run again: after a `git pull` it reinstalls and restarts the service.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PORT="${PORT:-8000}"
SERVICE=lm-course
RUN_USER="$(id -un)"

say() { printf '[pi-setup] %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
    say "Run this as your normal user (for example 'pi'), not with sudo. It asks for sudo itself."
    exit 1
fi

if [ "$(uname -m)" != "aarch64" ]; then
    say "Warning: this is a 32-bit system ($(uname -m)). The course works, but a 64-bit Raspberry Pi OS"
    say "installs numpy faster and trains faster. Continuing."
fi

say "Installing system packages (python3-venv, git)."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git avahi-daemon

say "Creating the Python environment in .venv and installing the course."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e .

if [ ! -f data/tinyshakespeare.txt ]; then
    say "Downloading the training text."
    .venv/bin/python -m course data shakespeare || say "Download failed; the Download button in the course retries it."
fi
.venv/bin/python -m course build

say "Installing the service ($SERVICE) so the course starts on boot."
sudo tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=NumPy LM course website
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
ExecStart=$ROOT/.venv/bin/python -m course serve --host 0.0.0.0 --port $PORT --no-browser
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q $SERVICE
sudo systemctl restart $SERVICE

for _ in $(seq 1 30); do [ -s access.txt ] && break; sleep 1; done
if ! systemctl is-active -q $SERVICE; then
    say "The service did not start. See why with:  journalctl -u $SERVICE -n 50"
    exit 1
fi

IP="$(hostname -I | awk '{print $1}')"
echo
say "Done. Open the course from any device on your home network:"
say "    http://$(hostname).local:$PORT/     or     http://$IP:$PORT/"
say "Passcode: $(cat access.txt)   (saved in $ROOT/access.txt)"
echo
say "Useful commands:"
say "    sudo systemctl restart $SERVICE     restart (after a git pull, or run this script again)"
say "    sudo systemctl stop $SERVICE        stop until the next boot"
say "    journalctl -u $SERVICE -f           show its output"
