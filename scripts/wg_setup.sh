#!/usr/bin/env bash
# Reach the course on this Raspberry Pi from outside your home, through WireGuard.
#
#   ./scripts/wg_setup.sh             set up the WireGuard server (once)
#   ./scripts/wg_setup.sh add phone   add a device: prints a QR code for the WireGuard app
#                                     and saves ~/wg-phone.conf for a laptop
#   ./scripts/wg_setup.sh list        show the devices and when each last connected
#
# The VPN only reaches this Pi: a connected device opens the course at
# http://10.8.0.1:8000/ and SSH at <user>@10.8.0.1. Its other traffic does not go
# through your home.
#
# You must also forward UDP port 51820 on your router to this Pi (the script tells you how).
# Settings: WG_PORT (default 51820), WG_ENDPOINT (public address; default: looked up once).
set -euo pipefail

PORT="${WG_PORT:-51820}"
NET=10.8.0
DIR="${WG_DIR:-/etc/wireguard}"
CONF="$DIR/wg0.conf"
COURSE_PORT="${PORT_COURSE:-8000}"

say() { printf '[wg-setup] %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
    exec sudo --preserve-env=WG_PORT,WG_ENDPOINT,WG_DIR,PORT_COURSE "$0" "$@"
fi
OWNER="${SUDO_USER:-root}"
OWNER_HOME="$(getent passwd "$OWNER" | cut -d: -f6)"

reload() {
    if systemctl is-active -q wg-quick@wg0; then
        wg syncconf wg0 <(wg-quick strip wg0)
    fi
}

setup() {
    if ! command -v wg >/dev/null || ! command -v qrencode >/dev/null; then
        say "Installing WireGuard and qrencode."
        apt-get update -qq
        apt-get install -y -qq wireguard-tools qrencode
    fi
    umask 077
    mkdir -p "$DIR/clients"
    if [ ! -f "$CONF" ]; then
        say "Creating the server keys and $CONF."
        wg genkey | tee "$DIR/server.key" | wg pubkey > "$DIR/server.pub"
        cat > "$CONF" <<EOF
[Interface]
Address = $NET.1/24
ListenPort = $PORT
PrivateKey = $(cat "$DIR/server.key")
EOF
    fi
    if [ -n "${WG_ENDPOINT:-}" ]; then
        echo "$WG_ENDPOINT" > "$DIR/endpoint"
    elif [ ! -s "$DIR/endpoint" ]; then
        say "Looking up this home's public IP address."
        curl -fsS --max-time 10 https://api.ipify.org > "$DIR/endpoint" || {
            say "Could not look it up. Run again with WG_ENDPOINT=<your public IP or DDNS name>."
            exit 1
        }
    fi
    systemctl enable -q wg-quick@wg0
    systemctl restart wg-quick@wg0

    local lan_ip
    lan_ip="$(hostname -I | awk '{print $1}')"
    echo
    say "WireGuard is running on UDP port $PORT. Public address: $(cat "$DIR/endpoint")"
    echo
    say "Now, once, in your router's admin page (often http://${lan_ip%.*}.1/):"
    say "    forward (port-forward / 포트포워드) UDP port $PORT to $lan_ip, port $PORT."
    say "    Also give this Pi a fixed address (DHCP reservation / 고정 IP) of $lan_ip."
    echo
    say "Then add each device:   ./scripts/wg_setup.sh add phone"
}

add() {
    local name="${1:-}"
    if [[ ! "$name" =~ ^[A-Za-z0-9_-]{1,32}$ ]]; then
        say "Give the device a short name (letters, digits, - or _), for example:  add phone"
        exit 1
    fi
    [ -f "$CONF" ] || setup
    if [ -f "$DIR/clients/$name.conf" ]; then
        say "A device called '$name' exists already. Its QR code:"
        qrencode -t ansiutf8 < "$DIR/clients/$name.conf"
        exit 0
    fi
    local i ip=""
    for i in $(seq 2 254); do
        if ! grep -q "AllowedIPs = $NET.$i/32" "$CONF"; then ip="$NET.$i"; break; fi
    done
    [ -n "$ip" ] || { say "No free addresses left."; exit 1; }

    umask 077
    local key pub psk
    key="$(wg genkey)"; pub="$(echo "$key" | wg pubkey)"; psk="$(wg genpsk)"
    cat >> "$CONF" <<EOF

[Peer]
# $name
PublicKey = $pub
PresharedKey = $psk
AllowedIPs = $ip/32
EOF
    cat > "$DIR/clients/$name.conf" <<EOF
[Interface]
PrivateKey = $key
Address = $ip/32

[Peer]
PublicKey = $(cat "$DIR/server.pub")
PresharedKey = $psk
Endpoint = $(cat "$DIR/endpoint"):$PORT
AllowedIPs = $NET.1/32
PersistentKeepalive = 25
EOF
    reload

    local copy="$OWNER_HOME/wg-$name.conf"
    install -m 600 -o "$OWNER" "$DIR/clients/$name.conf" "$copy"

    echo
    say "Added '$name' as $ip."
    say "Phone: install the WireGuard app, press + > scan QR code, and scan this:"
    qrencode -t ansiutf8 < "$DIR/clients/$name.conf"
    say "Laptop: copy $copy to it (for example: scp $OWNER@$(hostname).local:wg-$name.conf .),"
    say "        then in the WireGuard app press 'Import tunnel(s) from file'."
    say "        Delete the copy afterwards; it holds the device's private key."
    echo
    say "With the tunnel on, open the course at  http://$NET.1:$COURSE_PORT/"
}

case "${1:-setup}" in
    setup) setup ;;
    add)   add "${2:-}" ;;
    list)  wg show wg0 ; grep '^# ' "$CONF" | sed 's/^# /device: /' ;;
    *)     sed -n '2,15p' "$0"; exit 1 ;;
esac
