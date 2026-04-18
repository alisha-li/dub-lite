#!/bin/bash
# Installs the Dub Lite native messaging host for Chrome/Arc/Brave

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_NAME="com.dub_lite.host"

# Make host.py executable
chmod +x "$SCRIPT_DIR/host.py"

# Update the manifest with the correct absolute path to host.py
MANIFEST="$SCRIPT_DIR/$HOST_NAME.json"

# Detect the extension ID from the manifest
# For now, use a placeholder — we'll update this after loading the extension
echo ""
echo "=== Dub Lite Native Host Installer ==="
echo ""

# Check for yt-dlp
if ! command -v yt-dlp &> /dev/null; then
    echo "ERROR: yt-dlp is not installed."
    echo "Install it with: pip install yt-dlp"
    echo "Or: brew install yt-dlp"
    exit 1
fi
echo "Found yt-dlp: $(which yt-dlp)"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed."
    exit 1
fi
echo "Found python3: $(which python3)"

# Update the path in the manifest to point to this directory's host.py
python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
m['path'] = '$SCRIPT_DIR/host.py'
with open('$MANIFEST', 'w') as f:
    json.dump(m, f, indent=2)
"

# Chrome looks for native messaging host manifests in this directory on macOS
# Works for Chrome, Arc, Brave, and other Chromium browsers
CHROME_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
ARC_DIR="$HOME/Library/Application Support/Arc/User Data/NativeMessagingHosts"
BRAVE_DIR="$HOME/Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts"
CHROMIUM_DIR="$HOME/Library/Application Support/Chromium/NativeMessagingHosts"

# Install for all detected browsers
INSTALLED=0
for DIR in "$CHROME_DIR" "$ARC_DIR" "$BRAVE_DIR" "$CHROMIUM_DIR"; do
    BROWSER_BASE="$(dirname "$DIR")"
    if [ -d "$BROWSER_BASE" ]; then
        mkdir -p "$DIR"
        cp "$MANIFEST" "$DIR/$HOST_NAME.json"
        echo "Installed for: $DIR"
        INSTALLED=1
    fi
done

if [ $INSTALLED -eq 0 ]; then
    echo "WARNING: No Chromium browser detected. Installing for Chrome anyway."
    mkdir -p "$CHROME_DIR"
    cp "$MANIFEST" "$CHROME_DIR/$HOST_NAME.json"
fi

echo ""
echo "Done! Restart your browser for changes to take effect."
echo ""
