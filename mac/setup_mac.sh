#!/bin/bash
# One-time setup for the XAUUSD TypeScript trading bot on a new Mac.
# Run from Terminal:  cd ~/Desktop/XAUUSD_TradingBot && bash setup_mac.sh
set -e
cd "$(dirname "$0")/.."

echo "==> 1/5 Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew (you may be asked for your Mac password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for Apple Silicon Macs
  if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  fi
else
  echo "Homebrew already installed."
fi

echo "==> 2/5 Checking Node.js..."
if ! command -v node >/dev/null 2>&1; then
  brew install node
else
  echo "Node $(node -v) already installed."
fi

echo "==> 3/5 Installing project dependencies..."
rm -rf node_modules   # clear any partial install
npm install

echo "==> 4/5 Creating .env from template..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env (edit it to change settings; DRY_RUN=true by default)."
else
  echo ".env already exists, leaving it alone."
fi

echo "==> 5/5 Building and running tests..."
npm run build
npm test

echo ""
echo "Done. Start the bot (mock broker, no real trades) with:"
echo "  npm start"
echo ""
echo "Note: the Python files (main.py etc.) need the MetaTrader5 package,"
echo "which only works on Windows — they can't run on a Mac. The TypeScript"
echo "app is the one that runs here."
