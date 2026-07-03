# Running the Bot with Real MT5 on a Mac (Windows VM)

Goal: a free Windows virtual machine on your Mac that runs MetaTrader 5,
the Python bridge, and the bot — exactly like a normal Windows PC.

Time: ~1–2 hours (mostly waiting on downloads). Cost: $0.

Requirements: Apple Silicon Mac (M1–M4), ~50 GB free disk, 16 GB RAM
recommended (8 GB minimum).

---

## 1. Install VMware Fusion (free)

1. Go to https://support.broadcom.com and create a free account.
2. Download **VMware Fusion Pro** (free for personal use since 2024).
3. Install it like any Mac app.

Alternative: **UTM** (https://mac.getutm.app, also free) works too, but
Fusion handles Windows 11 setup and shared folders with less fiddling.

## 2. Install Windows 11 ARM

1. In Fusion: File → New → **Get Windows from Microsoft** — Fusion
   downloads Windows 11 ARM and creates the VM for you.
   (Manual route: download the Windows 11 ARM64 ISO from
   https://www.microsoft.com/software-download/windows11arm64.)
2. Give the VM: **4 CPU cores, 8 GB RAM, 60 GB disk** (Settings →
   Processors & Memory / Hard Disk).
3. Boot it and go through Windows setup. You can skip the product key —
   Windows runs fine unactivated for this purpose (watermark only).
4. Install **VMware Tools** when prompted (needed for shared folders,
   clipboard, resolution).

Note: Windows 11 ARM runs Intel programs (MT5, Python x64) through its
built-in emulation. Everything below works normally, just slightly slower
than native.

## 3. Inside Windows: install the stack

Do all of this **inside the VM**.

### MetaTrader 5
1. Download MT5 from your broker's website (important: the broker build,
   so your GOLD symbol and server list are right) and install.
2. Log in to your **demo account**.
3. Enable algo trading: Tools → Options → Expert Advisors →
   ✅ "Allow algorithmic trading". Also click the **Algo Trading** button
   in the toolbar so it's green.

### Python (must be x64, not ARM)
1. Download **Python 3.11 or 3.12, Windows 64-bit installer** from
   https://www.python.org/downloads/windows/ — pick
   "Windows installer (64-bit)". The MetaTrader5 package has no ARM build,
   but x64 Python runs fine on Windows ARM.
2. During install: ✅ **"Add python.exe to PATH"**.
3. Verify in Command Prompt:
   ```
   python --version
   ```

### Node.js
1. Download the Windows x64 installer from https://nodejs.org (LTS).
2. Install with defaults. Verify: `node --version`.

### Git (optional but handy)
https://git-scm.com/download/win — or skip it and use a shared folder
(step 4) to copy the project in.

## 4. Get the project into the VM

Option A — shared folder (no git needed):
1. VM Settings → Sharing → enable, add `~/Desktop/XAUUSD_TradingBot`.
2. In Windows it appears under `\\vmware-host\Shared Folders\`.
3. **Copy** the folder to `C:\bot` (don't run it from the share — SQLite
   and npm are unhappy on network shares).

Option B — git clone your repo into `C:\bot`.

## 5. Install and configure

In Command Prompt:

```
cd C:\bot
pip install -r requirements.txt
npm install
copy .env.example .env
```

Edit `.env` (Notepad) — make sure:

```
BROKER_MODE=mt5
```

(or just delete the BROKER_MODE line; mt5 is the default in
config/default.json).

Check `config/default.json` → `"mt5"` section: `"pythonPath": "python"`
is correct if Python is on PATH. Set `"dryRun": true` for the first runs —
the bot checks orders without sending them.

## 6. Run

**MT5 must be open and logged in** (same as your old Windows setup), then:

```
cd C:\bot
npm run build
npm start
```

You should see "MT5 broker connected through Python bridge" instead of
the `spawn python ENOENT` error you got on macOS.

When dry runs look right, set `"dryRun": false` in config/default.json to
let it place real demo orders.

## 7. Day-to-day

- The bot only trades while the **VM is running, MT5 is open, and the Mac
  is awake**. For sessions you care about: Mac System Settings → Displays →
  Advanced → prevent sleeping, or just run it while you're at the machine.
- Suspend the VM (not shut down) to resume in seconds.
- Data collection for AI training also works here:
  `python scripts\ai_collect.py --multi --years 3` — then copy the
  `data\*.csv` files back to the Mac via the shared folder and train on
  the Mac with numpy.

## Troubleshooting

| Problem | Fix |
|---|---|
| `spawn python ENOENT` | Python not on PATH — reinstall with the PATH checkbox, or set the full path in config/default.json → mt5.pythonPath |
| `ImportError: MetaTrader5` | You installed ARM Python — uninstall, install the **64-bit (x64)** build |
| Bridge connects but no candles | MT5 not logged in, or wrong symbol name — check the Market Watch symbol matches SYMBOL in .env (GOLD vs XAUUSD, broker-dependent) |
| Orders rejected | Algo Trading button off in MT5, or demo account has no trading rights |
| Everything is slow | Give the VM more cores/RAM; close Chrome in the VM |
