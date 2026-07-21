import http from "node:http";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Broker } from "../broker/Broker";
import { logger } from "../logger/logger";
import { AppConfig, TradeSignal } from "../types";
import { SignalDecision } from "../strategy/signalEngine";

export interface DashboardState {
  lastDecision?: SignalDecision;
  lastSignal?: TradeSignal;
  dailyPnl: number;
  regime?: {
    hmmState: number;
    hmmLabel: string;
    hmmConfidence: number;
    rules: string;
    updatedAt: number;
  };
  forecast?: {
    candles: Array<{ t: number; o: number; h: number; l: number; c: number }>;
    projection: Array<{ t: number; mid: number; upper: number; lower: number }>;
    direction: string;
    confidence: number;
    atr: number;
    updatedAt: number;
  };
}

/**
 * Validate an HTTP Basic Authorization header against the configured password.
 * Any username is accepted — only the password matters. No password configured
 * means auth is disabled (localhost use). Hash-then-compare keeps the check
 * constant-time regardless of input length.
 */
export function basicAuthOk(authorizationHeader: string | undefined, password: string | undefined): boolean {
  if (!password) return true;
  if (!authorizationHeader?.startsWith("Basic ")) return false;
  let supplied: string;
  try {
    const decoded = Buffer.from(authorizationHeader.slice(6), "base64").toString("utf8");
    supplied = decoded.slice(decoded.indexOf(":") + 1);
  } catch {
    return false;
  }
  const a = crypto.createHash("sha256").update(supplied).digest();
  const b = crypto.createHash("sha256").update(password).digest();
  return crypto.timingSafeEqual(a, b);
}

type SqliteRow = Record<string, unknown>;
type SqliteDb = {
  prepare(sql: string): { all(...params: unknown[]): unknown };
  close(): void;
};

interface BacktestRef {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  maxDrawdown: number;
}

export class DashboardServer {
  private server?: http.Server;
  private readonly startedAt = Date.now();
  private readonly equitySeries: Array<{ t: number; balance: number }> = [];
  private backtestRef?: BacktestRef;

  constructor(
    private readonly config: AppConfig,
    private readonly broker: Broker,
    private readonly state: DashboardState
  ) {
    try {
      const raw = JSON.parse(fs.readFileSync("backtest-baseline.json", "utf8"));
      this.backtestRef = {
        totalTrades: raw.totalTrades,
        winRate: raw.winRate,
        profitFactor: raw.profitFactor,
        expectancy: raw.expectancy,
        maxDrawdown: raw.maxDrawdown
      };
    } catch {
      this.backtestRef = undefined;
    }
  }

  start(): void {
    if (!this.config.dashboard.enabled || this.server) return;

    this.server = http.createServer(async (req, res) => {
      try {
        if (!basicAuthOk(req.headers.authorization, this.config.dashboard.password)) {
          res.writeHead(401, {
            "WWW-Authenticate": 'Basic realm="XAUUSD Bot Dashboard", charset="UTF-8"',
            "Content-Type": "text/plain"
          });
          res.end("Authentication required");
          return;
        }
        const url = req.url ?? "/";
        if (url.startsWith("/api/state")) {
          const payload = await this.buildState();
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(payload));
          return;
        }
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(PAGE_HTML);
      } catch (error) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
      }
    });

    this.server.listen(this.config.dashboard.port, this.config.dashboard.host, () => {
      logger.info("Dashboard started", {
        url: `http://${this.config.dashboard.host}:${this.config.dashboard.port}`
      });
    });
  }

  stop(): void {
    this.server?.close();
    this.server = undefined;
  }

  private async buildState(): Promise<Record<string, unknown>> {
    const [openTrades, balance, history] = await Promise.all([
      this.broker.getOpenPositions().catch(() => []),
      this.broker.getBalance().catch(() => 0),
      this.broker.getTradeHistory().catch(() => [])
    ]);

    const last = this.equitySeries[this.equitySeries.length - 1];
    if (!last || last.balance !== balance) {
      this.equitySeries.push({ t: Date.now(), balance });
      if (this.equitySeries.length > 1000) this.equitySeries.shift();
    }

    return {
      symbol: this.config.symbol,
      brokerMode: this.config.broker.mode,
      aiMode: this.config.strategy.aiMode ?? false,
      aiThreshold: this.config.strategy.aiConfidenceThreshold ?? null,
      aiModelPath: this.config.strategy.aiModelPath ?? null,
      riskPerTrade: this.config.risk.riskPerTrade,
      balance,
      dailyPnl: this.state.dailyPnl,
      openTrades,
      history,
      lastDecision: this.state.lastDecision ?? null,
      lastSignal: this.state.lastSignal ?? null,
      regime: this.state.regime ?? null,
      forecast: this.state.forecast ?? null,
      recentSignals: this.querySignals(120),
      recentTrades: this.queryTrades(20),
      equitySeries: this.equitySeries,
      backtestRef: this.backtestRef ?? null,
      uptimeMs: Date.now() - this.startedAt,
      timestamp: Date.now()
    };
  }

  private openDb(): SqliteDb | undefined {
    if (!this.config.journal.enabled) return undefined;
    try {
      const sqlite = require("node:sqlite") as { DatabaseSync: new (f: string) => SqliteDb };
      return new sqlite.DatabaseSync(path.resolve(this.config.journal.path));
    } catch {
      return undefined;
    }
  }

  private querySignals(limit: number): SqliteRow[] {
    const db = this.openDb();
    if (!db) return [];
    try {
      return db.prepare(`
        SELECT time, direction, score, required_score, allowed, status, blocked_by_json
        FROM signals ORDER BY time DESC LIMIT ?
      `).all(limit) as SqliteRow[];
    } catch {
      return [];
    } finally {
      db.close();
    }
  }

  private queryTrades(limit: number): SqliteRow[] {
    const db = this.openDb();
    if (!db) return [];
    try {
      return db.prepare(`
        SELECT open_time, direction, volume, entry, stop_loss, score, setup_type
        FROM trades ORDER BY open_time DESC LIMIT ?
      `).all(limit) as SqliteRow[];
    } catch {
      return [];
    } finally {
      db.close();
    }
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   Single-page dashboard. Client JS deliberately avoids template literals so
   this file's outer template literal needs no escaping.
──────────────────────────────────────────────────────────────────────────── */
const PAGE_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAUUSD BOT · LIVE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Doto:wght@700;900&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#f2f1ec; --card:#fbfaf6; --line:#dcd9cf; --ink:#22221e; --dim:#8b887c;
    --green:#157f3d; --green-soft:#e2f0e6; --red:#b23b2e; --red-soft:#f4e3e0;
    --mono:'Space Mono',ui-monospace,Menlo,monospace;
    --dot:'Doto','Space Mono',ui-monospace,monospace;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--ink); font-family:var(--mono); padding:18px; }
  .wrap { max-width:1180px; margin:0 auto; display:flex; flex-direction:column; gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
  .label { font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); }
  .pill { display:inline-block; border:1px solid var(--line); border-radius:6px; padding:4px 10px;
          font-size:10px; letter-spacing:.12em; text-transform:uppercase; background:#fff; }
  .pill.live { background:var(--green-soft); color:var(--green); border-color:#bcd9c4; }
  .pill.warn { background:var(--red-soft); color:var(--red); border-color:#dcb9b3; }
  .row { display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; }
  .grow { flex:1 1 340px; }
  /* header */
  .topbar { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  .brandbox { width:44px; height:44px; border:1px solid var(--line); border-radius:10px; background:#fff;
              display:flex; align-items:center; justify-content:center; font-size:18px; }
  .brand-title { font-size:20px; font-weight:700; letter-spacing:.04em; }
  .brand-title .accent { color:var(--red); }
  .brand-sub { font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); margin-bottom:3px; }
  /* hero */
  .bignum { font-family:var(--dot); font-weight:900; font-size:64px; line-height:1; color:var(--ink); }
  .bignum.neg { color:var(--red); }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  .chip { border:1px solid var(--line); background:#fff; border-radius:6px; padding:6px 10px; font-size:11px; }
  .chip b { font-size:12px; }
  .chip .up { color:var(--green); } .chip .down { color:var(--red); }
  /* stats grid inside panels */
  .stats { display:grid; grid-template-columns:auto auto; gap:6px 18px; font-size:12px; align-content:start; }
  .stats .label { align-self:center; }
  .stats .v { text-align:right; font-weight:700; }
  .v.up { color:var(--green); } .v.down { color:var(--red); }
  /* histogram */
  .histo { display:flex; align-items:flex-end; gap:6px; height:150px; padding:8px 4px 0; }
  .hcol { flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; gap:4px; }
  .hbar { width:100%; border-radius:3px 3px 0 0; background:#b9b6aa; min-height:2px; }
  .hbar.win { background:var(--green); opacity:.85; }
  .hlab { font-size:9px; color:var(--dim); white-space:nowrap; }
  .divline { border-left:1px dashed #9a9789; align-self:stretch; }
  /* meter */
  .meter { height:10px; border:1px solid var(--line); border-radius:5px; background:#fff; position:relative; overflow:hidden; margin-top:6px; }
  .meter .fill { position:absolute; left:0; top:0; bottom:0; background:var(--green); opacity:.8; }
  .meter .thr { position:absolute; top:-2px; bottom:-2px; width:2px; background:var(--red); }
  table { width:100%; border-collapse:collapse; font-size:11px; }
  th { text-align:left; font-size:9px; letter-spacing:.12em; text-transform:uppercase; color:var(--dim);
       border-bottom:1px solid var(--line); padding:4px 6px; }
  td { padding:5px 6px; border-bottom:1px solid #eceae2; }
  td.up { color:var(--green); font-weight:700; } td.down { color:var(--red); font-weight:700; }
  .reasons { font-size:11px; line-height:1.7; color:var(--ink); }
  .reasons li { list-style:none; padding-left:14px; position:relative; }
  .reasons li:before { content:"·"; position:absolute; left:2px; color:var(--dim); }
  .footer { display:flex; gap:16px; flex-wrap:wrap; font-size:9px; letter-spacing:.12em;
            text-transform:uppercase; color:var(--dim); padding:10px 4px 0; }
  .footer .ok { color:var(--green); }
  svg.spark { width:100%; height:120px; }
  .panel-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; gap:8px; flex-wrap:wrap; }
  .panel-title { font-size:13px; font-weight:700; }
  .panel-note { font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--dim); }
  @media (max-width:700px){ .bignum{font-size:44px;} }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar card">
    <div style="display:flex; gap:12px; align-items:center;">
      <div class="brandbox">◤</div>
      <div>
        <div class="brand-sub" id="hdr-sub">AUTONOMOUS · AI MODE · CONNECTING…</div>
        <div class="brand-title">XAUUSD BOT · <span class="accent">GOLDFISH</span></div>
      </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <span class="pill live" id="pill-live">● LIVE · DEMO</span>
      <span class="pill" id="pill-mode">MODE · —</span>
      <span class="pill" id="pill-regime">REGIME · —</span>
      <span class="pill" id="clock">--:--:-- UTC</span>
    </div>
  </div>

  <div class="row">
    <div class="card grow" style="flex:2 1 420px;">
      <div class="label">ACCOUNT BALANCE · <span id="lbl-sym">GOLD</span></div>
      <div class="bignum" id="balance" style="margin-top:8px;">$—</div>
      <div class="chips" id="hero-chips"></div>
    </div>
    <div class="card grow">
      <div class="panel-head">
        <span class="panel-title">★ EQUITY · SESSION</span>
        <span class="panel-note" id="equity-note">BALANCE SAMPLES</span>
      </div>
      <svg class="spark" id="spark" viewBox="0 0 300 120" preserveAspectRatio="none"></svg>
    </div>
  </div>

  <div class="card">
    <div class="panel-head">
      <span class="panel-title">● 5M Chart · Model Projection</span>
      <span class="panel-note" id="fc-note">WAITING FOR FIRST CANDLE…</span>
    </div>
    <svg id="fc-svg" viewBox="0 0 1140 240" preserveAspectRatio="none" style="width:100%; height:240px;"></svg>
  </div>

  <div class="card">
    <div class="panel-head">
      <span class="panel-title">● Probability Lattice</span>
      <span class="panel-note" id="lattice-note">EVERY SIGNAL · CONFIDENCE VS THRESHOLD · EDGE NEEDS REPETITION</span>
    </div>
    <div class="row">
      <div class="stats" id="lattice-stats" style="flex:0 0 220px;"></div>
      <div style="flex:1 1 380px;">
        <div class="histo" id="histo"></div>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="card grow">
      <div class="panel-head">
        <span class="panel-title">● Live Signal · Neural Net</span>
        <span class="panel-note" id="sig-time">—</span>
      </div>
      <div class="stats" id="sig-stats"></div>
      <div class="meter"><div class="fill" id="conf-fill" style="width:0%"></div><div class="thr" id="conf-thr" style="left:35%"></div></div>
      <ul class="reasons" id="sig-reasons" style="margin-top:10px;"></ul>
    </div>
    <div class="card grow">
      <div class="panel-head">
        <span class="panel-title">● Open Positions</span>
        <span class="panel-note" id="open-note">0 OPEN</span>
      </div>
      <table><thead><tr><th>DIR</th><th>VOL</th><th>ENTRY</th><th>SL</th><th>OPENED</th><th>PNL</th></tr></thead>
      <tbody id="open-body"></tbody></table>
      <div class="panel-head" style="margin-top:14px;">
        <span class="panel-title">● Closed Trades · Today</span>
        <span class="panel-note" id="closed-note">—</span>
      </div>
      <table><thead><tr><th>CLOSED</th><th>DIR</th><th>VOL</th><th>PNL</th><th>RESULT</th></tr></thead>
      <tbody id="closed-body"></tbody></table>
      <div class="panel-head" style="margin-top:14px;">
        <span class="panel-title">● Recent Signals</span>
      </div>
      <table><thead><tr><th>TIME</th><th>DIR</th><th>CONF</th><th>STATUS</th></tr></thead>
      <tbody id="sig-body"></tbody></table>
    </div>
  </div>

  <div class="card">
    <div class="panel-head">
      <span class="panel-title">● Holdout Backtest Reference · what to expect</span>
      <span class="panel-note">7-MONTH OUT-OF-SAMPLE · SAME COSTS · SEED NOISE ±0.03R</span>
    </div>
    <div class="chips" id="ref-chips"></div>
  </div>

  <div class="footer" id="footer">
    <span class="ok">● AGENT CONNECTING</span>
  </div>
</div>

<script>
(function () {
  "use strict";
  var THRESHOLD = 0.35;

  function $(id) { return document.getElementById(id); }
  function fmt(n, d) { return (n === null || n === undefined || isNaN(n)) ? "—" : Number(n).toFixed(d === undefined ? 2 : d); }
  function money(n) { return "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 2 }); }
  function utc(ts) {
    var x = new Date(ts);
    return ("0" + x.getUTCHours()).slice(-2) + ":" + ("0" + x.getUTCMinutes()).slice(-2);
  }
  function chip(label, value, cls) {
    return '<span class="chip">' + label + ' <b class="' + (cls || "") + '">' + value + "</b></span>";
  }

  setInterval(function () {
    var n = new Date();
    $("clock").textContent =
      ("0" + n.getUTCHours()).slice(-2) + ":" + ("0" + n.getUTCMinutes()).slice(-2) + ":" +
      ("0" + n.getUTCSeconds()).slice(-2) + " UTC";
  }, 1000);

  function renderSpark(series) {
    var svg = $("spark");
    if (!series || series.length < 2) { svg.innerHTML = '<text x="8" y="60" font-size="10" fill="#8b887c">COLLECTING BALANCE SAMPLES…</text>'; return; }
    var vals = series.map(function (p) { return p.balance; });
    var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
    var span = (max - min) || 1;
    var pts = "";
    for (var i = 0; i < vals.length; i++) {
      var x = (i / (vals.length - 1)) * 296 + 2;
      var y = 112 - ((vals[i] - min) / span) * 100;
      pts += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
    }
    var up = vals[vals.length - 1] >= vals[0];
    var col = up ? "#157f3d" : "#b23b2e";
    svg.innerHTML =
      '<path d="' + pts + ' L298 118 L2 118 Z" fill="' + col + '" opacity="0.10"></path>' +
      '<path d="' + pts + '" fill="none" stroke="' + col + '" stroke-width="1.6"></path>';
    $("equity-note").textContent = "FIRST " + money(vals[0]) + " · NOW " + money(vals[vals.length - 1]);
  }

  function renderHisto(state) {
    var host = $("histo");
    var stats = $("lattice-stats");
    var hist = state.history || [];
    var html = "", i, b;

    if (hist.length > 0) {
      $("lattice-note").textContent = "EVERY CLOSED TRADE · R MULTIPLE · EDGE NEEDS REPETITION";
      var edges = [-2, -1, -0.5, 0, 0.5, 1, 2, 3, 99];
      var labels = ["<-1R", "-1R", "-0.5", "0", "+0.5", "+1R", "+2R", "+3R+"];
      var bins = [0, 0, 0, 0, 0, 0, 0, 0];
      var wins = 0, totR = 0;
      for (i = 0; i < hist.length; i++) {
        var r = hist[i].realizedR || 0;
        totR += r;
        if (r > 0) wins++;
        for (b = 0; b < 8; b++) { if (r <= edges[b + 1]) { bins[b]++; break; } }
      }
      var mx = Math.max.apply(null, bins) || 1;
      for (b = 0; b < 8; b++) {
        if (b === 4) html += '<div class="divline"></div>';
        html += '<div class="hcol"><div class="hbar ' + (b >= 4 ? "win" : "") + '" style="height:' +
          Math.round((bins[b] / mx) * 120) + 'px"></div><div class="hlab">' + labels[b] + "</div></div>";
      }
      stats.innerHTML =
        '<span class="label">TRADES CLOSED</span><span class="v">' + hist.length + "</span>" +
        '<span class="label">LANDED GREEN</span><span class="v up">' + fmt((wins / hist.length) * 100, 1) + "%</span>" +
        '<span class="label">EV / TRADE</span><span class="v ' + (totR >= 0 ? "up" : "down") + '">' + (totR >= 0 ? "+" : "") + fmt(totR / hist.length, 3) + "R</span>" +
        '<span class="label">TOTAL R</span><span class="v ' + (totR >= 0 ? "up" : "down") + '">' + (totR >= 0 ? "+" : "") + fmt(totR, 1) + "R</span>" +
        '<span class="label">SESSION PNL</span><span class="v">' + money(state.dailyPnl || 0) + "</span>";
    } else {
      var sigs = state.recentSignals || [];
      $("lattice-note").textContent = "NO CLOSED TRADES YET · SHOWING AI CONFIDENCE DISTRIBUTION";
      var cbins = [0, 0, 0, 0, 0, 0, 0, 0];
      var clabels = ["<20", "20", "25", "30", "35", "40", "45", "50+"];
      var allowed = 0, confSum = 0;
      for (i = 0; i < sigs.length; i++) {
        var c = (sigs[i].score || 0);
        confSum += c;
        if (sigs[i].allowed) allowed++;
        var idx = Math.floor((c - 15) / 5);
        if (idx < 0) idx = 0; if (idx > 7) idx = 7;
        cbins[idx]++;
      }
      var cmx = Math.max.apply(null, cbins) || 1;
      for (b = 0; b < 8; b++) {
        if (b === 4) html += '<div class="divline"></div>';
        html += '<div class="hcol"><div class="hbar ' + (b >= 4 ? "win" : "") + '" style="height:' +
          Math.round((cbins[b] / cmx) * 120) + 'px"></div><div class="hlab">' + clabels[b] + "</div></div>";
      }
      stats.innerHTML =
        '<span class="label">SIGNALS SEEN</span><span class="v">' + sigs.length + "</span>" +
        '<span class="label">ABOVE THRESHOLD</span><span class="v up">' + allowed + "</span>" +
        '<span class="label">AVG CONFIDENCE</span><span class="v">' + fmt(sigs.length ? confSum / sigs.length : 0, 1) + "%</span>" +
        '<span class="label">SESSION PNL</span><span class="v">' + money(state.dailyPnl || 0) + "</span>";
    }
    host.innerHTML = html;
  }

  function renderSignal(state) {
    var d = state.lastDecision;
    var stats = $("sig-stats");
    var reasons = $("sig-reasons");
    if (!d) { stats.innerHTML = '<span class="label">STATUS</span><span class="v">WAITING FOR FIRST CYCLE</span>'; return; }
    var fd = d.finalDecision || {};
    var conf = d.score || 0;
    var dir = (fd.direction || "none").toUpperCase();
    stats.innerHTML =
      '<span class="label">DIRECTION</span><span class="v ' + (dir === "LONG" ? "up" : dir === "SHORT" ? "down" : "") + '">' + dir + "</span>" +
      '<span class="label">CONFIDENCE</span><span class="v">' + fmt(conf, 1) + "%</span>" +
      '<span class="label">THRESHOLD</span><span class="v">' + fmt(THRESHOLD * 100, 0) + "%</span>" +
      '<span class="label">DECISION</span><span class="v ' + (fd.allowed ? "up" : "") + '">' + (fd.action || d.status || "—").toUpperCase() + "</span>";
    $("conf-fill").style.width = Math.min(conf, 60) / 60 * 100 + "%";
    $("conf-thr").style.left = (THRESHOLD * 100) / 60 * 100 + "%";
    var html = "";
    var rs = d.reasons || [];
    for (var i = 0; i < Math.min(rs.length, 6); i++) html += "<li>" + rs[i] + "</li>";
    reasons.innerHTML = html;
    $("sig-time").textContent = "LAST CYCLE " + utc(state.timestamp) + " UTC";
  }

  function pnlCell(v) {
    if (v === null || v === undefined || isNaN(v)) return "<td>—</td>";
    var cls = v >= 0 ? "up" : "down";
    return '<td class="' + cls + '">' + (v >= 0 ? "+" : "") + fmt(v, 2) + "</td>";
  }

  function resultCell(v) {
    if (v === null || v === undefined || isNaN(v)) return "<td>—</td>";
    if (v > 0) return '<td class="up">WIN</td>';
    if (v < 0) return '<td class="down">LOSS</td>';
    return "<td>BE</td>";
  }

  function renderTables(state) {
    var open = state.openTrades || [];
    var openPnl = 0, hasPnl = false;
    var html = "", i;
    for (i = 0; i < open.length; i++) {
      var p = open[i];
      if (typeof p.profit === "number") { openPnl += p.profit; hasPnl = true; }
      html += "<tr><td class=" + (p.direction === "long" ? "up" : "down") + ">" + p.direction.toUpperCase() +
        "</td><td>" + p.volume + "</td><td>" + fmt(p.entry) + "</td><td>" + fmt(p.stopLoss) +
        "</td><td>" + utc(p.openedAt) + "</td>" + pnlCell(p.profit) + "</tr>";
    }
    $("open-note").textContent = open.length + " OPEN" +
      (hasPnl ? " · " + (openPnl >= 0 ? "+" : "") + fmt(openPnl, 2) + " FLOATING" : "");
    $("open-body").innerHTML = html || '<tr><td colspan="6" style="color:#8b887c">no open positions</td></tr>';

    var hist = (state.history || []).slice().sort(function (a, b) { return (b.closedAt || 0) - (a.closedAt || 0); });
    var closedPnl = 0, winCount = 0, lossCount = 0;
    html = "";
    for (i = 0; i < Math.min(hist.length, 12); i++) {
      var h = hist[i];
      html += "<tr><td>" + utc(h.closedAt) + "</td><td class=" + (h.direction === "long" ? "up" : "down") + ">" +
        h.direction.toUpperCase() + "</td><td>" + h.volume + "</td>" + pnlCell(h.profit) + resultCell(h.profit) + "</tr>";
    }
    for (i = 0; i < hist.length; i++) {
      var hp = hist[i].profit || 0;
      closedPnl += hp;
      if (hp > 0) winCount++; else if (hp < 0) lossCount++;
    }
    $("closed-note").textContent = hist.length === 0 ? "NONE YET" :
      hist.length + " CLOSED · " + winCount + "W " + lossCount + "L · " +
      (closedPnl >= 0 ? "+" : "") + fmt(closedPnl, 2) + " TOTAL";
    $("closed-body").innerHTML = html || '<tr><td colspan="5" style="color:#8b887c">no closed trades today</td></tr>';

    var sigs = (state.recentSignals || []).slice(0, 10);
    html = "";
    for (i = 0; i < sigs.length; i++) {
      var s = sigs[i];
      var st = s.allowed ? "TRADE" : (s.status || "wait").toUpperCase();
      html += "<tr><td>" + utc(s.time) + "</td><td class=" + (s.direction === "long" ? "up" : s.direction === "short" ? "down" : "") + ">" +
        (s.direction || "—").toUpperCase() + "</td><td>" + fmt(s.score, 1) + "%</td><td>" + st + "</td></tr>";
    }
    $("sig-body").innerHTML = html || '<tr><td colspan="4" style="color:#8b887c">no signals yet</td></tr>';
  }

  function renderForecast(state) {
    var svg = $("fc-svg");
    var fc = state.forecast;
    if (!fc || !fc.candles || fc.candles.length < 2) {
      svg.innerHTML = '<text x="10" y="30" font-size="11" fill="#8b887c">waiting for the first evaluated candle…</text>';
      return;
    }
    var W = 1140, H = 240, PAD = 8;
    var candles = fc.candles, proj = fc.projection || [];
    var n = candles.length, m = proj.length, slots = n + m, i;
    var lo = Infinity, hi = -Infinity;
    for (i = 0; i < n; i++) { if (candles[i].l < lo) lo = candles[i].l; if (candles[i].h > hi) hi = candles[i].h; }
    for (i = 0; i < m; i++) { if (proj[i].lower < lo) lo = proj[i].lower; if (proj[i].upper > hi) hi = proj[i].upper; }
    var span = (hi - lo) || 1;
    function X(idx) { return PAD + (idx + 0.5) * ((W - 2 * PAD) / slots); }
    function Y(p) { return PAD + (1 - (p - lo) / span) * (H - 2 * PAD); }
    var w = Math.max(1.5, (W - 2 * PAD) / slots * 0.55);
    var html = "";
    for (i = 0; i < n; i++) {
      var c = candles[i];
      var col = c.c >= c.o ? "#157f3d" : "#b23b2e";
      var x = X(i);
      html += '<line x1="' + x + '" y1="' + Y(c.h) + '" x2="' + x + '" y2="' + Y(c.l) + '" stroke="' + col + '" stroke-width="1"/>';
      var top = Y(Math.max(c.o, c.c)), bot = Y(Math.min(c.o, c.c));
      html += '<rect x="' + (x - w / 2) + '" y="' + top + '" width="' + w + '" height="' + Math.max(1, bot - top) + '" fill="' + col + '"/>';
    }
    if (m > 0) {
      var dcol = fc.direction === "long" ? "#157f3d" : fc.direction === "short" ? "#b23b2e" : "#8b887c";
      var pts = "";
      for (i = 0; i < m; i++) pts += X(n + i) + "," + Y(proj[i].upper) + " ";
      for (i = m - 1; i >= 0; i--) pts += X(n + i) + "," + Y(proj[i].lower) + " ";
      html += '<polygon points="' + pts + '" fill="' + dcol + '" opacity="0.10"/>';
      var d = "M" + X(n - 1) + " " + Y(candles[n - 1].c);
      for (i = 0; i < m; i++) d += " L" + X(n + i) + " " + Y(proj[i].mid);
      html += '<path d="' + d + '" fill="none" stroke="' + dcol + '" stroke-width="1.6" stroke-dasharray="5 4"/>';
      var dx = (X(n - 1) + X(n)) / 2;
      html += '<line x1="' + dx + '" y1="0" x2="' + dx + '" y2="' + H + '" stroke="#9a9789" stroke-dasharray="3 4" stroke-width="1"/>';
    }
    svg.innerHTML = html;
    $("fc-note").textContent =
      (fc.direction === "none" ? "NO LEAN" : fc.direction.toUpperCase() + " LEAN " + fmt(fc.confidence * 100, 0) + "%") +
      " · NEXT " + m + " BARS · CONE ±1 ATR·√t · MODEL LEAN, NOT A LITERAL PATH";
  }

  function renderRef(state) {
    var r = state.backtestRef;
    if (!r) { $("ref-chips").innerHTML = '<span class="chip">backtest-baseline.json not found</span>'; return; }
    $("ref-chips").innerHTML =
      chip("EXPECTANCY", "+" + fmt(r.expectancy, 3) + "R / trade", "up") +
      chip("WIN RATE", fmt(r.winRate * 100, 1) + "%") +
      chip("PROFIT FACTOR", fmt(r.profitFactor, 2)) +
      chip("MAX DRAWDOWN", fmt(r.maxDrawdown * 100, 1) + "%", "down") +
      chip("TRADES", r.totalTrades) +
      chip("RULE", "judge live vs these after ~100 trades");
  }

  function refresh() {
    fetch("/api/state").then(function (r) { return r.json(); }).then(function (state) {
      if (state.aiThreshold) THRESHOLD = state.aiThreshold;
      $("hdr-sub").textContent = "AUTONOMOUS · " + (state.aiMode ? "AI MODE" : "RULES MODE") +
        " · LIVE ON " + (state.brokerMode || "?").toUpperCase() + " · RISK " + fmt(state.riskPerTrade * 100, 1) + "%";
      $("pill-mode").textContent = "MODE · " + (state.aiMode ? "AI" : "RULES");
      var rg = state.regime;
      if (rg && rg.hmmLabel && rg.hmmLabel !== "unknown") {
        $("pill-regime").textContent = "REGIME · HMM " + rg.hmmLabel.toUpperCase() + " " +
          fmt(rg.hmmConfidence * 100, 0) + "% · " + (rg.rules || "").toUpperCase();
      } else {
        $("pill-regime").textContent = "REGIME · " + (rg && rg.rules ? rg.rules.toUpperCase() : "—");
      }
      $("lbl-sym").textContent = state.symbol || "GOLD";
      var bal = $("balance");
      bal.textContent = money(state.balance);
      bal.className = "bignum" + (state.dailyPnl < 0 ? " neg" : "");
      $("hero-chips").innerHTML =
        chip("SESSION PNL", (state.dailyPnl >= 0 ? "+" : "") + money(state.dailyPnl), state.dailyPnl >= 0 ? "up" : "down") +
        chip("OPEN", (state.openTrades || []).length) +
        chip("SIGNALS", (state.recentSignals || []).length) +
        chip("CLOSED TRADES", (state.history || []).length) +
        chip("UPTIME", Math.floor(state.uptimeMs / 3600000) + "h " + Math.floor((state.uptimeMs % 3600000) / 60000) + "m");
      renderSpark(state.equitySeries);
      renderForecast(state);
      renderHisto(state);
      renderSignal(state);
      renderTables(state);
      renderRef(state);
      $("footer").innerHTML =
        '<span class="ok">● AGENT ONLINE</span>' +
        "<span>MODEL " + (state.aiModelPath || "—") + "</span>" +
        "<span>BROKER " + (state.brokerMode || "—").toUpperCase() + "</span>" +
        "<span>THRESHOLD " + fmt(THRESHOLD * 100, 0) + "%</span>" +
        "<span>REFRESH 5S</span>";
      $("pill-live").textContent = "● LIVE · DEMO";
    }).catch(function () {
      $("pill-live").textContent = "● DISCONNECTED";
      $("pill-live").className = "pill warn";
      $("footer").innerHTML = '<span style="color:#b23b2e">● AGENT OFFLINE — is the bot running?</span>';
    });
  }

  refresh();
  setInterval(refresh, 5000);
})();
</script>
</body>
</html>`;
