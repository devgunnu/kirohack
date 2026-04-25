import * as vscode from 'vscode';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

const STATE_FILE = path.join(os.homedir(), '.meshrun', 'config.toml');

export function isJoined(): boolean {
    try {
        if (!fs.existsSync(STATE_FILE)) { return false; }
        const content = fs.readFileSync(STATE_FILE, 'utf8');
        return content.includes('joined = true');
    } catch {
        return false;
    }
}

/** Sets the `meshrun.joined` context key so welcome views toggle correctly. */
export function syncJoinedContext(): void {
    vscode.commands.executeCommand('setContext', 'meshrun.joined', isJoined());
}

export class OnboardingPanel {
    public static currentPanel: OnboardingPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;

    public static createOrShow(context: vscode.ExtensionContext): void {
        if (OnboardingPanel.currentPanel) {
            OnboardingPanel.currentPanel._panel.reveal(vscode.ViewColumn.One);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'meshrunOnboarding',
            'MeshRun — Join the Mesh',
            vscode.ViewColumn.One,
            { enableScripts: true }
        );
        OnboardingPanel.currentPanel = new OnboardingPanel(panel, context);
    }

    private constructor(panel: vscode.WebviewPanel, context: vscode.ExtensionContext) {
        this._panel = panel;
        this._panel.webview.html = this._getHtml();

        this._panel.webview.onDidReceiveMessage(async (message) => {
            if (message.command === 'join') {
                const { runMeshrun } = await import('./meshrunProcess');
                const computePct: number = message.data?.computePct ?? 50;
                const model: string = message.data?.model ?? 'qwen2.5-3b';
                try {
                    await runMeshrun([
                        'join',
                        '--non-interactive',
                        '--compute', String(computePct),
                        '--model', model,
                    ]);
                    vscode.window.showInformationMessage('MeshRun: Successfully joined the mesh!');
                    syncJoinedContext();
                    this._panel.dispose();
                    vscode.commands.executeCommand('meshrunNodes.refresh');
                } catch (e: any) {
                    vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
                }
            }
            if (message.command === 'cancel') {
                this._panel.dispose();
            }
        }, null, context.subscriptions);

        this._panel.onDidDispose(() => {
            OnboardingPanel.currentPanel = undefined;
        }, null, context.subscriptions);
    }

    private _getHtml(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MeshRun — Join</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; font-family: system-ui, -apple-system, sans-serif; color: #fff; padding: 32px; min-height: 100vh; }
  .container { max-width: 560px; margin: 0 auto; }
  .logo { font-size: 24px; font-weight: 800; color: #00d4ff; margin-bottom: 8px; letter-spacing: -0.5px; }
  .tagline { font-size: 13px; color: #555; margin-bottom: 32px; }

  /* Step indicator */
  .steps { display: flex; align-items: center; margin-bottom: 32px; }
  .step { display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .step-circle { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; transition: all 0.2s; }
  .step.done .step-circle { background: #00d4ff; color: #000; }
  .step.active .step-circle { background: transparent; border: 2px solid #00d4ff; color: #00d4ff; }
  .step.inactive .step-circle { background: #1a1a1a; border: 2px solid #2a2a2a; color: #444; }
  .step-label { font-size: 10px; color: #555; white-space: nowrap; }
  .step.active .step-label { color: #00d4ff; }
  .step.done .step-label { color: #888; }
  .step-line { flex: 1; height: 2px; background: #222; margin: 0 8px 18px; transition: background 0.3s; }
  .step-line.done { background: #00d4ff; }
</style>
<style>
  /* Cards */
  .card { background: #161616; border: 1px solid #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card-title { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
  .card-sub { font-size: 12px; color: #555; margin-bottom: 20px; }

  /* HW table */
  .hw-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid #1e1e1e; }
  .hw-row:last-child { border-bottom: none; }
  .hw-label { font-size: 12px; color: #666; }
  .hw-val { font-size: 12px; color: #00d4ff; font-weight: 600; }

  /* Compute */
  .compute-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom: 16px; }
  .compute-btn { padding: 14px 0; border-radius: 8px; border: 2px solid #2a2a2a; background: #1a1a1a; color: #666; font-size: 13px; font-weight: 700; cursor: pointer; text-align: center; transition: all 0.15s; }
  .compute-btn:hover { border-color: #00d4ff44; color: #aaa; }
  .compute-btn.selected { border-color: #00d4ff; background: #00d4ff15; color: #00d4ff; }
  .compute-sub { font-size: 10px; color: #444; text-align: center; margin-top: 4px; }
  .earning-bar { background: #00d4ff0a; border: 1px solid #00d4ff22; border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }
  .earning-label { font-size: 11px; color: #555; }
  .earning-val { font-size: 15px; color: #00d4ff; font-weight: 700; }

  /* Models */
  .model-card { border: 2px solid #2a2a2a; border-radius: 10px; padding: 14px 16px; cursor: pointer; margin-bottom: 10px; transition: all 0.15s; background: #1a1a1a; }
  .model-card:hover { border-color: #00d4ff44; }
  .model-card.selected { border-color: #00d4ff; background: #00d4ff0a; }
  .model-name { font-size: 13px; font-weight: 600; }
  .model-card.selected .model-name { color: #00d4ff; }
  .model-meta { font-size: 11px; color: #555; margin-top: 3px; }

  /* Confirm */
  .confirm-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid #1e1e1e; }
  .confirm-row:last-child { border-bottom: none; }
  .confirm-label { font-size: 12px; color: #666; }
  .confirm-val { font-size: 12px; color: #00d4ff; font-weight: 600; }
  .success-bar { background: #22c55e0a; border: 1px solid #22c55e33; border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; margin-top: 16px; }

  /* Nav */
  .nav { display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
  .btn-back { padding: 10px 20px; border-radius: 8px; border: 1px solid #2a2a2a; background: transparent; color: #666; font-size: 13px; cursor: pointer; }
  .btn-back:hover { color: #aaa; border-color: #444; }
  .btn-next { padding: 10px 28px; border-radius: 8px; border: none; background: #00d4ff; color: #000; font-size: 13px; font-weight: 700; cursor: pointer; }
  .btn-next:hover { background: #33ddff; }
  .btn-join { padding: 12px 32px; border-radius: 8px; border: none; background: #22c55e; color: #000; font-size: 14px; font-weight: 700; cursor: pointer; }
  .btn-join:hover { background: #4ade80; }
  .step-counter { font-size: 11px; color: #444; }
</style>
</head>
<body>
<div class="container">
  <div class="logo">MeshRun</div>
  <div class="tagline">Run AI on compute you already own. Earn credits. No cloud costs.</div>

  <div class="steps" id="stepIndicator"></div>

  <!-- Page 0: Hardware -->
  <div id="page0">
    <div class="card">
      <div class="card-title">Hardware Detected</div>
      <div class="card-sub">Here's what MeshRun found on your machine</div>
      <div class="hw-row"><span class="hw-label">GPU</span><span class="hw-val">NVIDIA RTX 4060</span></div>
      <div class="hw-row"><span class="hw-label">VRAM Available</span><span class="hw-val">8.0 GB</span></div>
      <div class="hw-row"><span class="hw-label">RAM</span><span class="hw-val">16.0 GB</span></div>
      <div class="hw-row"><span class="hw-label">Suggested layers</span><span class="hw-val">0 – 6</span></div>
      <div class="hw-row"><span class="hw-label">Model support</span><span class="hw-val" style="color:#22c55e">✓ Compatible</span></div>
    </div>
    <div class="nav">
      <span class="step-counter">Step 1 of 4</span>
      <button class="btn-next" onclick="goTo(1)">Next →</button>
    </div>
  </div>

  <!-- Page 1: Compute -->
  <div id="page1" style="display:none">
    <div class="card">
      <div class="card-title">Compute Allocation</div>
      <div class="card-sub">How much VRAM do you want to contribute?</div>
      <div class="compute-grid">
        <div>
          <div class="compute-btn" onclick="selectCompute(0,this)">25%</div>
          <div class="compute-sub">2.0 GB</div>
        </div>
        <div>
          <div class="compute-btn selected" id="compute-default" onclick="selectCompute(1,this)">50%</div>
          <div class="compute-sub">4.0 GB</div>
        </div>
        <div>
          <div class="compute-btn" onclick="selectCompute(2,this)">75%</div>
          <div class="compute-sub">6.0 GB</div>
        </div>
        <div>
          <div class="compute-btn" onclick="selectCompute(3,this)">100%</div>
          <div class="compute-sub">8.0 GB</div>
        </div>
      </div>
      <div class="earning-bar">
        <span class="earning-label">Earning rate</span>
        <span class="earning-val" id="earningDisplay">+2.4 credits / forward pass</span>
      </div>
    </div>
    <div class="nav">
      <button class="btn-back" onclick="goTo(0)">← Back</button>
      <span class="step-counter">Step 2 of 4</span>
      <button class="btn-next" onclick="goTo(2)">Next →</button>
    </div>
  </div>

  <!-- Page 2: Model -->
  <div id="page2" style="display:none">
    <div class="card">
      <div class="card-title">Select Model</div>
      <div class="card-sub">Choose which model your node will serve</div>
      <div class="model-card selected" id="model-0" onclick="selectModel(0)">
        <div class="model-name">Qwen2.5-3B (int8)</div>
        <div class="model-meta">3.0 GB · ★★★★★ Recommended for most hardware</div>
      </div>
      <div class="model-card" id="model-1" onclick="selectModel(1)">
        <div class="model-name">Llama 3.2-3B (int8)</div>
        <div class="model-meta">3.2 GB · ★★★★☆ Good general performance</div>
      </div>
      <div class="model-card" id="model-2" onclick="selectModel(2)">
        <div class="model-name">Phi-3-mini (int8)</div>
        <div class="model-meta">2.4 GB · ★★★☆☆ Fastest · Best for low VRAM</div>
      </div>
    </div>
    <div class="nav">
      <button class="btn-back" onclick="goTo(1)">← Back</button>
      <span class="step-counter">Step 3 of 4</span>
      <button class="btn-next" onclick="goTo(3)">Next →</button>
    </div>
  </div>

  <!-- Page 3: Confirm -->
  <div id="page3" style="display:none">
    <div class="card">
      <div class="card-title">Ready to Join</div>
      <div class="card-sub">Review your configuration</div>
      <div class="confirm-row"><span class="confirm-label">Model</span><span class="confirm-val" id="confirm-model">Qwen2.5-3B (int8)</span></div>
      <div class="confirm-row"><span class="confirm-label">Compute</span><span class="confirm-val" id="confirm-compute">50% · 4.0 GB VRAM</span></div>
      <div class="confirm-row"><span class="confirm-label">Layers hosting</span><span class="confirm-val">0 – 6</span></div>
      <div class="confirm-row"><span class="confirm-label">Starting credits</span><span class="confirm-val">10.0</span></div>
      <div class="confirm-row"><span class="confirm-label">Earning rate</span><span class="confirm-val" id="confirm-rate">+2.4 credits / pass</span></div>
      <div class="success-bar">
        <span class="earning-label">You're contributing to a decentralized AI mesh</span>
        <span style="color:#22c55e; font-size:18px">✓</span>
      </div>
    </div>
    <div class="nav">
      <button class="btn-back" onclick="goTo(2)">← Back</button>
      <span class="step-counter">Step 4 of 4</span>
      <button class="btn-join" onclick="doJoin()">✓ Join Mesh</button>
    </div>
  </div>
</div>

<script>
const vscode = acquireVsCodeApi();
const stepLabels = ['Hardware','Compute','Model','Confirm'];
let currentPage = 0;
let selectedCompute = 1;
let selectedModelIdx = 0;
const computeRates = [1.2, 2.4, 3.6, 4.8];
const computePcts = [25, 50, 75, 100];
const computeGbs = [2.0, 4.0, 6.0, 8.0];
const modelNames = ['Qwen2.5-3B (int8)', 'Llama 3.2-3B (int8)', 'Phi-3-mini (int8)'];
const modelKeys = ['qwen2.5-3b', 'llama3.2-3b', 'phi-3-mini'];

function renderSteps(active) {
  const el = document.getElementById('stepIndicator');
  el.innerHTML = '';
  stepLabels.forEach((label, i) => {
    const state = i < active ? 'done' : i === active ? 'active' : 'inactive';
    el.innerHTML += '<div class="step ' + state + '"><div class="step-circle">' + (i < active ? '✓' : i+1) + '</div><div class="step-label">' + label + '</div></div>';
    if (i < stepLabels.length - 1) {
      el.innerHTML += '<div class="step-line ' + (i < active ? 'done' : '') + '"></div>';
    }
  });
}

function goTo(n) {
  document.querySelectorAll('[id^=page]').forEach(p => p.style.display = 'none');
  document.getElementById('page' + n).style.display = 'block';
  currentPage = n;
  renderSteps(n);
  updateConfirmPage();
}

function selectCompute(idx, el) {
  document.querySelectorAll('.compute-btn').forEach(b => b.classList.remove('selected'));
  el.classList.add('selected');
  selectedCompute = idx;
  document.getElementById('earningDisplay').textContent = '+' + computeRates[idx] + ' credits / forward pass';
}

function selectModel(idx) {
  document.querySelectorAll('.model-card').forEach((m, i) => {
    m.classList.remove('selected');
  });
  document.getElementById('model-' + idx).classList.add('selected');
  selectedModelIdx = idx;
}

function updateConfirmPage() {
  document.getElementById('confirm-model').textContent = modelNames[selectedModelIdx];
  document.getElementById('confirm-compute').textContent = computePcts[selectedCompute] + '% · ' + computeGbs[selectedCompute] + ' GB VRAM';
  document.getElementById('confirm-rate').textContent = '+' + computeRates[selectedCompute] + ' credits / pass';
}

function doJoin() {
  vscode.postMessage({
    command: 'join',
    data: {
      computePct: computePcts[selectedCompute],
      model: modelKeys[selectedModelIdx]
    }
  });
}

renderSteps(0);
</script>
</body>
</html>`;
    }
}
