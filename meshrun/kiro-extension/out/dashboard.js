"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.MeshRunDashboardPanel = void 0;
const vscode = __importStar(require("vscode"));
const child_process_1 = require("child_process");
class MeshRunDashboardPanel {
    static createOrShow(context) {
        if (MeshRunDashboardPanel.currentPanel) {
            MeshRunDashboardPanel.currentPanel._panel.reveal(vscode.ViewColumn.Two);
            return;
        }
        const panel = vscode.window.createWebviewPanel('meshrunDashboard', 'MeshRun Dashboard', vscode.ViewColumn.Two, {
            enableScripts: true,
            retainContextWhenHidden: true,
        });
        MeshRunDashboardPanel.currentPanel = new MeshRunDashboardPanel(panel, context);
    }
    constructor(panel, context) {
        this._panel = panel;
        // Show loading state immediately
        this._panel.webview.html = this._getLoadingHtml();
        // Start the FastAPI dashboard server via the CLI
        this._serverProcess = (0, child_process_1.spawn)('meshrun', ['dashboard'], {
            shell: true,
            detached: false,
            env: { ...process.env }
        });
        // Wait for server to boot then load the iframe
        setTimeout(() => {
            this._panel.webview.html = this._getDashboardHtml();
        }, 2000);
        this._panel.onDidDispose(() => {
            this._serverProcess?.kill();
            MeshRunDashboardPanel.currentPanel = undefined;
        }, null, context.subscriptions);
    }
    _getLoadingHtml() {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {
            margin: 0;
            background: #0d0d0d;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            flex-direction: column;
            gap: 16px;
            font-family: system-ui;
            color: #00d4ff;
        }
        .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid #1a1a1a;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        p { font-size: 13px; color: #666; margin: 0; }
    </style>
</head>
<body>
    <div class="spinner"></div>
    <span>Starting MeshRun Dashboard...</span>
    <p>Launching FastAPI server on port 7654</p>
</body>
</html>`;
    }
    _getDashboardHtml() {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MeshRun Dashboard</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100vh;
            overflow: hidden;
            background: #0d0d0d;
        }
        iframe {
            width: 100%;
            height: 100vh;
            border: none;
        }
    </style>
</head>
<body>
    <iframe
        src="http://127.0.0.1:7654"
        title="MeshRun Dashboard"
        sandbox="allow-scripts allow-same-origin allow-forms"
    ></iframe>
</body>
</html>`;
    }
}
exports.MeshRunDashboardPanel = MeshRunDashboardPanel;
//# sourceMappingURL=dashboard.js.map