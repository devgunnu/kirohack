import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';

export class MeshRunDashboardPanel {
    public static currentPanel: MeshRunDashboardPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _serverProcess: ChildProcess | undefined;

    public static createOrShow(context: vscode.ExtensionContext): void {
        if (MeshRunDashboardPanel.currentPanel) {
            MeshRunDashboardPanel.currentPanel._panel.reveal(vscode.ViewColumn.Two);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            'meshrunDashboard',
            'MeshRun Dashboard',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            }
        );
        MeshRunDashboardPanel.currentPanel = new MeshRunDashboardPanel(panel, context);
    }

    private constructor(panel: vscode.WebviewPanel, context: vscode.ExtensionContext) {
        this._panel = panel;

        // Show loading state immediately
        this._panel.webview.html = this._getLoadingHtml();

        // Start the FastAPI dashboard server via the CLI
        this._serverProcess = spawn('meshrun', ['dashboard'], {
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

    private _getLoadingHtml(): string {
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

    private _getDashboardHtml(): string {
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
