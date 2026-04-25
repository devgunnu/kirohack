import * as vscode from 'vscode';
import { registerCommands } from './commands';
import { NodesProvider, CreditsProvider, JobsProvider } from './sidebar';
import { syncJoinedContext } from './onboarding';

export function activate(context: vscode.ExtensionContext): void {
    console.log('MeshRun Kiro extension activated');

    // Set the meshrun.joined context key so welcome views render correctly
    syncJoinedContext();

    // Register all CLI-backed commands
    registerCommands(context);

    // -----------------------------------------------------------------------
    // Sidebar tree-view providers
    // -----------------------------------------------------------------------
    const nodesProvider   = new NodesProvider();
    const creditsProvider = new CreditsProvider();
    const jobsProvider    = new JobsProvider();

    const nodesView = vscode.window.createTreeView('meshrunNodes', {
        treeDataProvider: nodesProvider,
        showCollapseAll: true,
    });

    vscode.window.registerTreeDataProvider('meshrunCredits', creditsProvider);
    vscode.window.registerTreeDataProvider('meshrunJobs', jobsProvider);

    // -----------------------------------------------------------------------
    // Feature 10 — Earnings badge on the Nodes view
    // -----------------------------------------------------------------------
    function updateBadge(): void {
        const earned = creditsProvider.sessionEarned;
        if (earned > 0) {
            nodesView.badge = {
                value: Math.round(earned),
                tooltip: `+${earned.toFixed(1)} credits earned this session`,
            };
        }
    }

    // -----------------------------------------------------------------------
    // Refresh command — refreshes all three providers
    // -----------------------------------------------------------------------
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrunNodes.refresh', () => {
            nodesProvider.refresh();
            creditsProvider.refresh();
            jobsProvider.refresh();
            updateBadge();
            vscode.window.showInformationMessage('MeshRun: Sidebar refreshed');
        })
    );

    // -----------------------------------------------------------------------
    // Feature 8 — Auto-refresh with configurable interval
    // -----------------------------------------------------------------------
    let autoRefreshTimer: ReturnType<typeof setInterval> | undefined;

    function startAutoRefresh(): void {
        stopAutoRefresh();
        const config = vscode.workspace.getConfiguration('meshrun');
        const enabled  = config.get<boolean>('autoRefresh.enabled', true);
        const interval = config.get<number>('autoRefresh.intervalSeconds', 30);

        if (!enabled || interval <= 0) { return; }

        autoRefreshTimer = setInterval(() => {
            nodesProvider.refresh();
            creditsProvider.refresh();
            jobsProvider.refresh();
            updateBadge();
        }, interval * 1000);
    }

    function stopAutoRefresh(): void {
        if (autoRefreshTimer) {
            clearInterval(autoRefreshTimer);
            autoRefreshTimer = undefined;
        }
    }

    // React to config changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('meshrun.autoRefresh')) {
                startAutoRefresh();
            }
        })
    );

    startAutoRefresh();
    context.subscriptions.push({ dispose: stopAutoRefresh });

    // -----------------------------------------------------------------------
    // Status bar item — always visible, click to open dashboard
    // -----------------------------------------------------------------------
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBar.text = '$(server) MeshRun';
    statusBar.tooltip = 'Click to open MeshRun Dashboard';
    statusBar.command = 'meshrun.dashboard';
    statusBar.show();
    context.subscriptions.push(statusBar);

    // Keep tree view subscription alive
    context.subscriptions.push(nodesView);
}

export function deactivate(): void {}
