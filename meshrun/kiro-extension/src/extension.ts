import * as vscode from 'vscode';
import { registerCommands } from './commands';
import { NodesProvider, CreditsProvider } from './sidebar';

export function activate(context: vscode.ExtensionContext): void {
    console.log('MeshRun Kiro extension activated');

    // Register all CLI-backed commands
    registerCommands(context);

    // Register sidebar tree view providers
    const nodesProvider = new NodesProvider();
    const creditsProvider = new CreditsProvider();

    vscode.window.registerTreeDataProvider('meshrunNodes', nodesProvider);
    vscode.window.registerTreeDataProvider('meshrunCredits', creditsProvider);

    // Wire up refresh command for both providers
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrunNodes.refresh', () => {
            nodesProvider.refresh();
            creditsProvider.refresh();
            vscode.window.showInformationMessage('MeshRun: Node list refreshed');
        })
    );

    // Status bar item — always visible, click to open dashboard
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBar.text = '$(server) MeshRun';
    statusBar.tooltip = 'Click to open MeshRun Dashboard';
    statusBar.command = 'meshrun.dashboard';
    statusBar.show();
    context.subscriptions.push(statusBar);
}

export function deactivate(): void {}
