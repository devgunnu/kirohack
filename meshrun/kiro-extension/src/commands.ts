import * as vscode from 'vscode';
import { runMeshrun, promptInput } from './meshrunProcess';
import { MeshRunDashboardPanel } from './dashboard';
import { isJoined, OnboardingPanel } from './onboarding';

export function registerCommands(context: vscode.ExtensionContext): void {

    function requireJoined(): boolean {
        if (!isJoined()) {
            vscode.window.showWarningMessage(
                'MeshRun: You need to join the mesh first.',
                'Join Now'
            ).then(selection => {
                if (selection === 'Join Now') {
                    OnboardingPanel.createOrShow(context);
                }
            });
            return false;
        }
        return true;
    }

    // Submit inference job
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.submit', async () => {
            if (!requireJoined()) { return; }

            const prompt = await promptInput(
                'Enter your inference prompt',
                'e.g. explain what a transformer is'
            );
            if (!prompt) { return; }

            const mode = await vscode.window.showQuickPick(
                ['Sync — stream output now', 'Async — get job ID and retrieve later'],
                { placeHolder: 'Select submission mode' }
            );
            if (!mode) { return; }

            const args = mode.startsWith('Async')
                ? ['submit', `"${prompt}"`, '--async']
                : ['submit', `"${prompt}"`];

            try {
                await runMeshrun(args);
            } catch (e: any) {
                vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
            }
        })
    );

    // Status
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.status', async () => {
            if (!requireJoined()) { return; }
            try {
                await runMeshrun(['status']);
            } catch (e: any) {
                vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
            }
        })
    );

    // Nodes
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.nodes', async () => {
            if (!requireJoined()) { return; }
            try {
                await runMeshrun(['nodes']);
            } catch (e: any) {
                vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
            }
        })
    );

    // Credits
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.credits', async () => {
            if (!requireJoined()) { return; }
            try {
                await runMeshrun(['credits']);
            } catch (e: any) {
                vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
            }
        })
    );

    // Join — opens onboarding webview
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.join', () => {
            OnboardingPanel.createOrShow(context);
        })
    );

    // Leave
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.leave', async () => {
            if (!requireJoined()) { return; }
            const confirmed = await vscode.window.showWarningMessage(
                'Deregister this machine from the MeshRun mesh?',
                'Yes, Leave',
                'Cancel'
            );
            if (confirmed !== 'Yes, Leave') { return; }
            try {
                await runMeshrun(['leave']);
            } catch (e: any) {
                vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
            }
        })
    );

    // Dashboard — opens as Kiro/VS Code webview panel
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.dashboard', () => {
            MeshRunDashboardPanel.createOrShow(context);
        })
    );

    // Refresh sidebar
    context.subscriptions.push(
        vscode.commands.registerCommand('meshrun.refreshSidebar', () => {
            vscode.commands.executeCommand('meshrunNodes.refresh');
        })
    );
}
