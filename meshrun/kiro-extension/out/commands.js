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
exports.registerCommands = registerCommands;
const vscode = __importStar(require("vscode"));
const meshrunProcess_1 = require("./meshrunProcess");
const dashboard_1 = require("./dashboard");
function registerCommands(context) {
    // Submit inference job
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.submit', async () => {
        const prompt = await (0, meshrunProcess_1.promptInput)('Enter your inference prompt', 'e.g. explain what a transformer is');
        if (!prompt) {
            return;
        }
        const mode = await vscode.window.showQuickPick(['Sync — stream output now', 'Async — get job ID and retrieve later'], { placeHolder: 'Select submission mode' });
        if (!mode) {
            return;
        }
        const args = mode.startsWith('Async')
            ? ['submit', prompt, '--async']
            : ['submit', prompt];
        try {
            await (0, meshrunProcess_1.runMeshrun)(args);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Status
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.status', async () => {
        try {
            await (0, meshrunProcess_1.runMeshrun)(['status']);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Nodes
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.nodes', async () => {
        try {
            await (0, meshrunProcess_1.runMeshrun)(['nodes']);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Credits
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.credits', async () => {
        try {
            await (0, meshrunProcess_1.runMeshrun)(['credits']);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Join
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.join', async () => {
        const confirmed = await vscode.window.showWarningMessage('Register this machine as a MeshRun worker node?', 'Yes, Join', 'Cancel');
        if (confirmed !== 'Yes, Join') {
            return;
        }
        try {
            await (0, meshrunProcess_1.runMeshrun)(['join']);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Leave
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.leave', async () => {
        const confirmed = await vscode.window.showWarningMessage('Deregister this machine from the MeshRun mesh?', 'Yes, Leave', 'Cancel');
        if (confirmed !== 'Yes, Leave') {
            return;
        }
        try {
            await (0, meshrunProcess_1.runMeshrun)(['leave']);
        }
        catch (e) {
            vscode.window.showErrorMessage(`MeshRun: ${e.message}`);
        }
    }));
    // Dashboard — opens as Kiro/VS Code webview panel
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.dashboard', () => {
        dashboard_1.MeshRunDashboardPanel.createOrShow(context);
    }));
    // Refresh sidebar
    context.subscriptions.push(vscode.commands.registerCommand('meshrun.refreshSidebar', () => {
        vscode.commands.executeCommand('meshrunNodes.refresh');
    }));
}
//# sourceMappingURL=commands.js.map