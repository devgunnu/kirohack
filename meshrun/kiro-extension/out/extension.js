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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const commands_1 = require("./commands");
const sidebar_1 = require("./sidebar");
function activate(context) {
    console.log('MeshRun Kiro extension activated');
    // Register all CLI-backed commands
    (0, commands_1.registerCommands)(context);
    // Register sidebar tree view providers
    const nodesProvider = new sidebar_1.NodesProvider();
    const creditsProvider = new sidebar_1.CreditsProvider();
    vscode.window.registerTreeDataProvider('meshrunNodes', nodesProvider);
    vscode.window.registerTreeDataProvider('meshrunCredits', creditsProvider);
    // Wire up refresh command for both providers
    context.subscriptions.push(vscode.commands.registerCommand('meshrunNodes.refresh', () => {
        nodesProvider.refresh();
        creditsProvider.refresh();
        vscode.window.showInformationMessage('MeshRun: Node list refreshed');
    }));
    // Status bar item — always visible, click to open dashboard
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = '$(server) MeshRun';
    statusBar.tooltip = 'Click to open MeshRun Dashboard';
    statusBar.command = 'meshrun.dashboard';
    statusBar.show();
    context.subscriptions.push(statusBar);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map