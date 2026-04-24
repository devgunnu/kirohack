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
exports.CreditsProvider = exports.NodesProvider = void 0;
const vscode = __importStar(require("vscode"));
// Mock data — TODO: replace with real coordinator HTTP call
const MOCK_NODES = [
    { id: 'node-a', layers: '0-6', status: 'active', credits: 42.1, latency: '38ms' },
    { id: 'node-b', layers: '7-13', status: 'active', credits: 38.7, latency: '41ms' },
    { id: 'node-c', layers: '14-20', status: 'idle', credits: 21.3, latency: '44ms' },
    { id: 'node-d', layers: '21-27', status: 'unreachable', credits: 9.8, latency: '--' },
];
class NodeItem extends vscode.TreeItem {
    constructor(nodeId, layers, status, credits, latency) {
        super(nodeId, vscode.TreeItemCollapsibleState.None);
        this.nodeId = nodeId;
        this.layers = layers;
        this.status = status;
        this.credits = credits;
        this.latency = latency;
        this.description = `layers ${layers} · ${latency}`;
        this.tooltip = `Status: ${status}\nCredits earned: ${credits}\nLatency: ${latency}`;
        this.iconPath = new vscode.ThemeIcon(status === 'active' ? 'circle-filled' :
            status === 'idle' ? 'circle-outline' : 'error');
    }
}
class NodesProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren() {
        // TODO: replace with real coordinator call
        return MOCK_NODES.map(n => new NodeItem(n.id, n.layers, n.status, n.credits, n.latency));
    }
}
exports.NodesProvider = NodesProvider;
class CreditsItem extends vscode.TreeItem {
    constructor(label, description, icon) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = description;
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}
class CreditsProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren() {
        // TODO: replace with real coordinator call
        return [
            new CreditsItem('Balance', '128.4 credits', 'star'),
            new CreditsItem('Compute contributed', '3.2 GPU-hours', 'server'),
            new CreditsItem('Priority score', '94.2', 'dashboard'),
            new CreditsItem('Cost saved', '$12.43 vs cloud APIs', 'arrow-down'),
            new CreditsItem('CO₂ avoided', '4.92g', 'leaf'),
        ];
    }
}
exports.CreditsProvider = CreditsProvider;
//# sourceMappingURL=sidebar.js.map