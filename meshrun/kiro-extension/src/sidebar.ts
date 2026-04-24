import * as vscode from 'vscode';

// Mock data — TODO: replace with real coordinator HTTP call
const MOCK_NODES = [
    { id: 'node-a', layers: '0-6',   status: 'active',      credits: 42.1, latency: '38ms' },
    { id: 'node-b', layers: '7-13',  status: 'active',      credits: 38.7, latency: '41ms' },
    { id: 'node-c', layers: '14-20', status: 'idle',        credits: 21.3, latency: '44ms' },
    { id: 'node-d', layers: '21-27', status: 'unreachable', credits: 9.8,  latency: '--'   },
];

type NodeStatus = 'active' | 'idle' | 'unreachable';

class NodeItem extends vscode.TreeItem {
    constructor(
        public readonly nodeId: string,
        public readonly layers: string,
        public readonly status: NodeStatus,
        public readonly credits: number,
        public readonly latency: string
    ) {
        super(nodeId, vscode.TreeItemCollapsibleState.None);
        this.description = `layers ${layers} · ${latency}`;
        this.tooltip = `Status: ${status}\nCredits earned: ${credits}\nLatency: ${latency}`;
        this.iconPath = new vscode.ThemeIcon(
            status === 'active' ? 'circle-filled' :
            status === 'idle'   ? 'circle-outline' : 'error'
        );
    }
}

export class NodesProvider implements vscode.TreeDataProvider<NodeItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<NodeItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: NodeItem): vscode.TreeItem {
        return element;
    }

    getChildren(): NodeItem[] {
        // TODO: replace with real coordinator call
        return MOCK_NODES.map(n =>
            new NodeItem(n.id, n.layers, n.status as NodeStatus, n.credits, n.latency)
        );
    }
}

class CreditsItem extends vscode.TreeItem {
    constructor(label: string, description: string, icon: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = description;
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}

export class CreditsProvider implements vscode.TreeDataProvider<CreditsItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<CreditsItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: CreditsItem): vscode.TreeItem {
        return element;
    }

    getChildren(): CreditsItem[] {
        // TODO: replace with real coordinator call
        return [
            new CreditsItem('Balance',             '128.4 credits',        'star'),
            new CreditsItem('Compute contributed', '3.2 GPU-hours',        'server'),
            new CreditsItem('Priority score',      '94.2',                 'dashboard'),
            new CreditsItem('Cost saved',          '$12.43 vs cloud APIs', 'arrow-down'),
            new CreditsItem('CO₂ avoided',         '4.92g',                'leaf'),
        ];
    }
}
