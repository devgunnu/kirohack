import * as vscode from 'vscode';

// ---------------------------------------------------------------------------
// Mock data — TODO: replace with real coordinator HTTP calls
// ---------------------------------------------------------------------------

const MOCK_NODES = [
    { id: 'node-a', layers: '0-6',   status: 'active',      credits: 42.1, latency: '38ms',  tokPerSec: 12.3 },
    { id: 'node-b', layers: '7-13',  status: 'active',      credits: 38.7, latency: '41ms',  tokPerSec: 11.1 },
    { id: 'node-c', layers: '14-20', status: 'idle',         credits: 21.3, latency: '44ms',  tokPerSec: 0 },
    { id: 'node-d', layers: '21-27', status: 'unreachable',  credits: 9.8,  latency: '--',    tokPerSec: 0 },
];

const MOCK_JOBS = [
    { id: 'job-1f3a', prompt: 'explain what a transformer is',       status: 'done',    latency: '1.2s', credits: 2.4, ts: Date.now() - 120_000 },
    { id: 'job-82cb', prompt: 'write a haiku about distributed AI',  status: 'running',  latency: '--',   credits: 0,   ts: Date.now() - 30_000 },
    { id: 'job-d90e', prompt: 'summarize the MeshRun whitepaper',    status: 'queued',   latency: '--',   credits: 0,   ts: Date.now() - 5_000 },
    { id: 'job-44af', prompt: 'translate hello world to 5 languages', status: 'done',    latency: '0.9s', credits: 1.8, ts: Date.now() - 300_000 },
];

type NodeStatus = 'active' | 'idle' | 'unreachable';
type JobStatus  = 'queued' | 'running' | 'done' | 'failed';

// ---------------------------------------------------------------------------
// Feature 9 — Node Grouping by Status
// ---------------------------------------------------------------------------

class NodeGroupItem extends vscode.TreeItem {
    constructor(
        public readonly groupStatus: NodeStatus,
        public readonly count: number
    ) {
        const label = groupStatus.charAt(0).toUpperCase() + groupStatus.slice(1);
        super(`${label} (${count})`, vscode.TreeItemCollapsibleState.Expanded);
        this.contextValue = 'nodeGroup';
        this.iconPath = new vscode.ThemeIcon(
            groupStatus === 'active' ? 'pass-filled' :
            groupStatus === 'idle'   ? 'circle-outline' : 'warning',
            new vscode.ThemeColor(
                groupStatus === 'active' ? 'testing.iconPassed' :
                groupStatus === 'idle'   ? 'descriptionForeground' : 'testing.iconFailed'
            )
        );
    }
}

class NodeItem extends vscode.TreeItem {
    constructor(
        public readonly nodeId: string,
        public readonly layers: string,
        public readonly status: NodeStatus,
        public readonly credits: number,
        public readonly latency: string,
        public readonly tokPerSec: number
    ) {
        super(nodeId, vscode.TreeItemCollapsibleState.None);

        // Feature 5-style live decorations
        const throughput = status === 'active' ? `${tokPerSec} tok/s` :
                           status === 'idle'   ? 'idle' : 'offline';
        this.description = `layers ${layers} · ${throughput}`;
        this.tooltip = new vscode.MarkdownString(
            `**${nodeId}**\n\n` +
            `| | |\n|---|---|\n` +
            `| Status | ${status} |\n` +
            `| Layers | ${layers} |\n` +
            `| Latency | ${latency} |\n` +
            `| Throughput | ${throughput} |\n` +
            `| Credits earned | ${credits} |`
        );
        this.iconPath = new vscode.ThemeIcon(
            status === 'active' ? 'circle-filled' :
            status === 'idle'   ? 'circle-outline' : 'error',
            new vscode.ThemeColor(
                status === 'active' ? 'testing.iconPassed' :
                status === 'idle'   ? 'descriptionForeground' : 'testing.iconFailed'
            )
        );
        this.contextValue = 'node';
    }
}

type NodesTreeItem = NodeGroupItem | NodeItem;

export class NodesProvider implements vscode.TreeDataProvider<NodesTreeItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<NodesTreeItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: NodesTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: NodesTreeItem): NodesTreeItem[] {
        if (!element) {
            // Root level — return groups that have at least one node
            const statuses: NodeStatus[] = ['active', 'idle', 'unreachable'];
            return statuses
                .map(s => {
                    const count = MOCK_NODES.filter(n => n.status === s).length;
                    return count > 0 ? new NodeGroupItem(s, count) : null;
                })
                .filter((g): g is NodeGroupItem => g !== null);
        }

        if (element instanceof NodeGroupItem) {
            return MOCK_NODES
                .filter(n => n.status === element.groupStatus)
                .map(n => new NodeItem(
                    n.id, n.layers, n.status as NodeStatus,
                    n.credits, n.latency, n.tokPerSec
                ));
        }

        return [];
    }
}

// ---------------------------------------------------------------------------
// Feature 2 — Job History / Queue View
// ---------------------------------------------------------------------------

class JobItem extends vscode.TreeItem {
    constructor(
        public readonly jobId: string,
        public readonly prompt: string,
        public readonly jobStatus: JobStatus,
        public readonly latency: string,
        public readonly credits: number,
        public readonly ts: number
    ) {
        super(prompt.length > 40 ? prompt.slice(0, 37) + '…' : prompt, vscode.TreeItemCollapsibleState.None);

        const statusIcon = jobStatus === 'done'    ? '$(check)'  :
                           jobStatus === 'running'  ? '$(sync~spin)' :
                           jobStatus === 'queued'   ? '$(clock)' : '$(error)';

        this.description = `${jobId} · ${jobStatus}`;
        this.tooltip = new vscode.MarkdownString(
            `**${jobId}**\n\n` +
            `| | |\n|---|---|\n` +
            `| Prompt | ${prompt} |\n` +
            `| Status | ${jobStatus} |\n` +
            `| Latency | ${latency} |\n` +
            `| Credits | ${credits} |\n` +
            `| Submitted | ${new Date(ts).toLocaleTimeString()} |`
        );
        this.iconPath = new vscode.ThemeIcon(
            jobStatus === 'done'    ? 'pass-filled' :
            jobStatus === 'running' ? 'sync~spin' :
            jobStatus === 'queued'  ? 'clock' : 'error',
            new vscode.ThemeColor(
                jobStatus === 'done'    ? 'testing.iconPassed' :
                jobStatus === 'running' ? 'progressBar.background' :
                jobStatus === 'queued'  ? 'descriptionForeground' : 'testing.iconFailed'
            )
        );
        this.contextValue = 'job';
    }
}

export class JobsProvider implements vscode.TreeDataProvider<JobItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<JobItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: JobItem): vscode.TreeItem {
        return element;
    }

    getChildren(): JobItem[] {
        // Sort: running first, then queued, then done (most recent first)
        const order: Record<string, number> = { running: 0, queued: 1, done: 2, failed: 3 };
        const sorted = [...MOCK_JOBS].sort((a, b) => {
            const diff = (order[a.status] ?? 4) - (order[b.status] ?? 4);
            return diff !== 0 ? diff : b.ts - a.ts;
        });
        return sorted.map(j =>
            new JobItem(j.id, j.prompt, j.status as JobStatus, j.latency, j.credits, j.ts)
        );
    }
}

// ---------------------------------------------------------------------------
// Feature 6 — Credits with Trend Indicators
// ---------------------------------------------------------------------------

class CreditsItem extends vscode.TreeItem {
    constructor(label: string, description: string, icon: string, color?: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = description;
        this.iconPath = new vscode.ThemeIcon(
            icon,
            color ? new vscode.ThemeColor(color) : undefined
        );
    }
}

export class CreditsProvider implements vscode.TreeDataProvider<CreditsItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<CreditsItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private _startBalance = 128.4;
    private _sessionEarned = 0;

    refresh(): void {
        // Simulate earning credits each refresh cycle
        this._sessionEarned += +(Math.random() * 0.8).toFixed(1);
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: CreditsItem): vscode.TreeItem {
        return element;
    }

    getChildren(): CreditsItem[] {
        const current = +(this._startBalance + this._sessionEarned).toFixed(1);
        const trend = this._sessionEarned > 0
            ? `↑ +${this._sessionEarned.toFixed(1)} this session`
            : '';

        return [
            new CreditsItem('Balance',             `${current} credits ${trend}`, 'star-full',   'testing.iconPassed'),
            new CreditsItem('Compute contributed', '3.2 GPU-hours',               'server',      'descriptionForeground'),
            new CreditsItem('Priority score',      '94.2',                        'dashboard',   'descriptionForeground'),
            new CreditsItem('Cost saved',          '$12.43 vs cloud APIs',        'arrow-down',  'testing.iconPassed'),
            new CreditsItem('CO₂ avoided',         '4.92 g',                      'leaf',        'testing.iconPassed'),
        ];
    }

    /** Returns session credits earned — used for the activity bar badge */
    get sessionEarned(): number {
        return this._sessionEarned;
    }
}
