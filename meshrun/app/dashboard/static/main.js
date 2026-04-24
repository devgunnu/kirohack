// MeshRun Dashboard - Main JavaScript

// State
const state = {
    feedEvents: [],
    maxFeedEvents: 20,
    nodes: [
        { id: 'node-a', layers: '0-6', status: 'active', credits: 42.1 },
        { id: 'node-b', layers: '7-13', status: 'active', credits: 38.7 },
        { id: 'node-c', layers: '14-20', status: 'idle', credits: 21.3 },
        { id: 'node-d', layers: '21-27', status: 'unreachable', credits: 9.8 }
    ],
    links: [
        { source: 'node-a', target: 'node-b' },
        { source: 'node-b', target: 'node-c' },
        { source: 'node-c', target: 'node-d' }
    ],
    stats: {
        totalTokens: 0,
        costSaved: 0,
        co2Avoided: 0
    }
};

// WebSocket connection
let ws = null;

function connectWebSocket() {
    ws = new WebSocket('ws://127.0.0.1:7654/ws');

    ws.onopen = () => {
        document.getElementById('status-dot').classList.add('connected');
        document.getElementById('status-text').textContent = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('status-dot').classList.remove('connected');
        document.getElementById('status-text').textContent = 'Disconnected';
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        document.getElementById('status-dot').classList.remove('connected');
        document.getElementById('status-text').textContent = 'Error';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleEvent(data);
    };
}

// Event handler dispatch
function handleEvent(event) {
    switch (event.type) {
        case 'job_started':
            addFeedEvent(event);
            break;
        case 'job_hop':
            addFeedEvent(event);
            animateEdge(event.from_node, event.to_node);
            break;
        case 'job_complete':
            addFeedEvent(event);
            updateStats(event);
            break;
        case 'node_status':
            addFeedEvent(event);
            updateNodeStatus(event.node_id, event.status);
            break;
        case 'queue_update':
            addFeedEvent(event);
            break;
        case 'stats_update':
            updateStatsFromEvent(event);
            break;
    }
}

// Feed management
function addFeedEvent(event) {
    state.feedEvents.push(event);
    if (state.feedEvents.length > state.maxFeedEvents) {
        state.feedEvents.shift();
    }
    renderFeed();
}

function renderFeed() {
    const tbody = document.getElementById('feed-body');
    tbody.innerHTML = '';

    state.feedEvents.forEach(event => {
        const row = document.createElement('tr');
        const time = new Date(event.timestamp * 1000).toLocaleTimeString();
        const detail = getEventDetail(event);

        row.innerHTML = `
            <td>${time}</td>
            <td class="event-${event.type}">${event.type}</td>
            <td>${event.job_id || '-'}</td>
            <td>${detail}</td>
        `;
        tbody.appendChild(row);
    });

    // Auto-scroll to bottom
    const container = document.getElementById('feed-container');
    container.scrollTop = container.scrollHeight;
}

function getEventDetail(event) {
    switch (event.type) {
        case 'job_started':
            return event.prompt_preview || '';
        case 'job_hop':
            return `${event.from_node} → ${event.to_node} (${event.latency_ms}ms)`;
        case 'job_complete':
            return `${event.tokens} tokens, ${event.total_latency}s`;
        case 'node_status':
            return `${event.node_id}: ${event.status}`;
        case 'queue_update':
            return `Queue depth: ${event.depth}`;
        case 'stats_update':
            return `${event.total_tokens} tokens total`;
        default:
            return '';
    }
}

// Stats updates
function updateStats(event) {
    if (event.tokens) {
        state.stats.totalTokens += event.tokens;
    }
    if (event.cost_saved_usd) {
        state.stats.costSaved += event.cost_saved_usd;
    }
    if (event.co2_avoided_g) {
        state.stats.co2Avoided += event.co2_avoided_g;
    }
    renderStats();
}

function updateStatsFromEvent(event) {
    state.stats.totalTokens = event.total_tokens || state.stats.totalTokens;
    state.stats.costSaved = event.total_cost_saved_usd || state.stats.costSaved;
    state.stats.co2Avoided = event.total_co2_avoided_g || state.stats.co2Avoided;
    renderStats();
}

function renderStats() {
    animateValue('total-tokens', state.stats.totalTokens.toLocaleString());
    animateValue('cost-saved', `$${state.stats.costSaved.toFixed(2)}`);
    animateValue('co2-avoided', `${state.stats.co2Avoided.toFixed(4)}g`);
}

function animateValue(elementId, newValue) {
    const el = document.getElementById(elementId);
    el.textContent = newValue;
    el.classList.add('updating');
    setTimeout(() => el.classList.remove('updating'), 200);
}

// D3 Node Graph
let simulation, svg, nodeElements, linkElements;

function initGraph() {
    const container = document.getElementById('node-graph');
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    svg = d3.select('#node-graph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create links
    linkElements = svg.append('g')
        .selectAll('line')
        .data(state.links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('id', d => `link-${d.source}-${d.target}`);

    // Create node groups
    const nodeGroups = svg.append('g')
        .selectAll('g')
        .data(state.nodes)
        .enter()
        .append('g')
        .attr('class', 'node-group');

    // Node circles
    nodeElements = nodeGroups.append('circle')
        .attr('class', 'node-circle')
        .attr('r', d => 20 + d.credits / 5)
        .attr('fill', d => getNodeColor(d.status))
        .attr('id', d => `node-${d.id}`);

    // Node labels
    nodeGroups.append('text')
        .attr('class', 'node-label')
        .attr('dy', 35)
        .text(d => d.id);

    // Node sublabels (layers)
    nodeGroups.append('text')
        .attr('class', 'node-sublabel')
        .attr('dy', 48)
        .text(d => `Layers ${d.layers}`);

    // Force simulation
    simulation = d3.forceSimulation(state.nodes)
        .force('link', d3.forceLink(state.links).id(d => d.id).distance(150))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .on('tick', ticked);
}

function ticked() {
    linkElements
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

    svg.selectAll('.node-group')
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
}

function getNodeColor(status) {
    switch (status) {
        case 'active': return '#22c55e';
        case 'idle': return '#eab308';
        case 'unreachable': return '#ef4444';
        default: return '#666666';
    }
}

function updateNodeStatus(nodeId, status) {
    const node = state.nodes.find(n => n.id === nodeId);
    if (node) {
        node.status = status;
        d3.select(`#node-${nodeId}`)
            .transition()
            .duration(300)
            .attr('fill', getNodeColor(status));
    }
}

function animateEdge(fromNode, toNode) {
    const linkId = `#link-${fromNode}-${toNode}`;
    const link = d3.select(linkId);
    
    if (!link.empty()) {
        link.classed('active', true);
        setTimeout(() => link.classed('active', false), 500);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initGraph();
    connectWebSocket();
    renderStats();
});
