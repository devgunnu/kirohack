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
exports.outputChannel = void 0;
exports.runMeshrun = runMeshrun;
exports.promptInput = promptInput;
const child_process_1 = require("child_process");
const vscode = __importStar(require("vscode"));
exports.outputChannel = vscode.window.createOutputChannel('MeshRun');
/**
 * Runs a meshrun CLI command and streams output to the MeshRun Output Channel.
 * Built on top of the existing meshrun CLI installed via pip install -e .
 */
function runMeshrun(args) {
    return new Promise((resolve, reject) => {
        exports.outputChannel.show(true);
        exports.outputChannel.appendLine(`\n→ meshrun ${args.join(' ')}`);
        exports.outputChannel.appendLine('─'.repeat(40));
        const child = (0, child_process_1.spawn)('meshrun', args, {
            shell: true,
            env: { ...process.env }
        });
        let output = '';
        child.stdout?.on('data', (data) => {
            const text = data.toString();
            output += text;
            exports.outputChannel.append(text);
        });
        child.stderr?.on('data', (data) => {
            const text = data.toString();
            exports.outputChannel.append(text);
        });
        child.on('close', (code) => {
            exports.outputChannel.appendLine('─'.repeat(40));
            if (code === 0 || code === null) {
                resolve(output);
            }
            else {
                reject(new Error(`meshrun exited with code ${code}`));
            }
        });
        child.on('error', (err) => {
            exports.outputChannel.appendLine(`Error: ${err.message}`);
            reject(err);
        });
    });
}
async function promptInput(prompt, placeholder) {
    return vscode.window.showInputBox({ prompt, placeHolder: placeholder });
}
//# sourceMappingURL=meshrunProcess.js.map