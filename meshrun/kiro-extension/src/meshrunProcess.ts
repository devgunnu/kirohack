import { spawn, ChildProcess } from 'child_process';
import * as vscode from 'vscode';

export const outputChannel = vscode.window.createOutputChannel('MeshRun');

/**
 * Runs a meshrun CLI command and streams output to the MeshRun Output Channel.
 * Built on top of the existing meshrun CLI installed via pip install -e .
 */
export function runMeshrun(args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
        outputChannel.show(true);
        outputChannel.appendLine(`\n→ meshrun ${args.join(' ')}`);
        outputChannel.appendLine('─'.repeat(40));

        const child: ChildProcess = spawn('meshrun', args, {
            shell: true,
            env: { ...process.env }
        });

        let output = '';

        child.stdout?.on('data', (data: Buffer) => {
            const text = data.toString();
            output += text;
            outputChannel.append(text);
        });

        child.stderr?.on('data', (data: Buffer) => {
            const text = data.toString();
            outputChannel.append(text);
        });

        child.on('close', (code: number | null) => {
            outputChannel.appendLine('─'.repeat(40));
            if (code === 0 || code === null) {
                resolve(output);
            } else {
                reject(new Error(`meshrun exited with code ${code}`));
            }
        });

        child.on('error', (err: Error) => {
            outputChannel.appendLine(`Error: ${err.message}`);
            reject(err);
        });
    });
}

export async function promptInput(
    prompt: string,
    placeholder: string
): Promise<string | undefined> {
    return vscode.window.showInputBox({ prompt, placeHolder: placeholder });
}
