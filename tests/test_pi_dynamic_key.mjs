// Offline integration fixture, launched by test_pi_q6.py. No real credentials or AWS.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { createServer } from 'node:http';
import { pathToFileURL, fileURLToPath } from 'node:url';

const [piDir, modelsPath, python] = process.argv.slice(2);
const { ModelRuntime } = await import(pathToFileURL(`${piDir}/dist/core/model-runtime.js`));
const { AuthStorage } = await import(pathToFileURL(`${piDir}/dist/core/auth-storage.js`));
const provider = 'sagemaker-qwen3-q6';
const seen = [];
let finishResponse;
const server = createServer(async (req, res) => {
    try {
        assert.equal(req.url, '/v1/chat/completions');
        assert.equal(req.method, 'POST');
        let body = '';
        for await (const chunk of req) body += chunk;
        assert.equal(JSON.parse(body).stream, true);
        seen.push(req.headers.authorization);
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        const chunk = (delta, finish_reason = null) => `data: ${JSON.stringify({
            id: 'local', object: 'chat.completion.chunk', created: 1, model: 'local',
            choices: [{ index: 0, delta, finish_reason }],
        })}\n\n`;
        // Only end after the client receives a delta: proves incremental SSE, not buffering.
        finishResponse = () => res.end(chunk({}, 'stop') + 'data: [DONE]\n\n');
        res.write(chunk({ role: 'assistant', content: 'OK' }));
    } catch (error) {
        res.destroy(error);
    }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
try {
    const document = JSON.parse(readFileSync(modelsPath, 'utf8'));
    document.providers = { [provider]: document.providers[provider] };
    document.providers[provider].baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
    writeFileSync(modelsPath, JSON.stringify(document));
    const runtime = await ModelRuntime.create({
        modelsPath, credentials: AuthStorage.inMemory(), refreshOnCreate: false,
        allowModelNetwork: false,
    });
    assert.equal(runtime.getError(), undefined); // Validate actual Pi models.json schema.
    const model = runtime.getModel(provider, 'qwen38-27b-uncensored-q6-sm');
    assert.ok(model);
    const context = { messages: [{ role: 'user', content: 'Say OK', timestamp: Date.now() }] };
    async function request() {
        const stream = runtime.streamSimple(model, context, { maxTokens: 8 });
        let text = '';
        for await (const event of stream) {
            if (event.type === 'text_delta') {
                text += event.delta;
                finishResponse();
            }
        }
        const result = await stream.result();
        assert.equal(result.stopReason, 'stop', result.errorMessage);
        assert.equal(text, 'OK');
    }
    await request();
    // Use the real Python refresh path with mocked minting; keep the same Pi runtime/model.
    const report = JSON.parse(execFileSync(python, ['-c',
        'import json, sys; from register_pi_model import register; from pathlib import Path; print(json.dumps(register(json.loads(Path("../deployment_config.example.json").read_text())["Qwen3-27B-Uncensored-Q6"], path=sys.argv[1], generator=lambda **kw: "second-token")))',
        modelsPath], { cwd: fileURLToPath(new URL('../src/', import.meta.url)), encoding: 'utf8' }));
    await request();
    assert.deepEqual(seen, ['Bearer first-token', 'Bearer second-token']);
    unlinkSync(report.token_path);
    const failed = await runtime.streamSimple(model, context, { maxTokens: 8 }).result();
    assert.equal(failed.stopReason, 'error');
    assert.equal(seen.length, 2); // Missing token fails closed, never reuses cached auth.
    console.log('dynamic auth + SSE passed (same runtime, two refreshed tokens)');
} finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
}
