// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the trial's brief (the provider directory, the sealed quality table, the workload
// manifest), probes each provider's catalog route in the provider's own wire, registers the catalog
// it found, and then serves the client's twenty requests: each one routed to the best free model that
// can serve it, failing over across providers when one refuses. It polls GET /work and exits 0 when
// the env says the workload is done. `node main.ts --list` prints the catalog document it registers,
// byte for byte, and outside a trial (no brief, no env) prints {"providers":[]}.
//
// How a request is served (lib/plan.ts ranks, lib/wire.ts translates):
//   1. the request is read into a canonical form with the providers' own parsing rules — the system
//      text, the last user text, the last image, the tool names, the output cap, the stops, JSON mode;
//   2. candidates are the free models meeting the manifest's requirements for the request, ranked by
//      rating, ties spread across providers;
//   3. a candidate on the client's own wire receives the client's body untouched but for the model
//      name, and its answer streams back untouched; a candidate on another wire receives a body built
//      from the canonical form (checked, before it is sent, to fingerprint identically), and its
//      answer is re-emitted in the client's wire with the same text and usage;
//   4. a refusal is read for its cause: a spent budget, an outage or a broken route retires the
//      provider for the trial; a scheduled refusal costs only that call, and the next candidate is
//      tried. Exactly one provider call may answer a request, so the loop stops at the first success.
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers
// (the ledger counts calls; nothing reads a clock or a random source), so a replay of the same trial
// makes the same calls and leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
} from './lib/providers.ts';
import { errorMessageOf, Ledger, rankCandidates, refusalIsFinal, requirementsFromBody } from './lib/plan.ts';
import {
  anthropicError,
  anthropicMessage,
  anthropicModelList,
  anthropicStreamText,
  buildAnthropicBody,
  buildGeminiBody,
  buildOpenAIBody,
  canonFromAnthropic,
  canonFromOpenAI,
  echoOfBody,
  echoOfCanon,
  openAICompletion,
  openAIError,
  openAIModelList,
  openAIStreamText,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  sameEcho,
  type Canon,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the router moves on, in milliseconds. */
const ATTEMPT_TIMEOUT_MS = 8000;
/** The most provider calls one client request may cost (refusals included) before it is given up. */
const MAX_ATTEMPTS = 8;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
const ledger = new Ledger();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  // A failed probe drops its provider from the catalog and from routing alike, so a provider whose
  // first probe failed is probed once more before it is given up: one transient failure must not
  // cost a whole provider. The retry is the same call with the same inputs, so the catalog a live
  // pass builds is the one its replay builds.
  probes = await Promise.all(
    probes.map(async (probe) => {
      if (probe.ok) return probe;
      const again = await probeProvider(ENV_URL, probe.provider, PROBE_TIMEOUT_MS);
      return again.ok ? again : probe;
    }),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The manifest row of the request now arriving. GET /work names the request the client is sending;
 * the arrival pointer is the fallback when the env cannot be asked. A row whose api disagrees with
 * the route the request arrived on is distrusted: the body's own requirements are used instead.
 */
async function manifestEntryFor(clientWire: 'openai' | 'anthropic') {
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0) return null;
  let entry = null;
  try {
    const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      const work: unknown = await res.json();
      if (isRecord(work) && typeof work.next === 'string') {
        entry = manifest.find((m) => m.id === work.next) ?? null;
      }
    }
  } catch {
    // The env could not be asked: the pointer below is the record of arrival order.
  }
  if (entry !== null) {
    manifestIndex = manifest.indexOf(entry) + 1;
  } else {
    entry = manifest[manifestIndex] ?? null;
    manifestIndex += 1;
  }
  if (entry !== null && entry.api !== clientWire) return null;
  return entry;
}

/** The provider call's URL, headers and body for one candidate. */
function callFor(
  provider: ProviderDirectoryEntry,
  modelId: string,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  canon: Canon,
): { url: string; headers: Record<string, string>; payload: Json } {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (canon.stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'gemini') {
    const verb = canon.stream ? 'streamGenerateContent' : 'generateContent';
    const url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:${verb}${canon.stream ? '?alt=sse' : ''}`;
    return { url, headers, payload: buildGeminiBody(canon) };
  }
  if (provider.wire === clientWire) {
    // Same wire: the client's body passes through with only the model named. A streamed OpenAI call
    // also asks for the final usage chunk, which the fingerprint never reads but a served answer needs.
    const payload: Json = { ...body, model: modelId };
    if (provider.wire === 'openai' && canon.stream) {
      const options = isRecord(body.stream_options) ? body.stream_options : {};
      payload.stream_options = { ...options, include_usage: true };
    }
    return { url: `${base}${provider.wire === 'openai' ? '/v1/chat/completions' : '/v1/messages'}`, headers, payload };
  }
  if (provider.wire === 'openai') return { url: `${base}/v1/chat/completions`, headers, payload: buildOpenAIBody(canon, modelId) };
  return { url: `${base}/v1/messages`, headers, payload: buildAnthropicBody(canon, modelId) };
}

/**
 * One attempt at one candidate. Answers the client and returns true when the provider served the
 * request; records what the refusal taught the ledger and returns false when it did not.
 */
async function attempt(
  candidate: { probe: Probe; model: { id: string } },
  clientWire: 'openai' | 'anthropic',
  body: Json,
  canon: Canon,
  res: ServerResponse,
): Promise<boolean> {
  const provider = candidate.probe.provider;
  const sameWire = provider.wire === clientWire;
  const call = callFor(provider, candidate.model.id, clientWire, body, canon);
  if (!sameWire) {
    // A translated body must fingerprint exactly as the client's body did; the builders are written
    // to guarantee it, and a disagreement is logged rather than silently shipped.
    const expected = echoOfCanon(canon);
    const built = echoOfBody(provider.wire, call.payload);
    if (!sameEcho(expected, built)) {
      log(`echo drift on ${provider.id}/${candidate.model.id}: built ${JSON.stringify(built)} for ${JSON.stringify(expected)}`);
    }
  }
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.payload),
      signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
    });
  } catch {
    return false;
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    if (refusalIsFinal(upstream.status, errorMessageOf(text))) ledger.markOut(provider.id);
    return false;
  }
  // The provider has answered and the call stands as this request's one successful call, whatever
  // happens next: failing over to a second success would break the one-provider rule. A body that
  // cannot even be read therefore ends the request with a wire-shaped error instead of a retry.
  let text: string;
  try {
    text = await upstream.text();
  } catch {
    sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered but its body could not be read');
    return true;
  }
  if (sameWire) {
    const contentType = upstream.headers.get('content-type') ?? (canon.stream ? 'text/event-stream' : 'application/json');
    res.writeHead(200, { 'content-type': contentType });
    res.end(text);
    return true;
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, canon.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, canon.stream)
        : parseGeminiAnswer(text, canon.stream);
  if (answer === null) return false;
  if (canon.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    res.end(clientWire === 'openai' ? openAIStreamText(answer, candidate.model.id) : anthropicStreamText(answer, candidate.model.id));
  } else if (clientWire === 'openai') {
    sendJson(res, 200, openAICompletion(answer, candidate.model.id));
  } else {
    sendJson(res, 200, anthropicMessage(answer, candidate.model.id));
  }
  return true;
}

/** Serve one client request: rank the candidates, try them in order until one answers. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const canon = clientWire === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const entry = await manifestEntryFor(clientWire);
  const requires = entry !== null ? entry.requires : requirementsFromBody(body);
  const candidates = rankCandidates(probes, brief.ratings, requires, clientWire, canon, ledger);
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (ledger.isOut(candidate.probe.provider.id)) continue;
    attempts += 1;
    ledger.recordAttempt(candidate.probe.provider.id);
    if (await attempt(candidate, clientWire, body, canon, res)) return;
  }
  sendError(
    res,
    clientWire,
    502,
    clientWire === 'anthropic' ? 'api_error' : 'upstream_error',
    'every free provider refused or failed this request',
  );
}

/** Inference requests are served one at a time: the client sends them that way, and providers fine concurrency. */
let chain: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void>): Promise<void> {
  const run = chain.then(task, task);
  chain = run.catch(() => undefined);
  return run;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const clientWire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) {
      sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    await enqueue(() => serve(clientWire, body, res));
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** POST /register with the port and the one catalog; retried, because the trial does not start without it. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report });
  let lastError = '';
  for (let i = 0; i < 10; i++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(5000),
      });
      const text = await res.text();
      if (res.ok) {
        const reply: unknown = JSON.parse(text);
        if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
        return;
      }
      lastError = `HTTP ${res.status}: ${text.slice(0, 200)}`;
    } catch (err) {
      lastError = (err as Error).message;
    }
    await sleep(250);
  }
  log(`register never answered ok (${lastError}); polling /work anyway`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
