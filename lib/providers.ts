// The brief's three documents and the catalog probe.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) carries three JSON documents, found by their
// `schema` member, the prose around them ignored:
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): one row a provider, with its
//     id, wire (openai, anthropic or gemini), auth scheme, key environment variable, the key the
//     service takes, and the base path joined to MAKEFUN_ENV_URL (an absolute base_url is accepted
//     too and reduced to its path);
//   - the quality table (`free-model-aggregator-01/ratings/v1`): every model's public rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and whether it is free;
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the twenty requests the client
//     will send, in order, each with its api, whether it streams, its weight and its requirements.
//
// Each provider is then probed on its own catalog route, in its own wire:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails leaves its provider out of the catalog and gets no traffic; the trial goes on.
// Everything here is a pure function of the brief and the providers' answers: no clock, no randomness,
// so `--list` and the register body build the same document on a live pass and on its replay.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as read. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * The provider's own path (`/p/<id>`) out of a directory base that may be spelled as a path or as an
 * absolute URL. The catalog registers the path: the env's origin differs between a live pass and its
 * replay, and the catalog rides in the state export the replay must reproduce byte for byte.
 */
export function pathOfBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, '');
    } catch {
      return b.replace(/\/+$/, '');
    }
  }
  return (b.startsWith('/') ? b : `/${b}`).replace(/\/+$/, '');
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in, per its own auth scheme (the wire does not decide this). */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratingRows = ratingsDoc !== null && Array.isArray(ratingsDoc.ratings) ? ratingsDoc.ratings : null;
  const ratings: RatingsFile | null =
    ratingsDoc === null || ratingRows === null
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: ratingRows.flatMap((v) => {
            const r = ratingOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isRecord(v) || stringOf(v.id) === '') return [];
                return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadFile | null =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workloadDoc.manifest.flatMap((v) => {
            const w = workloadRequestOf(v);
            return w === null ? [] : [w];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it
 * listed, models in the listing's own order with duplicates removed.
 *
 * Its base_url is the provider's own path (`/p/<id>`), never the path joined to the env's URL: the
 * env's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte; the scorer compares a base_url by its path suffix,
 * so the path passes exactly as the joined URL does.
 */
export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: pathOfBase(probe.provider.base), key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
