// The wires: client requests in, provider requests out, provider answers back to the client.
//
// The providers answer a request with a fingerprint of what they parsed out of it: the system text,
// the last user text, the last image, the tool names, the output cap, the stop sequences and whether
// JSON mode was on. A served request is one where the fingerprint the provider computed matches the
// fingerprint of the body the client sent. So every request is first read into one canonical form
// (Canon) with exactly the parsing rules the providers use, then written out in the target provider's
// wire so that wire's parser reads back the same Canon. When the client's wire and the provider's wire
// are the same, no translation happens at all: the body passes through with only the model field set
// (main.ts does that), which preserves the fingerprint trivially.
//
// Answers travel back the other way: a provider's JSON or SSE answer is read into {text, usage} and
// re-emitted in the client's wire with the wire's own event order, stop reason and usage fields. The
// answer text itself is never touched: it must arrive whole.
//
// Everything here is a pure function of its arguments: no clock, no randomness. The model listings and
// completions use fixed timestamps for the same reason.

import { createHash } from 'node:crypto';
import { isRecord, type Json, type Wire } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** Streamed answer text is cut into deltas of this many characters, as the providers cut theirs. */
const SSE_DELTA_CHARS = 24;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The first 8 hex digits of a SHA-256: how the providers fingerprint texts and image bytes. */
export function sha8Text(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 8);
}

export function sha8Bytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}

/** The bytes a base64 string carries; invalid characters make it no image at all. */
function base64Bytes(text: string): Buffer | null {
  const cleaned = text.replace(/\s+/g, '');
  if (cleaned === '' || !/^[A-Za-z0-9+/]*={0,2}$/.test(cleaned)) return null;
  try {
    return Buffer.from(cleaned, 'base64');
  } catch {
    return null;
  }
}

/** The text of a string or of a block array's text blocks, joined with newlines; '' when there is none. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** An image a request carries: inline bytes (base64 plus its media type) or a URL the provider reads. */
export type CanonImage = { kind: 'data'; mediaType: string; data: string } | { kind: 'url'; url: string };

/** The fingerprint an image hashes to, under the rule every wire's parser shares. */
export function imageSha8(image: CanonImage): string | null {
  if (image.kind === 'url') {
    if (image.url.startsWith('data:')) {
      const parsed = dataUrlOf(image.url);
      if (parsed === null) return null;
      const bytes = base64Bytes(parsed.data);
      return bytes === null ? null : sha8Bytes(bytes);
    }
    return sha8Text(image.url);
  }
  const bytes = base64Bytes(image.data);
  return bytes === null ? null : sha8Bytes(bytes);
}

/** A `data:` URL split into media type and base64 payload; null when it is not a base64 data URL. */
function dataUrlOf(url: string): { mediaType: string; data: string } | null {
  if (!url.startsWith('data:')) return null;
  const comma = url.indexOf(',');
  if (comma === -1) return null;
  const header = url.slice('data:'.length, comma);
  if (!header.endsWith(';base64')) return null;
  const mediaType = header.slice(0, header.length - ';base64'.length);
  return { mediaType: mediaType === '' ? 'application/octet-stream' : mediaType, data: url.slice(comma + 1) };
}

/** One tool definition, wire-neutral: the name is what the fingerprint reads; the rest rides along. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

/**
 * The canonical form of one client request: exactly the fields the providers fingerprint, read with
 * the providers' own parsing rules, so a body built from it in any wire parses back to the same form.
 */
export interface Canon {
  systemSeen: boolean;
  systemText: string;
  userText: string;
  image: CanonImage | null;
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** An OpenAI image_url part read as a Canon image; null when the part carries no readable image. */
function openAIImageOf(part: Json): CanonImage | null {
  const imageUrl = isRecord(part.image_url) ? part.image_url : null;
  const url = imageUrl !== null && typeof imageUrl.url === 'string' ? imageUrl.url : null;
  if (url === null) return null;
  if (url.startsWith('data:')) {
    const parsed = dataUrlOf(url);
    return parsed === null ? null : { kind: 'data', mediaType: parsed.mediaType, data: parsed.data };
  }
  return { kind: 'url', url };
}

/** An Anthropic image block read as a Canon image. */
function anthropicImageOf(block: Json): CanonImage | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'data', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    if (source.url.startsWith('data:')) {
      const parsed = dataUrlOf(source.url);
      return parsed === null ? null : { kind: 'data', mediaType: parsed.mediaType, data: parsed.data };
    }
    return { kind: 'url', url: source.url };
  }
  return null;
}

/** A Chat Completions request read into its Canon (OpenAI parsing rules: last user message wins). */
export function canonFromOpenAI(body: Json): Canon {
  const canon: Canon = { systemSeen: false, systemText: '', userText: '', image: null, tools: [], maxTokens: null, stops: [], jsonMode: false, stream: body.stream === true };
  const systemTexts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      canon.systemSeen = true;
      systemTexts.push(blocksText(raw.content));
      continue;
    }
    if (raw.role !== 'user') continue;
    canon.userText = blocksText(raw.content);
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (isRecord(part) && part.type === 'image_url') {
          const image = openAIImageOf(part);
          if (image !== null) canon.image = image;
        }
      }
    }
  }
  canon.systemText = systemTexts.join('\n');
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      canon.tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  canon.maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  canon.stops = stringList(body.stop);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  canon.jsonMode = format === 'json_object' || format === 'json_schema';
  return canon;
}

/** A Messages request read into its Canon (Anthropic parsing rules: system is a field, JSON mode never). */
export function canonFromAnthropic(body: Json): Canon {
  const canon: Canon = { systemSeen: false, systemText: '', userText: '', image: null, tools: [], maxTokens: null, stops: [], jsonMode: false, stream: body.stream === true };
  if (body.system !== undefined) {
    canon.systemSeen = true;
    canon.systemText = blocksText(body.system);
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    canon.userText = blocksText(raw.content);
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (isRecord(block) && block.type === 'image') {
          const image = anthropicImageOf(block);
          if (image !== null) canon.image = image;
        }
      }
    }
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      canon.tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  canon.maxTokens = numberOrNull(body.max_tokens);
  canon.stops = stringList(body.stop_sequences);
  return canon;
}

/** The fingerprint a Canon hashes to: what a provider's answer must carry for the request to count. */
export interface Echo {
  systemPresent: boolean;
  systemSha8: string | null;
  userSha8: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  imageSha8: string | null;
  jsonMode: boolean;
}

export function echoOfCanon(canon: Canon): Echo {
  return {
    systemPresent: canon.systemSeen,
    systemSha8: canon.systemSeen ? sha8Text(canon.systemText) : null,
    userSha8: sha8Text(canon.userText),
    tools: canon.tools.map((t) => t.name),
    maxTokens: canon.maxTokens,
    stops: [...canon.stops],
    imageSha8: canon.image === null ? null : imageSha8(canon.image),
    jsonMode: canon.jsonMode,
  };
}

/** A Canon image as an OpenAI content part. */
function openAIImagePart(image: CanonImage): Json {
  if (image.kind === 'data') return { type: 'image_url', image_url: { url: `data:${image.mediaType};base64,${image.data}` } };
  return { type: 'image_url', image_url: { url: image.url } };
}

/** A Canon image as an Anthropic content block. */
function anthropicImageBlock(image: CanonImage): Json {
  if (image.kind === 'data') return { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } };
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** A Canon image as a Gemini content part. */
function geminiImagePart(image: CanonImage): Json {
  if (image.kind === 'data') return { inlineData: { mimeType: image.mediaType, data: image.data } };
  return { fileData: { fileUri: image.url } };
}

/**
 * A Canon as a Chat Completions body naming `model`. One user message carries the image (if any) and
 * the last user text exactly, so the provider's parser reads the same fingerprint the client sent.
 */
export function buildOpenAIBody(canon: Canon, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemSeen) messages.push({ role: 'system', content: canon.systemText });
  if (canon.image !== null) {
    messages.push({ role: 'user', content: [openAIImagePart(canon.image), { type: 'text', text: canon.userText }] });
  } else {
    messages.push({ role: 'user', content: canon.userText });
  }
  const out: Json = { model, messages };
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) out.stop = [...canon.stops];
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    // Without this an OpenAI-wire provider's stream carries no usage, and usage is part of a served answer.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Canon as a Messages body naming `model` (the Anthropic wire carries no JSON mode; routing keeps it that way). */
export function buildAnthropicBody(canon: Canon, model: string): Json {
  const content: Json[] = [];
  if (canon.image !== null) content.push(anthropicImageBlock(canon.image));
  content.push({ type: 'text', text: canon.userText });
  const out: Json = { model, messages: [{ role: 'user', content }] };
  if (canon.systemSeen) out.system = canon.systemText;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) out.stop_sequences = [...canon.stops];
  if (canon.stream) out.stream = true;
  return out;
}

/** A Canon as a Gemini generateContent body (the model rides in the call's path, never in the body). */
export function buildGeminiBody(canon: Canon): Json {
  const parts: Json[] = [];
  if (canon.image !== null) parts.push(geminiImagePart(canon.image));
  parts.push({ text: canon.userText });
  const out: Json = { contents: [{ role: 'user', parts }] };
  if (canon.systemSeen) out.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    out.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) generationConfig.stopSequences = [...canon.stops];
  if (canon.jsonMode) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/**
 * The fingerprint a body in `wire` parses to, computed with the same rules the providers use. The
 * builders are written to make this equal echoOfCanon(canon) for the Canon the body was built from;
 * main.ts checks that equality before every translated call and falls back if it ever broke.
 */
export function echoOfBody(wire: Wire, body: Json, geminiModel = '', geminiStream = false): Echo {
  if (wire === 'openai') {
    const canon = canonFromOpenAI(body);
    return echoOfCanon(canon);
  }
  if (wire === 'anthropic') {
    const canon = canonFromAnthropic(body);
    return echoOfCanon(canon);
  }
  // Gemini rules: system from systemInstruction parts, the last non-model content's text, images from
  // inlineData/fileData, tools from functionDeclarations, caps and stops from generationConfig.
  let systemSeen = false;
  let systemText = '';
  if (isRecord(body.systemInstruction)) {
    const text = geminiPartsText(body.systemInstruction.parts);
    if (text !== null) {
      systemSeen = true;
      systemText = text;
    }
  }
  let userText = '';
  let image: CanonImage | null = null;
  const contents = Array.isArray(body.contents) ? body.contents : [];
  for (const raw of contents) {
    if (!isRecord(raw) || raw.role === 'model') continue;
    const text = geminiPartsText(raw.parts);
    if (text !== null) userText = text;
    if (Array.isArray(raw.parts)) {
      for (const part of raw.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') {
          image = { kind: 'data', mediaType: typeof part.inlineData.mimeType === 'string' ? part.inlineData.mimeType : 'image/png', data: part.inlineData.data };
        } else if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') {
          image = { kind: 'url', url: part.fileData.fileUri };
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !Array.isArray(raw.functionDeclarations)) continue;
      for (const decl of raw.functionDeclarations) {
        if (isRecord(decl) && typeof decl.name === 'string') tools.push(decl.name);
      }
    }
  }
  const config = isRecord(body.generationConfig) ? body.generationConfig : {};
  const canon: Canon = {
    systemSeen,
    systemText,
    userText,
    image,
    tools: tools.map((name) => ({ name, description: '', parameters: {} })),
    maxTokens: numberOrNull(config.maxOutputTokens),
    stops: stringList(config.stopSequences),
    jsonMode: config.responseMimeType === 'application/json',
    stream: geminiStream,
  };
  void geminiModel;
  return echoOfCanon(canon);
}

/** The text of Gemini parts: their text members joined with newlines, null when a part is not text-shaped. */
function geminiPartsText(parts: unknown): string | null {
  if (!Array.isArray(parts)) return null;
  const texts: string[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
  }
  if (texts.length === 0 && parts.length > 0) return null;
  return texts.join('\n');
}

/** Whether two fingerprints agree on every field the client compares. */
export function sameEcho(a: Echo, b: Echo): boolean {
  return (
    a.systemPresent === b.systemPresent &&
    a.systemSha8 === b.systemSha8 &&
    a.userSha8 === b.userSha8 &&
    a.maxTokens === b.maxTokens &&
    a.imageSha8 === b.imageSha8 &&
    a.jsonMode === b.jsonMode &&
    a.tools.length === b.tools.length &&
    a.tools.every((t, i) => t === b.tools[i]) &&
    a.stops.length === b.stops.length &&
    a.stops.every((s, i) => s === b.stops[i])
  );
}

/** One provider's answer, read off its wire: the answer text whole, and the usage it reported. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** SSE frames' data payloads, in order. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** SSE frames with their event names, for the Anthropic stream's named events. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions answer (JSON body or SSE stream) read into text and usage. */
export function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof choice.message.content === 'string' ? choice.message.content : blocksText(choice.message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
    };
  }
  let out = '';
  let input = 0;
  let output = 0;
  let saw = false;
  for (const payload of ssePayloads(text)) {
    if (payload.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    saw = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
  }
  return saw ? { text: out, inputTokens: input, outputTokens: output } : null;
}

/** A Messages answer (JSON body or SSE stream) read into text and usage. */
export function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: blocksText(body.content), inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
  }
  let out = '';
  let input = 0;
  let output = 0;
  let saw = false;
  for (const frame of sseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    saw = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta' && isRecord(data.usage)) {
      output = numberOrZero(data.usage.output_tokens);
    }
  }
  return saw ? { text: out, inputTokens: input, outputTokens: output } : null;
}

/** A Gemini answer (JSON body or SSE stream) read into text and usage. */
export function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer | null {
  const read = (body: Json, acc: ProviderAnswer): boolean => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return false;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') acc.text += part.text;
    }
    if (isRecord(body.usageMetadata)) {
      acc.inputTokens = numberOrZero(body.usageMetadata.promptTokenCount);
      acc.outputTokens = numberOrZero(body.usageMetadata.candidatesTokenCount);
    }
    return true;
  };
  const acc: ProviderAnswer = { text: '', inputTokens: 0, outputTokens: 0 };
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    return read(body, acc) ? acc : null;
  }
  let saw = false;
  for (const payload of ssePayloads(text)) {
    const body = jsonOrNull(payload);
    if (body === null) continue;
    if (read(body, acc)) saw = true;
  }
  return saw ? acc : null;
}

/** One server-sent event frame: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** A provider answer as a Chat Completions JSON body naming the model that served it. */
export function openAICompletion(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A provider answer as a Messages JSON body naming the model that served it. */
export function anthropicMessage(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/**
 * A provider answer as a Chat Completions SSE stream: a role chunk, the text in deltas, a finish
 * chunk, a final usage chunk and [DONE] — the event order the wire's readers require.
 */
export function openAIStreamText(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A provider answer as a Messages SSE stream: message_start (with the input usage), one text block's
 * start/deltas/stop, message_delta with the stop reason and output usage, then message_stop.
 */
export function anthropicStreamText(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
