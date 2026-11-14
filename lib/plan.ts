// Routing: which free model serves a request, and what a refusal teaches the router.
//
// The plan is greedy with a memory, the way a free-tier gateway is actually run:
//
//   - candidates are every free model, on a provider that answered its probe and is not known to be
//     out, whose capabilities meet the request's requirements — ranked by the quality table's rating,
//     best first (quality is the score's largest column and the rating gaps between adjacent models
//     are small, so serving at all beats squeezing the last milli-points);
//   - ties spread across providers: a provider this server has already called more often yields to
//     one it has called less, then the client's own wire wins (a same-wire call passes through
//     untouched), then provider and model ids settle the rest — every key a function of the brief and
//     of the calls already made, so a replay walks the same list;
//   - a refusal is read before it is believed: the providers say why (their budget is spent, they are
//     out, or this one call was refused), and only the first two retire a provider for the trial. A
//     call that names no listed free model or carries a wrong key (400/401) retires it too — the
//     router never sends those on purpose, so one means the route itself is broken;
//   - an Anthropic-wire provider can never serve a request whose client asked for JSON mode: that
//     wire cannot carry it, so its fingerprint would never match. Those candidates are excluded up
//     front instead of failing one request at a time.

import {
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './providers.ts';
import type { Canon } from './wire.ts';

/** One routable pick: a provider that answered, and one of its free models. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * What the trial has taught the router about each provider: how many inference calls it has been
 * sent, and whether it is out for the rest of the trial (budget spent, outage, or a broken route).
 * The catalog probes are not inference calls and are not counted here.
 */
export class Ledger {
  private attempts = new Map<string, number>();
  private out = new Set<string>();

  attemptsOf(providerId: string): number {
    return this.attempts.get(providerId) ?? 0;
  }

  recordAttempt(providerId: string): void {
    this.attempts.set(providerId, this.attemptsOf(providerId) + 1);
  }

  markOut(providerId: string): void {
    this.out.add(providerId);
  }

  isOut(providerId: string): boolean {
    return this.out.has(providerId);
  }
}

/**
 * Every candidate for one request, best first. `clientWire` is the wire the client used (openai or
 * anthropic); a candidate on the same wire passes the request through untouched.
 */
export function rankCandidates(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  clientWire: Wire,
  canon: Canon,
  ledger: Ledger,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (ledger.isOut(probe.provider.id)) continue;
    // The Anthropic wire cannot echo JSON mode: a client that asked for it must not be routed there.
    if (canon.jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const spread = ledger.attemptsOf(a.probe.provider.id) - ledger.attemptsOf(b.probe.provider.id);
    if (spread !== 0) return spread;
    const aSame = a.probe.provider.wire === clientWire ? 0 : 1;
    const bSame = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * What one refusal means. The providers' error bodies name the cause: a spent budget and an outage
 * both last the rest of the trial; a scheduled refusal costs only that call. A 400 or 401 is the
 * router's own fault (a model the provider does not list as free, or a wrong key) and retires the
 * route rather than burning a call on it per request.
 */
export function refusalIsFinal(status: number, message: string): boolean {
  if (status === 400 || status === 401) return true;
  if (message.includes('budget is spent')) return true;
  if (message.endsWith(' is out')) return true;
  return false;
}

/** The message inside a provider's error body, whichever wire shaped it; '' when there is none. */
export function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
      if (typeof value.message === 'string') return value.message;
    }
  } catch {
    // Not JSON: no message to read.
  }
  return '';
}

/** Requirements read off a request body itself, for a brief whose manifest has no row for it. */
export function requirementsFromBody(body: Record<string, unknown>): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}
