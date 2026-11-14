# Free Model Aggregator

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API — in JSON and
in server-sent events — and routes every request to the best free model that can serve it, across
ten free-tier providers that each speak their own wire (OpenAI, Anthropic or Gemini), carry their
own auth scheme, and run dry on their own schedule.

Built for the make.fun competition `free-model-aggregator-01`, and written the way a production
free-tier gateway is run, because that is what it is: point it at real providers with real keys and
the same routing, failover and wire translation applies.

## Running it

In a trial the harness runs

```
node main.ts
```

with `MAKEFUN_PROMPT_FILE` (the brief: provider directory, quality table, workload manifest) and
`MAKEFUN_ENV_URL` (the env serving the providers) set. The server probes each provider's catalog,
registers its port and the catalog it found, serves the client's requests, polls `GET /work`, and
exits 0 when the workload is done.

```
node main.ts --list
```

prints the catalog document the server registers — every provider, its base path, key variable and
wire, and each model with its free flag and context length — byte for byte as registered. Run
outside a trial it prints `{"providers":[]}`.

## How routing works

- **Canonical form first.** A request is read with the parsing rules the providers themselves use —
  the system text, the last user text, the last image, the tool names, the output cap, the stop
  sequences, and whether JSON mode was asked for. Everything downstream works from that form.
- **Quality-ranked candidates.** For each request, the free models whose capabilities meet the
  request's requirements (tools, vision, JSON mode, minimum context) are ranked by the quality
  table's rating. Ties spread across providers by how much each has already been called, prefer the
  client's own wire, and settle on provider/model ids — a pure function of the brief and the calls
  already made, so a replay makes the same calls.
- **Same wire passes through.** When the chosen provider speaks the client's wire, the client's body
  is forwarded with only the model field set, and the provider's answer returns untouched.
- **Other wires are translated.** The canonical form is written out in the provider's wire (OpenAI,
  Anthropic, or Gemini — including Gemini's path-named models and `generationConfig`), verified to
  fingerprint identically to the client's request before it is sent, and the answer is re-emitted in
  the client's wire with its text whole and its usage mapped into the client's field names.
- **Refusals are read, not just counted.** A provider that answers 402/429/5xx is asked why: a spent
  budget or an outage retires it for the rest of the trial; a one-call refusal just moves the router
  to the next candidate. A provider that serves one call at a time is never called concurrently —
  requests are served strictly one at a time. Exactly one provider call answers a request; the
  failover loop stops at the first success.
- **The catalog is the truth the providers told.** `GET /v1/models` answers in the wire its auth
  header names (Bearer → OpenAI list shape, `x-api-key` → Anthropic list shape) and lists exactly
  the free models the registered catalog holds. Paid models are listed in the catalog with
  `free: false` and are never routed to and never listed as servable.

## Files

- `main.ts` — the server: brief loading, registration, routes, the failover loop, the `/work` poll.
- `lib/providers.ts` — the brief's three documents, the per-wire catalog probe, the catalog builder.
- `lib/wire.ts` — the canonical form, the per-wire body builders and fingerprint checks, the
  provider-answer parsers and the client-wire emitters (JSON and SSE).
- `lib/plan.ts` — candidate ranking, the per-provider ledger, and refusal classification.

## License

MIT — see `LICENSE`.
