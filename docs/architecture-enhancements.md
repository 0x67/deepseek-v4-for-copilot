# Architecture: Extension Enhancements

> Reference document for enhancing the DeepSeek V4 for Copilot extension with custom model injection, a RAG pipeline, and native Copilot model proxying.

---

## Table of Contents

- [Goals](#goals)
- [Current Architecture](#current-architecture)
- [Phase 1 — Custom Model Injection](#phase-1--custom-model-injection)
- [Phase 2 — Pipeline Framework](#phase-2--pipeline-framework)
- [Phase 3 — RAG Stages](#phase-3--rag-stages)
- [Phase 4 — Native Copilot Model Proxying](#phase-4--native-copilot-model-proxying)
- [Pipeline Routing with `anyOf` Match](#pipeline-routing-with-anyof-match)
- [LangGraph Integration](#langgraph-integration)
- [Proposed File Structure](#proposed-file-structure)
- [Configuration Reference](#configuration-reference)

---

## Goals

1. **Keep existing DeepSeek model injection** — no regressions to current functionality.
2. **Add custom model injections** — users can register arbitrary OpenAI-compatible models.
3. **Custom pipeline for RAG + Reranker + Embedding** — runs before the actual LLM call; works with native Copilot models and custom models alike.
4. **Configurable via JSON** — all settings live in VS Code settings or an external JSON file.

---

## Current Architecture

```
┌──────────────────────────────────────────────────────┐
│  VS Code Copilot Chat                                │
│    ↕ vscode.LanguageModelChatProvider                │
├──────────────────────────────────────────────────────┤
│  DeepSeekChatProvider (src/provider/index.ts)        │
│    • provideLanguageModelChatInformation → MODELS[]  │
│    • provideLanguageModelChatResponse → stream       │
├──────────────────────────────────────────────────────┤
│  prepareChatRequest (src/provider/request.ts)        │
│    • Auth, model resolution, vision, tool handling   │
├──────────────────────────────────────────────────────┤
│  DeepSeekClient (src/client/core.ts)                 │
│    • SSE streaming to DeepSeek API                   │
└──────────────────────────────────────────────────────┘
```

Key files:
- `src/consts.ts` — `MODELS[]` registry (DeepSeek V4 Flash, Pro)
- `src/types.ts` — `ModelDefinition`, `DeepSeekRequest`, etc.
- `src/config.ts` — Runtime settings readers
- `src/provider/request.ts` — Request preparation logic
- `src/client/core.ts` — HTTP/SSE client

---

## Phase 1 — Custom Model Injection

### Overview

Allow users to define additional OpenAI-compatible models via JSON config. These appear in the Copilot Chat model picker alongside the built-in DeepSeek models.

### Configuration

```jsonc
// settings.json
{
  "deepseek-copilot.customModels": [
    {
      "id": "my-local-llama",
      "name": "Local Llama 3",
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "maxInputTokens": 128000,
      "maxOutputTokens": 8192,
      "capabilities": {
        "toolCalling": true,
        "imageInput": false,
        "thinking": false
      }
    },
    {
      "id": "openrouter-claude",
      "name": "Claude 4 (OpenRouter)",
      "baseUrl": "https://openrouter.ai/api/v1",
      "apiKeyRef": "secret:openrouter-key",
      "maxInputTokens": 200000,
      "maxOutputTokens": 16384,
      "capabilities": {
        "toolCalling": true,
        "imageInput": true,
        "thinking": true
      }
    }
  ]
}
```

### Key Changes

| File | Change |
|------|--------|
| `src/types.ts` | Extend `ModelDefinition` with optional `baseUrl`, `apiKey`, `apiKeyRef` |
| `src/consts.ts` | Keep `MODELS[]` for built-ins only |
| `src/models/registry.ts` (new) | Merges built-in + custom models at runtime |
| `src/models/custom.ts` (new) | Parses & validates custom model config |
| `src/provider/index.ts` | `provideLanguageModelChatInformation` returns merged list |
| `src/provider/request.ts` | Resolves per-model `baseUrl` + `apiKey` |
| `package.json` | Add `customModels` configuration schema |

### Auth Resolution Order

1. Model-specific `apiKey` field (plaintext, for local models)
2. Model-specific `apiKeyRef` → VS Code SecretStorage lookup
3. Global DeepSeek API key (existing behavior, for built-in models only)

---

## Phase 2 — Pipeline Framework

### Overview

A middleware pipeline that intercepts requests between the VS Code provider layer and the LLM call. Each stage transforms a `PipelineContext` and passes it forward.

### Architecture

```
User Message
    │
    ▼
┌─────────────────────────────────┐
│  Pipeline Runner                 │
│                                  │
│  Stage 1: Embedding              │  → vectorize user query
│  Stage 2: RAG Retrieval          │  → fetch relevant docs
│  Stage 3: Reranker               │  → score & filter docs
│  Stage 4: Context Injection      │  → prepend context to messages
│                                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  LLM Call (DeepSeek / Custom /   │
│  Native Copilot Proxy)           │
└─────────────────────────────────┘
```

### Core Interfaces

```typescript
// src/pipeline/types.ts

export interface PipelineContext {
  /** The original user messages from VS Code */
  originalMessages: readonly vscode.LanguageModelChatRequestMessage[];
  /** Transformed messages (mutated by stages) */
  messages: DeepSeekMessage[];
  /** The resolved model definition */
  model: ModelDefinition;
  /** Retrieved documents from RAG */
  documents: RetrievedDocument[];
  /** Embeddings computed for the query */
  queryEmbedding?: number[];
  /** Arbitrary metadata stages can pass downstream */
  metadata: Record<string, unknown>;
  /** Cancellation token */
  token: vscode.CancellationToken;
}

export interface PipelineStage {
  readonly name: string;
  /** Return false to skip this stage dynamically */
  shouldRun?(ctx: PipelineContext): boolean;
  /** Process context and return updated context */
  process(ctx: PipelineContext): Promise<PipelineContext>;
}

export interface RetrievedDocument {
  id: string;
  content: string;
  score?: number;
  metadata?: Record<string, unknown>;
}
```

### Pipeline Runner

```typescript
// src/pipeline/index.ts

export class PipelineRunner {
  constructor(private readonly stages: PipelineStage[]) {}

  async run(ctx: PipelineContext): Promise<PipelineContext> {
    let current = ctx;
    for (const stage of this.stages) {
      if (stage.shouldRun && !stage.shouldRun(current)) continue;
      current = await stage.process(current);
      if (current.token.isCancellationRequested) break;
    }
    return current;
  }
}
```

### Integration Point

In `src/provider/request.ts` → `prepareChatRequest()`:

```typescript
// After convertMessages(), before building DeepSeekRequest:
const pipelineConfig = getPipelineConfig();
if (pipelineConfig.enabled && matchesModel(pipelineConfig.applyTo, modelDef)) {
  const pipeline = buildPipeline(pipelineConfig.stages);
  const pipelineCtx = await pipeline.run({
    originalMessages: messages,
    messages: deepseekMessages,
    model: modelDef,
    documents: [],
    metadata: {},
    token,
  });
  deepseekMessages = pipelineCtx.messages;
}
```

---

## Phase 3 — RAG Stages

### Embedding Stage

Calls a configurable OpenAI-compatible `/embeddings` endpoint to vectorize the user's latest message.

```typescript
// src/pipeline/stages/embedding.ts
export class EmbeddingStage implements PipelineStage {
  name = 'embedding';

  async process(ctx: PipelineContext): Promise<PipelineContext> {
    const userQuery = extractLatestUserMessage(ctx.messages);
    const embedding = await this.callEmbeddingApi(userQuery);
    return { ...ctx, queryEmbedding: embedding };
  }
}
```

**Config:**
```jsonc
{
  "type": "embedding",
  "endpoint": "http://localhost:11434/v1/embeddings",
  "model": "nomic-embed-text",
  "apiKey": ""
}
```

### RAG Retrieval Stage

Queries a vector store using the computed embedding. Returns top-K documents.

```typescript
// src/pipeline/stages/retrieval.ts
export class RetrievalStage implements PipelineStage {
  name = 'retrieval';

  async process(ctx: PipelineContext): Promise<PipelineContext> {
    if (!ctx.queryEmbedding) return ctx;
    const docs = await this.store.search(ctx.queryEmbedding, this.config.topK);
    return { ...ctx, documents: docs };
  }
}
```

**Supported stores:**

| Provider | Description |
|----------|-------------|
| `local-sqlite` | SQLite + cosine similarity (zero external deps) |
| `qdrant` | Qdrant REST API |
| `pinecone` | Pinecone REST API |
| `custom` | User-provided HTTP endpoint |

**Config:**
```jsonc
{
  "type": "retrieval",
  "provider": "local-sqlite",
  "dbPath": "~/.deepseek-copilot/vectors.db",
  "topK": 20
}
```

### Reranker Stage

Calls a reranker API to score and filter retrieved documents.

```typescript
// src/pipeline/stages/reranker.ts
export class RerankerStage implements PipelineStage {
  name = 'reranker';

  shouldRun(ctx: PipelineContext): boolean {
    return ctx.documents.length > 0;
  }

  async process(ctx: PipelineContext): Promise<PipelineContext> {
    const query = extractLatestUserMessage(ctx.messages);
    const reranked = await this.callRerankerApi(query, ctx.documents);
    return { ...ctx, documents: reranked.slice(0, this.config.topK) };
  }
}
```

**Config:**
```jsonc
{
  "type": "reranker",
  "endpoint": "https://api.jina.ai/v1/rerank",
  "model": "jina-reranker-v2",
  "apiKey": "${secret:jina-api-key}",
  "topK": 5
}
```

### Context Injection Stage

Prepends retrieved documents into the conversation as context.

```typescript
// src/pipeline/stages/context-injection.ts
export class ContextInjectionStage implements PipelineStage {
  name = 'context-injection';

  shouldRun(ctx: PipelineContext): boolean {
    return ctx.documents.length > 0;
  }

  async process(ctx: PipelineContext): Promise<PipelineContext> {
    const contextText = ctx.documents
      .map((d, i) => `[${i + 1}] ${d.content}`)
      .join('\n\n');
    const injected = this.config.template.replace('{{documents}}', contextText);
    // Prepend as system message or inject at configured position
    const messages = injectContext(ctx.messages, injected, this.config.position);
    return { ...ctx, messages };
  }
}
```

**Config:**
```jsonc
{
  "type": "context-injection",
  "position": "system",
  "template": "Use the following retrieved context to inform your answer. Cite sources by number when applicable.\n\n{{documents}}"
}
```

---

## Phase 4 — Native Copilot Model Proxying

> **This phase is required** — it ensures the pipeline works with GitHub Copilot's native models (GPT-4o, Claude, etc.) without requiring external API keys.

### Overview

Register "proxy" models in the Copilot Chat picker that wrap native Copilot models with the full pipeline applied. The proxy:

1. Appears in the model picker (e.g., "GPT-4o + RAG")
2. Receives the user's messages
3. Runs the pipeline (embedding → retrieval → rerank → inject context)
4. Forwards the enriched messages to the underlying native Copilot model via `vscode.lm.sendChatRequest()`
5. Streams the response back

### Architecture

```
┌──────────────────────────────────────┐
│ Copilot Chat Model Picker            │
│                                      │
│  • DeepSeek V4 Flash                 │  ← built-in
│  • DeepSeek V4 Pro                   │  ← built-in
│  • Local Llama 3                     │  ← custom (Phase 1)
│  • GPT-4o + RAG                      │  ← proxy (Phase 4)
│  • Claude Sonnet + RAG               │  ← proxy (Phase 4)
└──────────────────────────────────────┘
         │ (user selects proxy)
         ▼
┌──────────────────────────────────────┐
│ ProxyChatProvider                     │
│   1. Run Pipeline (embed/rag/rerank)  │
│   2. Inject retrieved context         │
│   3. Forward to native model          │
│      via vscode.lm.sendChatRequest()  │
│   4. Stream response back             │
└──────────────────────────────────────┘
```

### Configuration

```jsonc
{
  "deepseek-copilot.proxyModels": [
    {
      "name": "GPT-4o + RAG",
      "source": {
        "vendor": "copilot",
        "family": "gpt-4o"
      },
      "pipeline": true
    },
    {
      "name": "Claude Sonnet + RAG",
      "source": {
        "vendor": "copilot",
        "family": "claude-sonnet"
      },
      "pipeline": true
    }
  ]
}
```

### Key Implementation

```typescript
// src/provider/proxy.ts

export class ProxyChatProvider implements vscode.LanguageModelChatProvider {
  async provideLanguageModelChatResponse(
    modelInfo: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    // 1. Run pipeline
    const enrichedMessages = await this.runPipeline(messages, token);

    // 2. Resolve the native Copilot model
    const [nativeModel] = await vscode.lm.selectChatModels({
      vendor: this.config.source.vendor,
      family: this.config.source.family,
    });

    // 3. Forward enriched messages to native model
    const response = await nativeModel.sendRequest(enrichedMessages, options, token);

    // 4. Stream response parts back
    for await (const part of response.stream) {
      progress.report(part);
    }
  }
}
```

### Why This Is Required

- Makes the extension valuable for users **without external API keys** — they get RAG on their existing Copilot subscription.
- Native Copilot models (GPT-4o, Claude) are already available; the proxy adds intelligence (RAG) on top.
- Unifies the pipeline across all model sources.

---

## Pipeline Routing with `anyOf` Match

The pipeline does NOT apply to all models by default. Users configure which models use the pipeline via `anyOf`-style matching rules.

### Configuration

```jsonc
{
  "deepseek-copilot.pipeline": {
    "enabled": true,
    "stages": [ /* ... */ ],
    "applyTo": {
      "anyOf": [
        { "id": "deepseek-v4-pro" },
        { "id": "my-local-llama" },
        { "family": "gpt-4o" },
        { "source": "proxy" }
      ]
    }
  }
}
```

### Matching Rules

| Rule | Matches |
|------|---------|
| `{ "id": "deepseek-v4-pro" }` | Exact model ID |
| `{ "id": "deepseek-v4-*" }` | Wildcard on model ID |
| `{ "family": "deepseek" }` | All models in a family |
| `{ "source": "proxy" }` | All proxy models |
| `{ "source": "custom" }` | All custom models |
| `{ "source": "builtin" }` | Built-in DeepSeek models |
| `"*"` | All models |

### Matching Logic

```typescript
function matchesModel(applyTo: ApplyToConfig, model: ModelDefinition): boolean {
  if (applyTo === '*') return true;
  return applyTo.anyOf.some(rule => matchRule(rule, model));
}

function matchRule(rule: MatchRule, model: ModelDefinition): boolean {
  if (rule.id) return globMatch(rule.id, model.id);
  if (rule.family) return model.family === rule.family;
  if (rule.source) return model.source === rule.source;
  return false;
}
```

### Use Case Examples

- **"RAG only on reasoning models"**: `{ "anyOf": [{ "id": "deepseek-v4-pro" }, { "family": "gpt-4o" }] }`
- **"RAG on everything"**: `"*"`
- **"RAG only on proxy models"**: `{ "anyOf": [{ "source": "proxy" }] }`
- **"RAG on all except flash"**: combine with `noneOf`: `{ "anyOf": ["*"], "noneOf": [{ "id": "*-flash" }] }`

---

## LangGraph Integration

### Overview

For users needing advanced orchestration (conditional branches, retry loops, multi-step agentic RAG), the pipeline engine can optionally use [LangGraph.js](https://github.com/langchain-ai/langgraphjs).

### Engine Selection

```jsonc
{
  "deepseek-copilot.pipeline": {
    "engine": "simple",    // Default: linear pipeline
    // OR
    "engine": "langgraph"  // Advanced: LangGraph state graph
  }
}
```

### Simple Engine (Default)

Linear stage execution. No external dependencies. Stages run in order with optional `shouldRun` skip logic.

### LangGraph Engine

Full state graph with conditional edges, cycles, and checkpoints.

```jsonc
{
  "deepseek-copilot.pipeline": {
    "engine": "langgraph",
    "graph": {
      "nodes": {
        "embed": { "type": "embedding", "endpoint": "http://localhost:11434/v1/embeddings", "model": "nomic-embed-text" },
        "retrieve": { "type": "retrieval", "provider": "local-sqlite", "topK": 20 },
        "rerank": { "type": "reranker", "endpoint": "https://api.jina.ai/v1/rerank", "topK": 5 },
        "grade": { "type": "grader", "threshold": 0.7 },
        "inject": { "type": "context-injection", "position": "system" },
        "rephrase": { "type": "rephrase", "model": "deepseek-v4-flash" }
      },
      "edges": [
        { "from": "__start__", "to": "embed" },
        { "from": "embed", "to": "retrieve" },
        { "from": "retrieve", "to": "rerank", "condition": "state.documents.length > 0" },
        { "from": "retrieve", "to": "inject", "condition": "state.documents.length === 0" },
        { "from": "rerank", "to": "grade" },
        { "from": "grade", "to": "inject", "condition": "state.relevantDocs.length >= 3" },
        { "from": "grade", "to": "rephrase", "condition": "state.relevantDocs.length < 3" },
        { "from": "rephrase", "to": "embed" },
        { "from": "inject", "to": "__end__" }
      ],
      "maxIterations": 3
    }
  }
}
```

### Trade-offs

| Aspect | Simple Pipeline | LangGraph |
|--------|----------------|-----------|
| Bundle size | Minimal (~0 KB added) | +2–5 MB (`@langchain/langgraph`, `@langchain/core`) |
| Complexity | Linear, easy to debug | Arbitrary DAGs, cycles, conditionals |
| Use cases | Straightforward RAG | Adaptive RAG, self-correcting, multi-hop |
| Config style | Array of stages | Graph nodes + edges |
| Dependencies | Zero | `@langchain/langgraph`, `@langchain/core` |
| Debugging | Console logs | LangSmith integration available |

### Recommendation

Start with the **simple engine** as default. Offer LangGraph as an opt-in for power users:

```bash
npm install @langchain/langgraph @langchain/core  # only if engine=langgraph
```

The extension can lazy-load LangGraph dependencies only when the engine is set to `"langgraph"`, keeping the default install lightweight.

---

## Proposed File Structure

```
src/
├── pipeline/
│   ├── index.ts              // PipelineRunner, buildPipeline()
│   ├── types.ts              // PipelineContext, PipelineStage, RetrievedDocument
│   ├── config.ts             // Read pipeline config, applyTo matching
│   ├── engines/
│   │   ├── simple.ts         // Linear pipeline engine
│   │   └── langgraph.ts      // LangGraph engine (lazy-loaded)
│   └── stages/
│       ├── embedding.ts      // Embedding API call
│       ├── retrieval.ts      // Vector store query
│       ├── reranker.ts       // Reranker API call
│       └── context-injection.ts  // Inject docs into messages
├── models/
│   ├── registry.ts           // Merge built-in + custom + proxy models
│   └── custom.ts             // Custom model config parsing/validation
├── stores/
│   ├── types.ts              // VectorStore interface
│   ├── sqlite.ts             // Local SQLite vector store
│   └── remote.ts             // Remote vector DB adapter (Qdrant, Pinecone, custom)
├── proxy/
│   ├── index.ts              // ProxyChatProvider
│   └── resolver.ts           // Resolve native Copilot models
├── provider/                 // (existing — minimal changes)
│   ├── index.ts              // DeepSeekChatProvider (add pipeline hook)
│   ├── request.ts            // prepareChatRequest (add pipeline call)
│   └── ...
├── client/                   // (existing — unchanged)
├── config.ts                 // (existing — add pipeline/custom model readers)
├── consts.ts                 // (existing — unchanged)
├── types.ts                  // (existing — extend ModelDefinition)
└── ...
```

---

## Configuration Reference

### Full Example

```jsonc
// .vscode/settings.json
{
  // === Existing Settings (unchanged) ===
  "deepseek-copilot.baseUrl": "https://api.deepseek.com",
  "deepseek-copilot.maxTokens": 0,
  "deepseek-copilot.modelIdOverrides": {
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro"
  },
  "deepseek-copilot.debugMode": "minimal",

  // === Phase 1: Custom Models ===
  "deepseek-copilot.customModels": [
    {
      "id": "local-llama",
      "name": "Local Llama 3.3 70B",
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "maxInputTokens": 128000,
      "maxOutputTokens": 8192,
      "capabilities": {
        "toolCalling": true,
        "imageInput": false,
        "thinking": false
      }
    }
  ],

  // === Phase 4: Proxy Models ===
  "deepseek-copilot.proxyModels": [
    {
      "name": "GPT-4o + RAG",
      "source": { "vendor": "copilot", "family": "gpt-4o" },
      "pipeline": true
    },
    {
      "name": "Claude Sonnet + RAG",
      "source": { "vendor": "copilot", "family": "claude-sonnet" },
      "pipeline": true
    }
  ],

  // === Phase 2 & 3: Pipeline ===
  "deepseek-copilot.pipeline": {
    "enabled": true,
    "engine": "simple",
    "applyTo": {
      "anyOf": [
        { "id": "deepseek-v4-pro" },
        { "source": "proxy" }
      ]
    },
    "stages": [
      {
        "type": "embedding",
        "endpoint": "http://localhost:11434/v1/embeddings",
        "model": "nomic-embed-text",
        "apiKey": ""
      },
      {
        "type": "retrieval",
        "provider": "local-sqlite",
        "dbPath": "~/.deepseek-copilot/vectors.db",
        "topK": 20
      },
      {
        "type": "reranker",
        "endpoint": "https://api.jina.ai/v1/rerank",
        "model": "jina-reranker-v2",
        "apiKey": "${secret:jina-api-key}",
        "topK": 5
      },
      {
        "type": "context-injection",
        "position": "system",
        "template": "Use the following retrieved context to inform your answer. Cite sources by number when applicable.\n\n{{documents}}"
      }
    ]
  }
}
```

---

## Implementation Roadmap

| Phase | Scope | Effort |
|-------|-------|--------|
| **1** | Custom model injection | Moderate |
| **2** | Pipeline framework + context injection stage | Moderate |
| **3** | Embedding, retrieval, reranker stages | Large |
| **4** | Native Copilot model proxying | Moderate |

**Order:** Phase 1 → Phase 2 → Phase 4 → Phase 3 (Phase 4 can start once the pipeline framework exists; RAG stages can be added incrementally after.)

---

## Design Principles

1. **Zero-config for existing users** — nothing changes unless `pipeline.enabled` is `true`.
2. **Model-agnostic pipeline** — works with DeepSeek, custom models, and native Copilot models.
3. **Lazy loading** — LangGraph dependencies only loaded if `engine: "langgraph"`.
4. **Secret references** — API keys can reference VS Code SecretStorage via `${secret:key-name}`.
5. **Composable stages** — each stage is independent; skip any stage by removing it from config.
6. **Streaming preserved** — the pipeline enriches messages pre-LLM; streaming is unaffected.
