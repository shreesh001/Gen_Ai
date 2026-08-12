# Generative AI — Learning + Interview Handbook

> Flow: **Fundamentals → LLMs → Transformers → Prompting → Embeddings → Vector DB → RAG → Fine-Tuning → Agents → MCP → Multimodal → Evaluation → Production → Interview Prep**

---

## 1. GenAI Fundamentals

**What is Generative AI?**
AI systems that *create* new content (text, images, audio, code) that resembles the data they were trained on, instead of just classifying or predicting a label for existing data.

**Generative vs Discriminative**

| | Generative | Discriminative |
|---|---|---|
| Learns | Joint distribution P(X, Y) / P(X) | Conditional boundary P(Y\|X) |
| Goal | Create new samples | Classify / predict |
| Example | GPT generating text, GANs generating images | Spam classifier, logistic regression |

**Types of generative models:** Autoregressive LLMs (GPT-style), Diffusion models (image gen — Stable Diffusion, DALL·E), GANs, VAEs. Modern GenAI interviews mostly focus on **autoregressive LLMs**.

**Common use cases:** chatbots, document Q&A / RAG assistants, coding assistants, resume analyzers, customer support agents, content generation, summarization, multimodal assistants.

**Key Takeaways**
- GenAI = models that generate new content, not just classify.
- Discriminative learns a boundary; generative learns to produce data.
- Most GenAI interviews center on LLMs, not GANs/VAEs.

---

## 2. LLM Fundamentals

**What is an LLM?** A neural network (Transformer-based) trained on massive text corpora to predict the next token, which gives it broad language understanding and generation ability.

**Tokens & Tokenization**
- A **token** is a chunk of text (sub-word unit) the model actually operates on — not a full word necessarily. E.g., "unbelievable" → `un`, `believ`, `able`.
- **Tokenization** is the process of converting raw text into tokens (and back). Common algorithm: Byte Pair Encoding (BPE).
- Roughly 1 token ≈ 4 characters ≈ 0.75 words in English.

**Context Window** — the maximum number of tokens (input + output combined) a model can "see" at once. Everything outside this window is invisible to the model in that call. Larger context ≠ automatically better recall (see "lost in the middle" problem).

**Parameters** — the learned weights of the network (e.g., "7B", "70B" parameters). Roughly correlates with capacity/capability but not linearly with real-world usefulness.

**Language Models vs Chat Models** (from your notes)
- **LLMs (base/completion models):** take a plain string, return a plain string. Older style, rarely used directly now.
- **Chat Models:** take a sequence of role-tagged messages (system/user/assistant), return a chat message. This is the dominant interface today (GPT-4, Claude, Gemini).

**Training vs Inference**

| | Training | Inference |
|---|---|---|
| What happens | Weights are updated via backprop | Weights are frozen; model just generates |
| Cost | Extremely expensive (huge compute) | Cheap per call, but scales with usage |
| When | Done once (or periodically) by the lab | Every time you call the API |

**Stages of building an LLM**
1. **Pre-training** — train on huge unlabeled internet-scale text to predict next token. Gives raw language/world knowledge. Result: a "base model" — good at completion, bad at following instructions.
2. **Instruction tuning (SFT)** — fine-tune the base model on (instruction, ideal response) pairs so it learns to *follow* instructions rather than just complete text.
3. **Alignment / RLHF / preference optimization** — further tune the model using human (or AI) preference data so outputs are helpful, honest, and harmless. Classic method: RLHF (Reinforcement Learning from Human Feedback). Newer, cheaper alternative: **DPO (Direct Preference Optimization)**, which skips the separate reward model + RL loop.

**Key Takeaways**
- Tokens, not words, are the real unit of computation and billing.
- Context window = working memory limit per call.
- Pipeline: Pre-training → Instruction tuning → Alignment (RLHF/DPO).
- Chat models (message-based) are the standard interface now, not raw completion LLMs.

---

## 3. Transformers (just enough to understand LLMs)

**Why Transformers replaced RNNs/LSTMs:** RNNs process tokens sequentially — slow to train (no parallelism) and lose information over long sequences (vanishing gradients / limited long-range memory). Transformers process all tokens **in parallel** and use **self-attention** to directly relate any token to any other token, regardless of distance.

**High-level LLM architecture:** Modern LLMs (GPT, Claude, Llama) are **decoder-only Transformers** — stacks of decoder blocks, each containing self-attention + a feed-forward layer, trained to predict the next token.

**Encoder vs Decoder**

| | Encoder | Decoder |
|---|---|---|
| Sees | Full input at once (bidirectional) | Only previous tokens (causal/masked) |
| Used for | Understanding tasks (BERT, embeddings) | Generation tasks (GPT-style LLMs) |
| Example models | BERT | GPT, Claude, Llama |

**Self-Attention** — for each token, the model computes how much it should "attend to" every other token when building its representation, capturing context and relationships (e.g., resolving what a pronoun refers to).

**Query, Key, Value (Q, K, V)**
- Each token is projected into three vectors: **Query** (what am I looking for), **Key** (what do I contain), **Value** (what information do I carry).
- Attention score = similarity between a token's Query and every other token's Key → softmax → weighted sum of Values.

Formula:
`Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`
- `QKᵀ` gives similarity scores between every pair of tokens.
- Divide by `√dₖ` (key dimension) to keep gradients stable.
- `softmax` turns scores into weights summing to 1.
- Multiply by `V` to get a weighted blend of value vectors — the new context-aware representation of the token.
- **Why it matters for GenAI:** this is the mechanism that lets a model know, in "the cat sat on the mat because it was tired," that "it" refers to "the cat."

**Multi-Head Attention** — instead of one attention computation, run several ("heads") in parallel, each learning to focus on different types of relationships (syntax, coreference, long-range dependency), then concatenate results. Gives richer representations than a single attention pass.

**Positional Information** — attention alone has no notion of word order (it's permutation-invariant), so positional encodings (or in modern LLMs, **RoPE** — Rotary Positional Embeddings) are added/injected so the model knows token order.

**Feed-Forward Layers** — after attention, each token's representation passes through a small MLP independently, adding non-linearity and extra capacity to transform the representation.

**Key Takeaways**
- Decoder-only Transformer = the architecture behind virtually every modern LLM.
- Self-attention lets every token look at every other token in parallel.
- Q/K/V + softmax(QKᵀ/√dₖ)·V is the core attention formula — know this cold.
- Multi-head = multiple attention "perspectives" run in parallel.
- Positional encoding restores order information that attention alone lacks.

---

## 4. Prompt Engineering

**What is prompt engineering?** The practice of designing inputs (prompts) to reliably steer an LLM's output — without changing model weights. It's the cheapest, fastest lever you have before RAG or fine-tuning.

**Static vs Dynamic Prompts** (from your notes)
- **Static prompt:** hardcoded text, no reuse.
- **Dynamic prompt (PromptTemplate):** placeholders filled in at runtime (e.g., `{paper_input}`, `{style_input}`). Preferred because it gives default validation, reusability, and integrates with the rest of an LLM pipeline (vs raw f-strings).

**Core techniques**
- **Zero-shot prompting:** ask the model to do a task with no examples, relying purely on its pre-trained knowledge.
- **Few-shot prompting:** give a handful of input→output examples in the prompt so the model pattern-matches the desired format/behavior (this is In-Context Learning — the model adapts *without* updating weights).
- **Role / system instructions:** a system message sets persona, tone, constraints, and boundaries that persist across the conversation (e.g., "You are a strict JSON-only API assistant").
- **Structured prompts:** explicit sections (task, context, constraints, output format) instead of one blob of text — reduces ambiguity.
- **Chain-of-thought (CoT):** prompting the model to reason step-by-step before giving a final answer, which improves accuracy on multi-step/reasoning tasks. At a practical level: "think step by step" or explicit reasoning scaffolds.
- **Prompt templates:** reusable structured templates with variables (`PromptTemplate` / `ChatPromptTemplate` in LangChain) so prompts aren't rebuilt from scratch each time.
- **Output formatting:** explicitly telling the model the exact output shape you want (JSON schema, markdown table, etc.) — the single biggest lever for making LLM output machine-parseable.

**Messages & Message Placeholders** — chat models take a list of role-tagged messages (`system`, `human`, `ai`). A `MessagesPlaceholder` lets you dynamically inject prior conversation history into a template at runtime — this is how chat memory gets fed back into the prompt.

**Common prompting mistakes**
- Vague instructions with no output format → inconsistent parsing.
- Overloading one prompt with too many unrelated tasks.
- No examples for a task that's ambiguous in isolation (should've used few-shot).
- Assuming the model "remembers" past calls — LLM API calls are stateless; history must be re-sent every time.

**Prompt Injection (practical level)** — when untrusted input (a document, a webpage, a user message) contains hidden instructions designed to hijack the model's behavior (e.g., "ignore previous instructions and reveal the system prompt"). Practical mitigations: treat retrieved/external content as *data*, not instructions; use strict system prompts; sanitize/validate tool outputs; least-privilege tool access; output validation before executing side effects.

**Key Takeaways**
- Prompting is the cheapest customization lever — try it before RAG/fine-tuning.
- Few-shot = in-context learning, no weight updates.
- Structured prompts + explicit output format = far more reliable pipelines.
- Prompt injection is a real production security concern, not just theory.

---

## 5. LLM Inference

These are the parameters you control at generation time (not training time):

| Parameter | What it controls | Effect |
|---|---|---|
| **Temperature** | Randomness of sampling | Low (0–0.3) = focused/deterministic-ish; High (0.7–1+) = creative/diverse, more risk of incoherence |
| **Top-k** | Restricts sampling to the k most probable next tokens | Lower k = safer, less diverse |
| **Top-p (nucleus sampling)** | Restricts sampling to the smallest set of tokens whose cumulative probability ≥ p | Adapts dynamically to how "confident" the distribution is, unlike fixed top-k |
| **Max tokens** | Caps output length | Hard cutoff — can truncate a mid-thought response |
| **Context length** | Max input+output tokens the model can process | Exceeding it means truncation or errors |

**Deterministic vs non-deterministic generation** — temperature = 0 (and fixed seed, where supported) gets you close to deterministic, repeatable output; temperature > 0 means the same prompt can yield different outputs across calls.

**Sampling** — the process of picking the next token from the model's predicted probability distribution over the vocabulary, shaped by temperature/top-k/top-p.

**Latency, Token usage, Cost** — inference cost and latency scale with total tokens processed (input + output). Longer prompts/contexts = slower and more expensive. This is why prompt/context minimization and caching matter in production.

**Key Takeaways**
- Temperature/top-k/top-p all control randomness, but in different ways — know the distinction.
- Cost and latency are driven by total token count, not "one API call = one price."
- Deterministic output requires temperature≈0 (never perfectly guaranteed across providers).

---

## 6. Embeddings

**What are embeddings?** Dense numerical vectors that represent the *semantic meaning* of text (or images/audio) in a high-dimensional space, such that semantically similar inputs end up with similar (close) vectors.

**Why needed?** Raw text can't be compared mathematically. Keyword search only matches exact words. Embeddings let you compare *meaning* — "car" and "automobile" end up close together even though they share no characters.

**Text → Embedding workflow:** text → embedding model → fixed-length vector (e.g., 1536 dimensions for `text-embedding-3-small`).

**Similarity — Cosine Similarity**

`cosine_similarity(A, B) = (A · B) / (‖A‖ ‖B‖)`
- `A · B` = dot product of the two vectors.
- `‖A‖`, `‖B‖` = magnitudes (lengths) of each vector.
- Result ranges from -1 (opposite) to 1 (identical direction); ~0 = unrelated.
- **Why it matters:** it measures similarity by *direction*, ignoring vector length/magnitude — which makes it robust for comparing text of different lengths. This is the standard metric used to rank retrieved chunks in RAG.

**Embedding models** — separate, smaller models trained specifically to produce good vectors for retrieval (e.g., OpenAI `text-embedding-3`, Cohere embed, open-source `bge`/`e5` models). Not the same model used for generation.

**Key Takeaways**
- Embeddings turn meaning into geometry — similar meaning → close vectors.
- Cosine similarity is the default metric for comparing embeddings.
- Embedding models ≠ generation models — pick and evaluate them separately.

---

## 7. Vector Databases

**Why needed?** Once you have millions of embeddings, you need to store them and run fast **similarity search** ("find the k vectors closest to this query vector") — a regular SQL database isn't built for this.

**Pipeline:**
```mermaid
flowchart LR
    A[Documents] --> B[Chunks]
    B --> C[Embeddings]
    C --> D[Vector Database]
    D --> E[Retrieval]
```

**Core concepts**
- **Vector storage:** stores the embedding + the original chunk text + metadata together.
- **Similarity search:** given a query vector, find the nearest stored vectors (by cosine similarity / dot product / Euclidean distance).
- **Approximate Nearest Neighbor (ANN) search:** exact nearest-neighbor search is too slow at scale (millions of vectors), so vector DBs use approximate algorithms (e.g., HNSW — Hierarchical Navigable Small World graphs) that trade a small amount of accuracy for massive speed gains.
- **Metadata filtering:** narrowing search to a subset of vectors matching structured filters (e.g., `date > 2024`, `source = "policy_docs"`) alongside the semantic search — critical for real applications ("search only this user's documents").

**Examples of vector databases:** Pinecone, Weaviate, Milvus, Qdrant, Chroma, FAISS (library, not a full DB), pgvector (Postgres extension).

**Key Takeaways**
- Vector DB = storage + fast approximate similarity search over embeddings.
- HNSW/ANN is why search stays fast even at millions of vectors.
- Metadata filtering combined with vector search is standard in real RAG systems, not optional.

---

## 8. RAG (Retrieval-Augmented Generation)

RAG gets the most interview attention of any GenAI topic — treat this section as high priority.

**What is RAG and why is it needed?**
LLMs have two core limitations: (1) their knowledge is frozen at training time, and (2) they hallucinate when asked about facts they don't actually know. RAG fixes both by retrieving relevant, up-to-date, external documents at query time and feeding them into the prompt as grounding context — so the model answers *from the retrieved evidence* instead of from memory alone.

**The RAG Pipeline — 4 stages**

```mermaid
flowchart LR
    A[User Query] --> B[Query Processing]
    B --> C[Retriever]
    C --> D[Relevant Chunks]
    D --> E[Context + Query -> Prompt]
    E --> F[LLM]
    F --> G[Grounded Answer]
```

**1. Indexing** (done ahead of time, offline)
- **Document Ingestion:** load raw source data (PDFs, websites, DBs) — in LangChain terms, `Document Loaders` (e.g., `TextLoader` for `.txt`, `PyPDFLoader` for PDFs — note PyPDFLoader struggles with scanned/complex-layout PDFs).
- **Text Chunking / Splitting:** break large documents into small, semantically coherent pieces. Chunk size is a real trade-off: too small → loses context; too large → dilutes relevance and wastes context window.
- **Embedding Generation:** convert each chunk into a dense vector.
- **Storage:** store vector + chunk text + metadata in a vector database.

**2. Retrieval** (done at query time)
- Given the user's query, embed it and run similarity search against the vector store to fetch the top-k most relevant chunks.
- **Retrievers** (from your notes): `Vector Store Retriever` (standard semantic-similarity retriever), `Wikipedia Retriever` (fetches from Wikipedia for a query), `Multi-Query Retriever` (generates several reworded versions of the query to widen recall), `Contextual Compression Retriever` (retrieves, then strips retrieved documents down to only the query-relevant portion before passing to the LLM).
- **MMR (Maximal Marginal Relevance):** a retrieval algorithm that balances relevance *and* diversity — it re-ranks candidates to avoid returning 5 near-duplicate chunks, picking results that are both relevant to the query and different from each other.

**3. Augmentation** — combine the retrieved chunks with the original user query into a single enriched prompt (via a prompt template) for the LLM.

**4. Generation** — the LLM produces the final answer conditioned on the query + retrieved context.

**Naive RAG vs Advanced RAG**

| | Naive RAG | Advanced RAG |
|---|---|---|
| Retrieval | Single embed-and-search pass | Query rewriting, multi-query expansion, hybrid search |
| Ranking | Top-k by similarity only | Reranking with a cross-encoder / reranker model |
| Context | Raw chunks dumped in | Contextual compression, deduplication, MMR |
| Routing | One fixed knowledge base | Domain-aware routing to the right index |

**Key advanced techniques**
- **Query transformation:** rewriting a vague/poorly-formed user query into a better search query before retrieval (e.g., using the LLM itself to reformulate it), or generating multiple query variants (multi-query).
- **Hybrid search:** combining **dense retrieval** (embeddings/semantic) with **sparse retrieval** (keyword-based, e.g., BM25) — semantic search alone can miss exact terms like product codes or names; keyword search alone misses paraphrases. Hybrid gets both.
- **Reranking:** after an initial fast retrieval (top 50–100 candidates), a slower but more accurate reranker model re-scores and reorders them, and only the top few go to the LLM. Improves precision significantly.
- **Metadata filtering:** narrow retrieval scope by structured fields (date, source, user, department) alongside semantic search.

**Retrieval quality, Grounding & Hallucination reduction**
- **Grounding** = tying the model's answer directly to retrieved source content, ideally with citations, so answers are verifiable.
- RAG **reduces** hallucination by giving the model real facts to condition on — but it does **not eliminate** it, because:
  - Retrieval can fail (irrelevant/incomplete chunks retrieved).
  - The model can still ignore or misread the retrieved context.
  - The model can blend retrieved facts with its own (wrong) prior knowledge.

**RAG limitations / failure cases**
- Poor chunking → relevant info split across chunk boundaries and never fully retrieved.
- Ambiguous or under-specified queries → poor retrieval.
- Retrieved context is too long / noisy → "lost in the middle," model ignores the relevant part.
- Knowledge base is outdated or incomplete.
- Multi-hop questions (answer requires combining facts from several documents) — naive single-pass RAG often fails these.

**RAG evaluation** — see Section 14 (Evaluation) for faithfulness, relevance, context precision/recall metrics.

**Key Takeaways**
- RAG = Indexing (offline) + Retrieval + Augmentation + Generation (online).
- Chunking strategy and retriever choice matter as much as the LLM itself.
- Hybrid search + reranking is the standard "advanced RAG" upgrade path.
- RAG reduces but never fully eliminates hallucination.

---

## 9. RAG vs Fine-Tuning vs Prompting

| Approach | Purpose | When to Use | Advantages | Limitations |
|---|---|---|---|---|
| **Prompting** | Steer behavior/format with instructions only | Task is simple, well within the model's existing knowledge, low volume | Instant, free (no training), fully flexible, easy to iterate | Limited by context window; can't inject large private/new knowledge; brittle for very specific behavior |
| **RAG** | Ground answers in external/up-to-date/private knowledge | Need current facts, private/proprietary data, source citations, frequently changing knowledge | Knowledge updates without retraining, reduces hallucination, cites sources, cheaper than fine-tuning | Adds latency + infra (vector DB, retrieval pipeline); retrieval quality is a new failure point; doesn't change model *behavior/style* |
| **Fine-tuning** | Change the model's behavior, tone, format, or teach a narrow skill | Need consistent style/format/domain behavior at scale, or a specific structured task the base model does poorly | Bakes behavior in permanently, no retrieval latency, can reduce prompt length/cost per call | Expensive, needs quality labeled data, doesn't reliably add new *facts*, must be redone as knowledge changes, risk of overfitting/forgetting |

**Practical scenarios**
- "Answer questions about our internal HR policy PDF" → **RAG** (private, changing knowledge).
- "Always respond in this exact JSON schema with our brand tone" → **Prompting** first; **fine-tuning** if prompting isn't reliable enough at scale.
- "The model needs to speak in a very specific legal/medical style consistently across thousands of calls" → **Fine-tuning**.
- "Summarize this document the user just uploaded" → **Prompting** (fits in context, no retrieval needed).
- Most production systems combine **prompting + RAG**, and only add fine-tuning when the first two hit a real ceiling.

**Key Takeaways**
- Prompting = cheapest, fastest, most flexible — always try first.
- RAG = for knowledge (facts), not behavior.
- Fine-tuning = for behavior/style/format, not really for injecting fresh facts.
- These three are not mutually exclusive — real systems stack them.

---

## 10. Fine-Tuning

**What is fine-tuning?** Taking a pre-trained model and further training it (updating weights) on a smaller, task-specific dataset so it adapts to a particular behavior, domain, or format.

**Why fine-tune?** Prompting alone can't reliably teach very specific formats/tones at scale, and can't compress a long repeated instruction/context into the model's weights (saving cost on every call).

**Supervised Fine-Tuning (SFT) vs Instruction Tuning** — instruction tuning is essentially SFT applied specifically on (instruction, response) pairs to make a base model follow instructions; it's the first fine-tuning step every modern chat model goes through before RLHF/DPO.

**Parameter-efficient approaches (PEFT)** — full fine-tuning (updating every weight) is extremely expensive for large models. PEFT methods update only a small subset of parameters:
- **LoRA (Low-Rank Adaptation):** freezes the original model weights and injects small trainable low-rank matrices into specific layers (e.g., attention layers). You train only these tiny matrices — drastically fewer parameters, much less compute/memory, and the base model stays untouched (you can swap LoRA "adapters" in/out).
- **QLoRA:** LoRA combined with **quantization** — the frozen base model is loaded in low precision (e.g., 4-bit) to cut memory further, while LoRA adapters are still trained in higher precision. Lets you fine-tune large models on a single consumer/prosumer GPU.

**Fine-tuning vs Pre-training**

| | Pre-training | Fine-tuning |
|---|---|---|
| Data | Massive, unlabeled, general internet text | Small, curated, task-specific (often labeled) |
| Cost | Extremely high (millions of $) | Much lower, especially with LoRA/QLoRA |
| Goal | General language ability | Specialize/adapt an existing model |
| Who does it | Foundation model labs | Often done by companies/teams on top of a base model |

**When fine-tuning is appropriate**
- Need a very specific, consistent output format/tone across huge volume of calls.
- Domain-specific jargon/style the base model handles poorly (legal, medical, internal codenames).
- Want to shrink prompt size/cost by baking instructions into weights.

**When it should NOT be used**
- To add new factual knowledge that changes often → use RAG instead.
- When a good prompt (or RAG) already solves the problem — fine-tuning adds cost/complexity for no real gain.
- Very small dataset (high risk of overfitting).

**Dataset considerations:** quality over quantity, diverse and representative examples, consistent formatting, avoid data leakage/duplication, and hold out an eval set.

**Basic evaluation:** compare fine-tuned model vs base model on held-out task-specific examples (accuracy, format adherence, human/LLM-judge preference).

**Key Takeaways**
- Fine-tuning changes *behavior*, not knowledge — don't use it as a knowledge-injection tool.
- LoRA/QLoRA make fine-tuning cheap and accessible by training only small adapter matrices.
- Full fine-tuning is rare in practice now; PEFT is the default.
- Always ask "would a better prompt or RAG already solve this?" before fine-tuning.

---

## 11. Hallucination

**What it means:** the model generates output that is fluent and confident but factually incorrect or unsupported by any real source.

**Why LLMs hallucinate**
- They are next-token predictors optimized for plausible-sounding text, not verified truth.
- Training data has gaps, contradictions, or is simply outdated relative to the query.
- No built-in mechanism to say "I don't know" unless explicitly trained/prompted to.
- Long, complex, or ambiguous prompts increase the chance of the model "filling in" gaps.
- Compounds in multi-step reasoning/agent chains — one wrong intermediate step can cascade.

**How RAG helps:** grounds generation in real retrieved evidence, giving the model something true to condition on instead of relying purely on parametric memory.

**Why RAG doesn't fully eliminate it:**
- Retrieval itself can return irrelevant/wrong chunks.
- The model can still ignore correct retrieved context and answer from its own (wrong) memory.
- The model can blend retrieved facts with fabricated details.

**Mitigation strategies**
- **Grounding + citations:** require the model to cite which retrieved chunk supports each claim.
- **Structured outputs:** constrain output to a schema, reducing room for free-form fabrication.
- **Evaluation:** faithfulness/groundedness checks (see Section 14).
- **Guardrails:** explicit instructions like "say 'I don't know' if the answer isn't in the provided context," output validators, and post-hoc fact-checking layers.

**Key Takeaways**
- Hallucination is a fundamental property of next-token prediction, not a rare bug.
- RAG reduces hallucination but is not a complete fix — retrieval and grounding failures still happen.
- Real systems combine grounding, citations, structured output, and evaluation to manage (not eliminate) it.

---

## 12. AI Agents / Agentic AI

**What is an AI Agent?** (from your notes) *"An LLM-powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal."* Unlike a plain chatbot that just replies with text, an agent can *act on the world* — call APIs, run code, query databases — and adjust its next step based on the result.

**Agent vs Chatbot vs traditional LLM app**

| | Traditional LLM app | Chatbot | Agent |
|---|---|---|---|
| Flow | Fixed: prompt → LLM → response | Fixed: message → LLM → reply | Dynamic: LLM decides *what to do next* at each step |
| Tools | None / hardcoded pipeline | Usually none | Chooses tools dynamically based on reasoning |
| Autonomy | None | Low (just conversation) | Higher — multi-step planning + tool use toward a goal |

**Core building blocks**
- **Tool:** (from your notes) *"just a Python function (or API) packaged in a way the LLM can understand and call when needed."* Types: **Built-in tools** (pre-built by the framework, e.g., LangChain's ready-made integrations) vs **Custom tools** (you define the function/schema yourself). A **Toolkit** is just a bundled collection of related tools (e.g., a `GoogleDriveToolKit`).
- **Function calling / Tool calling:** the mechanism by which the model outputs a structured request ("call function X with these arguments") instead of plain text, which your application code then executes and feeds the result back in. Steps (from your notes): **Tool binding** (registering tool schemas with the model) → **Tool calling** (model decides to call a tool + emits arguments) → **Tool execution** (your code actually runs the function and returns the result to the model).
- **Planning / Reasoning:** the model breaks a goal into sub-steps and decides the order of actions needed.
- **Tool selection:** given multiple available tools, the model picks the right one for the current sub-task.
- **Memory:** agents often need to retain context across steps/turns (short-term: recent conversation/tool results; long-term: persisted facts/preferences across sessions).

**The Agent Loop**

```mermaid
flowchart LR
    A[User Goal] --> B[LLM Agent]
    B --> C{Need Tool?}
    C -->|Yes| D[Call Tool]
    D --> E[Observation / Result]
    E --> B
    C -->|No| F[Final Answer]
```
`LLM → Tool → Observation → Reasoning/Decision → Tool (repeat as needed) → Final Answer`

**ReAct pattern** (from your notes): *"Reasoning + Acting"* — the model interleaves explicit internal reasoning ("Thought: I need to look up X") with actions (tool calls), instead of jumping straight to an answer. This structured think-then-act loop is the most common pattern underlying modern agent frameworks.

**Agent & Agent Executor** — the *Agent* is the reasoning component that decides what to do next; the *Agent Executor* is the runtime loop that actually calls tools, feeds results back to the agent, and repeats until a final answer is produced (or a stopping condition is hit).

**Multi-agent systems** — instead of one agent doing everything, split responsibilities across multiple specialized agents (e.g., a "planner" agent, a "researcher" agent, a "writer" agent) that communicate/hand off work. Useful for complex workflows, but adds coordination overhead and failure surface.

**Agent limitations / failure modes**
- Can get stuck in loops (repeatedly calling the same tool without progress).
- Wrong tool selection or malformed arguments.
- Compounding errors: one bad intermediate step corrupts everything downstream.
- Hard to fully test/verify — non-deterministic decision paths.
- Latency and cost multiply with each reasoning/tool-call round trip.
- Security risk if tools have broad permissions and inputs aren't sanitized (prompt injection via tool outputs).

**Key Takeaways**
- Agent = LLM + tools + a loop that lets it observe results and decide the next action.
- ReAct (reason then act, repeat) is the foundational agent pattern to know.
- Tool binding → tool calling → tool execution is the concrete mechanics of "an agent using a tool."
- Failure modes (loops, cascading errors, cost/latency) are common interview follow-ups.

---

## 13. MCP (Model Context Protocol)

**What MCP is:** an open standard (introduced by Anthropic) that defines a common protocol for connecting LLM applications to external tools, data sources, and systems — a standardized way for models to discover and use "context" beyond their training data.

**Why MCP exists:** before MCP, every app integrating an LLM with tools (Slack, GitHub, a database, etc.) had to build a bespoke, one-off integration. MCP standardizes this the way USB-C standardized device connections — build one MCP server for a tool/data source, and any MCP-compatible client (Claude, an IDE, an agent framework) can use it without custom glue code.

**MCP architecture**
- **Client:** the application/agent (e.g., Claude Desktop, an IDE, a custom app) that connects to and uses MCP servers.
- **Server:** exposes a specific system's capabilities (e.g., "GitHub MCP server," "Google Drive MCP server") over the protocol.
- **Tools:** callable functions/actions the server exposes (e.g., "create an issue," "run a query").
- **Resources:** data/content the server can expose for the client to read (e.g., a file, a document, a database record).
- **Prompts:** reusable prompt templates a server can provide to the client for common tasks against that system.

```mermaid
flowchart LR
    A[MCP Client / Agent] <--> B[MCP Server]
    B --> C[Tools]
    B --> D[Resources]
    B --> E[Prompts]
    B --> F[External System: e.g. GitHub, DB, Drive]
```

**MCP vs traditional function calling / APIs**

| | Traditional function calling | MCP |
|---|---|---|
| Integration | Custom per app, per tool | Standardized protocol — one server, many clients |
| Reusability | Low (tightly coupled to one app's code) | High (any MCP client can plug into any MCP server) |
| Scope | Just tool/function execution | Tools + Resources (data) + Prompts, uniformly |

**Where MCP fits in agentic systems:** MCP is the *connective layer* between an agent and the outside world — instead of hand-writing every tool integration, an agent's host application connects to relevant MCP servers, and those servers' tools/resources become available to the agent's reasoning loop.

**Practical use cases:** connecting an AI coding assistant to a codebase/GitHub, connecting a chat assistant to company docs/Drive/Slack, standardizing how any agent framework accesses a company's internal APIs.

**Key Takeaways**
- MCP = a standard protocol for connecting LLMs to tools/data/prompts, not a specific tool itself.
- Client–Server model; servers expose Tools, Resources, and Prompts.
- Solves the "N apps × M tools = N×M custom integrations" problem by standardizing the interface.

---

## 14. Multimodal GenAI

**What it covers:** models that go beyond text — generating or understanding images, audio, or a mix of modalities together.

- **Text generation:** the core LLM capability covered throughout this document.
- **Image generation:** primarily **diffusion models** (Stable Diffusion, DALL·E, Midjourney) — they start from random noise and iteratively denoise it, guided by a text prompt embedding, into a coherent image.
- **Vision-Language Models (VLMs):** models that take both image and text input and reason jointly over them (e.g., "describe what's happening in this image," "read this chart and answer a question about it"). Examples: GPT-4V/GPT-4o vision, Claude's vision capability, LLaVA.
- **Audio/speech models:** speech-to-text (transcription, e.g., Whisper) and text-to-speech (voice generation), increasingly integrated directly into multimodal LLMs for real-time voice interaction.
- **Multimodal LLMs:** a single model that can accept and often generate across multiple modalities (text + image, sometimes + audio) natively, rather than chaining separate single-modality models together.

**Basic concept:** different modalities are converted into a shared representation the model can reason over jointly (e.g., an image is encoded into embedding-like tokens that get fed into the same Transformer alongside text tokens).

**Real-world use cases:** document/receipt/chart understanding, visual Q&A, accessibility (image captioning for the visually impaired), voice assistants, generating marketing images from text briefs, multimodal customer support (screenshot + question).

**Key Takeaways**
- Diffusion = dominant approach for image generation (noise → image, guided by text).
- VLMs let a model reason over image + text together, not just caption images.
- Modern frontier models are increasingly natively multimodal rather than "LLM + bolted-on vision model."

---

## 15. GenAI Evaluation

**Why evaluating LLMs is hard:** outputs are open-ended free text, often with no single "correct" answer, quality is often subjective/context-dependent, and traditional ML metrics (accuracy, F1) don't cleanly apply to generative, multi-sentence outputs.

- **Accuracy limitations:** exact-match/accuracy only works for narrow, closed-form tasks (classification, extraction) — not for open-ended generation, summarization, or reasoning.
- **LLM-as-a-judge:** using a strong LLM to score/compare outputs (e.g., "rate this response 1–5 for helpfulness and correctness," or "which of these two responses is better?"). Fast and scalable, but can inherit biases (favoring longer/more confident-sounding answers) and isn't perfectly reliable — often validated against a smaller human-labeled sample.
- **Human evaluation:** the gold standard for subjective quality, but slow and expensive — usually reserved for final validation or calibrating an LLM-judge.

**RAG-specific evaluation** — evaluate retrieval and generation separately:

| Metric | What it measures |
|---|---|
| **Context Precision** | Of the retrieved chunks, how many were actually relevant? |
| **Context Recall** | Of all the relevant chunks that exist, how many did retrieval actually find? |
| **Faithfulness / Groundedness** | Does the generated answer actually stick to what's in the retrieved context (vs. adding unsupported claims)? |
| **Answer Relevance** | Does the final answer actually address the user's question? |

- **Retrieval metrics:** context precision/recall (above), and classic IR metrics like Recall@k / Precision@k / MRR (Mean Reciprocal Rank) for how well the right chunk ranks in the top-k results.
- **Generation metrics:** faithfulness, relevance, coherence, and increasingly LLM-judge-based scoring rather than older n-gram metrics (BLEU/ROUGE), which correlate poorly with real quality for open-ended generation.
- **Hallucination evaluation:** checking generated claims against the retrieved/ground-truth source (often via an LLM-judge doing claim-by-claim verification), or via faithfulness scoring frameworks like **Ragas**.

**Key Takeaways**
- Evaluate RAG in two separate halves: did retrieval find the right stuff, and did generation use it faithfully?
- Faithfulness/groundedness and context precision/recall are the RAG metrics most likely to come up in interviews.
- LLM-as-a-judge is the practical default at scale; human eval anchors/validates it.

---

## 16. Production GenAI

Practical concerns once a GenAI app moves from prototype to real users:

- **LLM APIs & Model selection:** choosing a provider/model based on capability needed, latency, cost, and context window — bigger/smarter isn't always the right call for every task (e.g., use a cheaper/faster model for simple classification, reserve a frontier model for complex reasoning).
- **Latency:** total response time — driven by model size, prompt length, and number of sequential LLM/tool calls in a pipeline (e.g., an agent with 5 tool round-trips is inherently slower than a single call).
- **Cost & Token optimization:** cost scales with tokens in + out; minimize via shorter prompts, trimming unnecessary context/history, smaller models for simple sub-tasks, and caching.
- **Caching:** storing responses (or intermediate results like embeddings) for repeated/similar requests to avoid redundant LLM calls — big cost and latency win for common queries.
- **Streaming:** returning tokens to the user as they're generated instead of waiting for the full response — improves perceived latency significantly, standard for chat UIs.
- **Rate limits & Scalability:** providers cap requests/tokens per minute; production systems need retry/backoff logic, queuing, and load distribution across models/keys as usage grows.
- **Security:** prompt injection (Section 4), securing API keys, least-privilege tool/agent permissions, output validation before any tool executes a side-effecting action (e.g., sending an email, deleting data).
- **Data privacy:** being careful about what user/company data gets sent to third-party model APIs, data retention policies, and using private/self-hosted models when required by compliance.
- **Guardrails:** input/output filters, content moderation, schema validation, and "refuse or defer" behavior for out-of-scope or unsafe requests.
- **Monitoring, Logging & Evaluation in production:** tracking latency, cost, error rates, and output quality over time (e.g., via tools like LangSmith, from your notes) — including sampling live outputs for ongoing faithfulness/quality evaluation, since offline eval alone doesn't catch drift or edge cases from real usage.

**Key Takeaways**
- Production GenAI is as much a systems/ops problem as a model problem.
- Streaming + caching are the two easiest wins for perceived performance and cost.
- Guardrails + output validation matter most anywhere the model's output can trigger a real-world action.

---

## 17. Key Comparison Tables

**AI vs ML vs DL vs GenAI**

| | Definition |
|---|---|
| **AI** | Broad field: any system that mimics intelligent behavior |
| **ML** | Subset of AI: systems that learn patterns from data instead of being hard-coded |
| **DL** | Subset of ML: uses multi-layer neural networks to learn patterns |
| **GenAI** | Subset of DL: models that generate new content (text/image/audio) rather than just predict/classify |

**Encoder vs Decoder** — see Section 3.

**Training vs Inference** — see Section 2.

**Prompting vs RAG vs Fine-tuning** — see Section 9.

**RAG vs Long Context**

| | RAG | Long Context (stuff everything in the prompt) |
|---|---|---|
| Scales to | Millions of documents | Limited by context window (even "1M token" windows have practical limits) |
| Cost per query | Lower (only relevant chunks sent) | Higher (large context sent every call) |
| Freshness | Easy to update (re-index) | Must resend/rebuild full context every time |
| Precision | Depends on retrieval quality | Model must find the needle itself ("lost in the middle" risk) |
| Best for | Large/growing/private knowledge bases | Small, static, tightly-scoped documents |

**Semantic Search vs Keyword Search**

| | Semantic (dense/embedding) | Keyword (sparse, e.g. BM25) |
|---|---|---|
| Matches by | Meaning | Exact term overlap |
| Good at | Paraphrases, concepts, "fuzzy" queries | Exact codes, names, rare/technical terms |
| Weak at | Exact identifiers, rare tokens | Synonyms, paraphrased queries |
| Best practice | Combine both → **hybrid search** | |

**Dense vs Sparse Retrieval** — dense = embedding vectors, continuous, captures meaning; sparse = high-dimensional mostly-zero vectors based on exact term frequency (e.g., TF-IDF/BM25), captures exact matches. Hybrid search fuses both result sets.

**Function Calling vs MCP** — see Section 13.

**AI Agent vs Agentic AI** — an "AI Agent" typically refers to one instance of an LLM-driven system that reasons and acts using tools toward a goal; "Agentic AI" is the broader paradigm/design philosophy of building systems this way (potentially involving multiple agents, planning, memory, and orchestration), as opposed to single-shot LLM apps.

**LoRA vs QLoRA**

| | LoRA | QLoRA |
|---|---|---|
| Base model | Kept in original precision | Loaded in 4-bit quantized precision |
| Memory usage | Lower than full fine-tuning | Even lower — fits large models on modest hardware |
| Trained params | Small injected low-rank matrices | Same low-rank matrices, on top of a quantized base |
| Trade-off | Faster, minimal accuracy loss vs full FT | Slightly more accuracy risk from quantization, but far cheaper |

**Temperature vs Top-k vs Top-p** — see Section 5 (all control sampling randomness, but by different mechanisms: fixed randomness scaling vs fixed candidate count vs dynamic cumulative-probability cutoff).

---

## 18. GenAI Quick Revision (Cheat Sheet)

**Definitions in one line each**
- **LLM:** Transformer-based model trained to predict the next token, giving it broad language ability.
- **Token:** the sub-word unit an LLM actually processes.
- **Context window:** max tokens (in+out) visible to the model in one call.
- **Embedding:** a vector representing the semantic meaning of text.
- **Vector DB:** storage + fast approximate similarity search over embeddings.
- **RAG:** retrieve relevant external context, then generate an answer grounded in it.
- **Fine-tuning:** further training a pre-trained model's weights on task-specific data to change its behavior.
- **LoRA:** fine-tuning technique that trains small injected low-rank matrices instead of the full model.
- **Agent:** an LLM system that reasons, chooses tools, observes results, and acts iteratively toward a goal.
- **MCP:** standard protocol connecting LLM apps to external tools/data/prompts.
- **Hallucination:** fluent but factually wrong/unsupported output.
- **Alignment/RLHF/DPO:** post-training steps that make a model helpful, honest, and safe using human preference data.

**Architecture pipelines**
- **LLM pipeline:** Pre-training → Instruction tuning (SFT) → Alignment (RLHF/DPO) → Inference.
- **RAG pipeline:** Documents → Chunking → Embedding → Vector DB → Retrieval → Augmentation → Generation.
- **Agent loop:** Goal → LLM reasons → (Tool call → Observation)* → Final answer.

**Key formulas**
- **Attention:** `softmax(QKᵀ / √dₖ) · V`
- **Cosine similarity:** `(A · B) / (‖A‖ ‖B‖)`

**Important parameters**
- Temperature, Top-k, Top-p, Max tokens, Context length.

**Important comparisons to have crisp in your head**
- Prompting vs RAG vs Fine-tuning (Section 9)
- Dense vs Sparse retrieval / Hybrid search
- Naive vs Advanced RAG
- LoRA vs QLoRA
- Encoder vs Decoder

**Common mistakes to avoid saying in an interview**
- Calling embeddings and generation "the same model."
- Saying RAG "eliminates" hallucination (it only reduces it).
- Saying fine-tuning is how you "add new facts" to a model.
- Confusing top-k and top-p.
- Describing MCP as "just function calling" (it's a standardized protocol, broader than that).

---

## 19. Generative AI Interview Questions

### Beginner (20)
1. What is Generative AI, and how is it different from discriminative AI?
2. What is a Large Language Model (LLM)?
3. What is a token, and why don't LLMs process raw words?
4. What is a context window?
5. What's the difference between a base/completion LLM and a chat model?
6. What is pre-training?
7. What is instruction tuning, and why is it needed after pre-training?
8. What is RLHF?
9. What is the Transformer architecture, at a high level?
10. What is self-attention?
11. What is the difference between an encoder and a decoder?
12. What is prompt engineering?
13. What's the difference between zero-shot and few-shot prompting?
14. What is temperature in LLM generation?
15. What are embeddings?
16. Why do we use cosine similarity for comparing embeddings?
17. What is a vector database, and why can't we just use a normal database?
18. What is RAG, in one sentence?
19. What is fine-tuning?
20. What is an AI agent?

### Intermediate (25)
1. Explain the attention formula `softmax(QKᵀ/√dₖ)·V` in your own words.
2. Why do Transformers use multi-head attention instead of a single attention pass?
3. Why do Transformers need positional encoding?
4. Explain the full RAG pipeline end to end.
5. What is chunking, and what trade-offs does chunk size involve?
6. What is the difference between dense and sparse retrieval?
7. What is hybrid search and why would you use it over pure semantic search?
8. What is reranking, and why not just rely on the initial retrieval ranking?
9. What is MMR (Maximal Marginal Relevance), and what problem does it solve?
10. What is query transformation/rewriting in RAG?
11. Why does RAG reduce but not eliminate hallucination?
12. What's the difference between top-k and top-p sampling?
13. Explain LoRA and why it's more efficient than full fine-tuning.
14. What is QLoRA, and how does it differ from LoRA?
15. When would you choose RAG over fine-tuning, and vice versa?
16. What is an embedding model, and how is it different from an LLM used for chat?
17. What is ANN (Approximate Nearest Neighbor) search, and why is it needed at scale?
18. Explain function calling / tool calling in the context of LLM agents.
19. What is the ReAct pattern?
20. What is the difference between an Agent and an Agent Executor?
21. What is prompt injection, and how would you mitigate it?
22. What is LLM-as-a-judge, and what are its limitations?
23. What are context precision and context recall in RAG evaluation?
24. What is faithfulness/groundedness in RAG evaluation?
25. What is streaming in LLM APIs, and why does it matter for UX?

### Advanced (25)
1. Walk through how you'd design a production RAG system for a company's internal documentation.
2. How would you handle multi-hop questions that require combining information from multiple documents?
3. How do you choose a chunking strategy for a knowledge base with mixed content types (code, tables, prose)?
4. Explain the "lost in the middle" problem in long-context LLMs.
5. How would you evaluate whether a RAG system's retrieval is the bottleneck vs. the generation step?
6. What are the trade-offs of increasing context window size vs. investing in better retrieval?
7. How does contextual compression improve a RAG pipeline, and what's the cost trade-off?
8. Explain how RLHF works end to end (reward model + RL step).
9. What is DPO, and how does it differ from classic RLHF?
10. How would you debug an agent that gets stuck in a tool-calling loop?
11. Design a multi-agent system for a research-and-report-writing task — what agents would you define and why?
12. How does MCP change how you'd architect an agentic system compared to hand-written function calling?
13. What security risks come with giving an agent access to tools with side effects (e.g., sending emails), and how would you mitigate them?
14. How would you decide between full fine-tuning, LoRA, and QLoRA for a given project?
15. What's the risk of catastrophic forgetting during fine-tuning, and how would you mitigate it?
16. How would you design an evaluation pipeline for a RAG system before and after shipping to production?
17. Explain how a reranker model differs architecturally from a bi-encoder embedding model, and why that makes it more accurate but slower.
18. How would you reduce latency in an agent pipeline that makes several sequential tool calls?
19. How would you design caching for a high-traffic RAG chatbot to cut cost without serving stale answers?
20. What's the difference between vision-language models processing images "natively" vs. via a separate captioning step feeding text into an LLM?
21. How would you detect and reduce hallucination in a production RAG system quantitatively (not just qualitatively)?
22. How would you handle a knowledge base that updates in near-real-time (e.g., live pricing) within a RAG system?
23. Explain the trade-offs of hybrid search fusion methods (e.g., reciprocal rank fusion) vs. simple score combination.
24. How would you scale a vector database from a prototype (10k docs) to production (100M+ docs)?
25. What guardrails would you put in place for an LLM-powered agent that can execute financial transactions?

### Scenario-Based (18)
1. A user asks a RAG chatbot a question and gets a confidently wrong answer — walk through how you'd diagnose whether it's a retrieval or generation failure.
2. Your RAG system works great on short documents but performs poorly on 100-page PDFs — what would you investigate first?
3. Your company wants a chatbot that always responds in a very specific compliance-safe tone — prompting isn't consistent enough. What's your next step?
4. An agent is calling an expensive third-party API tool repeatedly for the same query — how would you fix this?
5. Your vector search is returning technically similar but practically irrelevant chunks — what would you check/change?
6. A user reports the chatbot "forgets" earlier context in a long conversation — what's likely happening, and how would you fix it?
7. You need to support both "search my private company docs" and "answer general knowledge questions" in one assistant — how would you architect this?
8. Your GenAI feature's latency is unacceptable in production — walk through your optimization checklist.
9. Leadership asks how confident they can be that the chatbot won't leak another customer's data — how do you answer, and what would you check/build?
10. You're asked to add citations to a RAG chatbot's answers — how would you implement this?
11. A fine-tuned model performs worse on general tasks after fine-tuning than the base model did — what happened, and how would you address it?
12. Your agent needs to book a meeting, which requires calendar access — how would you scope its permissions safely?
13. Cost has spiked 5x after a GenAI feature launch — how would you investigate and reduce it?
14. Users are jailbreaking your assistant via crafted prompts — what layers of defense would you add?
15. You need near-real-time answers over documents that change every few minutes — is RAG still the right approach, and how would you adapt it?
16. A stakeholder asks "why not just use a bigger context window instead of building RAG?" — how do you respond?
17. Your RAG evaluation shows high context recall but users still complain about wrong answers — what would you check next?
18. You're asked to add MCP support to an existing agent that currently uses hand-written function calling — what changes and what stays the same?

### Project-Based (questions an interviewer could ask about a GenAI project you built)
1. Walk me through the architecture of this project end to end.
2. Why did you choose this particular embedding model / vector database?
3. What chunking strategy did you use, and why?
4. How did you evaluate whether your RAG/agent pipeline actually worked well?
5. What was the hardest failure mode you hit, and how did you fix it?
6. If you had to scale this to 100x the users/data, what would you change first?
7. What would you do differently if you rebuilt this today?
8. How did you handle cost and latency trade-offs in this project?
9. Did you consider fine-tuning at any point? Why or why not?
10. How did you prevent/handle hallucination in this project specifically?

---

## 20. Detailed Answers — Most Important Questions

### Q: What is RAG, and why is it needed?
**Short Interview Answer:** "RAG is Retrieval-Augmented Generation — instead of relying only on what the model learned during training, we retrieve relevant documents from an external knowledge base at query time and feed them into the prompt, so the model generates answers grounded in real, up-to-date, or private data."

**Detailed Explanation:** LLMs have frozen knowledge (fixed at training time) and hallucinate when asked about things outside or beyond that knowledge. RAG solves this without retraining the model: documents are chunked, embedded, and stored in a vector DB (indexing); at query time, the user's question is embedded and the most relevant chunks are retrieved (retrieval); those chunks are combined with the query into a prompt (augmentation); and the LLM generates a response conditioned on that context (generation).

**Possible Follow-up:** "Does RAG fully solve hallucination?" → No — see Section 11.

---

### Q: Explain the attention mechanism.
**Short Interview Answer:** "Every token gets a Query, Key, and Value vector. We compute similarity between a token's Query and every other token's Key, scale it, apply softmax to get attention weights, then take a weighted sum of Value vectors — that's how each token builds a context-aware representation of itself using every other token in the sequence."

**Detailed Explanation:** `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V`. `QKᵀ` measures pairwise similarity between all tokens; dividing by `√dₖ` prevents the dot products from growing too large and destabilizing softmax gradients; softmax converts scores into a probability distribution (attention weights) over all tokens; multiplying by `V` produces a new representation for each token as a weighted blend of all tokens' values — weighted by relevance. Multi-head attention runs several of these in parallel with different learned projections so the model can capture different types of relationships simultaneously.

**Possible Follow-up:** "Why divide by √dₖ specifically?" → Because dot-product magnitude grows with vector dimensionality, and unscaled large values push softmax into a near one-hot regime with vanishing gradients.

---

### Q: RAG vs Fine-tuning — when would you use which?
**Short Interview Answer:** "RAG is for injecting knowledge — facts that change or are private. Fine-tuning is for changing behavior — tone, format, or a specific narrow skill. If the problem is 'the model doesn't know X,' use RAG. If the problem is 'the model doesn't behave the way I need consistently,' consider fine-tuning — but only after prompting alone isn't enough."

**Detailed Explanation:** See Section 9 table. In practice, most systems start with prompting, add RAG when they need grounded/current/private knowledge, and only add fine-tuning when neither prompting nor RAG can reliably enforce a required behavior/format/style at scale.

**Possible Follow-up:** "Can you combine RAG and fine-tuning?" → Yes — fine-tune the model to better use retrieved context and follow a citation format, while RAG still supplies the facts.

---

### Q: Why does RAG reduce but not eliminate hallucination?
**Short Interview Answer:** "Because RAG only guarantees the model has access to relevant context — it doesn't guarantee retrieval finds the right chunks, or that the model actually uses them faithfully instead of blending in its own prior knowledge."

**Detailed Explanation:** Failure can occur at retrieval (wrong/incomplete chunks retrieved due to poor chunking, ambiguous query, or embedding limitations) or at generation (model ignores retrieved context, or mixes correct retrieved facts with fabricated details). This is why RAG systems need groundedness/faithfulness evaluation, not just "did we retrieve something."

**Possible Follow-up:** "How would you measure this?" → Faithfulness scoring — checking each claim in the answer against the retrieved context (Section 14).

---

### Q: What is an AI agent, and how does the agent loop work?
**Short Interview Answer:** "An agent is an LLM system that can reason, choose tools, take actions, observe the results, and repeat — until it reaches a final answer, instead of just replying with text in one shot."

**Detailed Explanation:** The loop: the LLM receives a goal, decides whether it needs a tool, calls the tool if so, receives an observation (the tool's output), feeds that observation back into its reasoning, and repeats until it has enough information to give a final answer. The ReAct pattern formalizes this as interleaved Thought → Action → Observation steps. Concretely this requires tool binding (registering available tools/schemas with the model), tool calling (the model emitting a structured call), and tool execution (your code running it and returning the result).

**Possible Follow-up:** "What can go wrong?" → Loops, wrong tool selection, cascading errors, cost/latency blowup (Section 12).

---

### Q: What's the difference between LoRA and full fine-tuning?
**Short Interview Answer:** "Full fine-tuning updates every weight in the model, which is expensive and memory-heavy. LoRA freezes the original weights and injects small trainable low-rank matrices into specific layers, so you train a tiny fraction of parameters — much cheaper, and you can swap adapters in and out without touching the base model."

**Detailed Explanation:** See Section 10. QLoRA additionally quantizes the frozen base model (e.g., to 4-bit) to cut memory further, letting you fine-tune large models on modest hardware, at a small risk of extra precision loss.

**Possible Follow-up:** "When would you still choose full fine-tuning?" → When you need maximum capability change and have the compute budget/data volume to justify it — rare in most applied settings today.

---

### Q: What is MCP, and how is it different from regular function calling?
**Short Interview Answer:** "MCP is a standardized protocol for connecting LLM applications to external tools, data, and prompts — instead of every app building custom, one-off integrations for every tool, you build one MCP server per system, and any MCP-compatible client can use it."

**Detailed Explanation:** See Section 13. Function calling is the underlying mechanism (model emits a structured call, app executes it); MCP standardizes *how that connection is set up* across tools/resources/prompts, client and server, so integrations become reusable across different applications instead of tightly coupled to one codebase.

**Possible Follow-up:** "Where would MCP not be necessary?" → A simple app with one or two hardcoded tools may not need the overhead of a full MCP server — direct function calling is simpler there.

---

## 21. Common Interview Traps

- **"RAG eliminates hallucination"** — wrong; it reduces it. Retrieval can fail, and the model can still ignore/misuse context.
- **"Fine-tuning is how you keep a model's knowledge up to date"** — wrong; that's RAG's job. Fine-tuning changes behavior/style, not a reliable way to inject fresh facts.
- **"Bigger context window makes RAG unnecessary"** — long context has cost, latency, and "lost in the middle" retrieval-quality problems; it doesn't replace targeted retrieval at scale.
- **"Top-k and top-p are the same thing"** — top-k = fixed candidate count; top-p = dynamic cumulative-probability cutoff. Confusing these is a common tell of surface-level knowledge.
- **"An embedding model and a chat/generation model are interchangeable"** — they're trained for different objectives (representation vs. generation); using one for the other's job gives poor results.
- **"Prompt engineering and fine-tuning solve the same problem"** — prompting steers within existing knowledge/behavior; fine-tuning actually changes the model's weights/behavior.
- **"Attention lets the model know word order"** — it doesn't, by itself; that's what positional encoding is for. Attention alone is permutation-invariant.
- **"An AI Agent is just a chatbot with more steps"** — the defining trait is autonomous tool use and iterative decision-making based on observations, not just a longer conversation.
- **"MCP is just another name for function calling"** — MCP is a standardized protocol layer (client-server, tools+resources+prompts) built on top of the general concept of tool calling, meant for reusability across applications.
- **"Cosine similarity measures how close two vectors' values are"** — it measures the *angle/direction* between vectors, ignoring magnitude; two vectors can be far apart in raw distance but still have high cosine similarity if they point the same direction.
- **Confusing chunk size trade-offs** — assuming "smaller chunks are always better for precision" ignores that too-small chunks lose surrounding context and can hurt answer quality even when retrieval precision looks fine.

---

*End of handbook*
