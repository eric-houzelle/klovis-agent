# Diagrams

## PNG exports (hand-drawn)

Static images checked into [`assets/`](./) for the README and slides:

| File | Topic |
|------|--------|
| `schema1.png` | LangGraph execution loop |
| `schema2.png` | Reactive daemon + `EventBus` |
| `schema3.png` | Two-zone memory lifecycle |
| `schema4.png` | Skill acquisition lifecycle |

---

## Mermaid (editable)

Alternative Mermaid versions of the README diagrams.
Paste into any Mermaid-compatible renderer (GitHub, Notion, mermaid.live, etc.).

---

## Execution loop (LangGraph)

```mermaid
flowchart TD
    PLAN["🧠 PLAN<br/><i>LLM breaks down the goal into steps</i>"]
    EXECUTE["⚡ EXECUTE<br/><i>Run tool via ToolRegistry</i>"]
    CHECK{"🔍 CHECK<br/><i>Router decides next action</i>"}
    REPLAN["🔄 REPLAN<br/><i>LLM generates new plan<br/>with error context</i>"]
    FINISH["✅ FINISH<br/><i>Summary + memory consolidation</i>"]

    PLAN --> EXECUTE
    EXECUTE --> CHECK
    CHECK -- "next step" --> EXECUTE
    CHECK -- "retry" --> EXECUTE
    CHECK -- "replan" --> REPLAN
    CHECK -- "finish" --> FINISH
    REPLAN --> EXECUTE
```

---

## Reactive daemon (event-driven)

```mermaid
flowchart TD
    subgraph PERCEPTION["🎛️ Perception Layer"]
        direction LR
        S1["Moltbook<br/>Listener"]
        S2["Discord<br/>Listener"]
        S3["GitHub<br/>Listener"]
        S4["Inbox<br/>Watcher"]
    end

    BUS["📬 EVENT BUS<br/><i>async queue</i>"]

    S1 -->|push| BUS
    S2 -->|push| BUS
    S3 -->|push| BUS
    S4 -->|push| BUS

    subgraph REACTIVE["⚙️ Reactive Loop"]
        DRAIN["Drain bus<br/><i>blocks until events arrive</i>"]
        RECALL["Recall memories<br/><i>semantic + episodic</i>"]
        DECIDE{"🧠 DECIDE<br/><i>LLM: should I act?</i>"}
    end

    BUS --> DRAIN
    DRAIN --> RECALL
    RECALL --> DECIDE

    DECIDE -- "No → idle" --> DRAIN

    ACT["⚡ agent.run(goal)<br/><i>LangGraph execution</i>"]
    CONSOLIDATE["💾 CONSOLIDATE<br/><i>extract memories</i>"]
    REDRAIN["🔄 Re-drain bus<br/><i>process events arrived<br/>during action</i>"]

    DECIDE -- "Yes → act" --> ACT
    ACT --> CONSOLIDATE
    CONSOLIDATE --> REDRAIN
    REDRAIN --> DRAIN

    style PERCEPTION fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style REACTIVE fill:#0f3460,stroke:#16213e,color:#e0e0e0
    style BUS fill:#e94560,stroke:#e94560,color:#fff
    style DECIDE fill:#533483,stroke:#533483,color:#fff
    style ACT fill:#0f3460,stroke:#16213e,color:#e0e0e0
```

---

## Memory system

```mermaid
flowchart TD
    RUN["Run completed"]
    CONSOL["consolidate_run()<br/><i>LLM extracts 2-6 memories</i>"]
    
    RUN --> CONSOL

    subgraph ZONES["Memory Zones"]
        direction LR
        EPISODIC["📝 Episodic<br/><i>TTL: 14 days</i><br/><i>score = sim × 0.6 + recency × 0.4</i><br/><i>INSERT only</i>"]
        SEMANTIC["🧠 Semantic<br/><i>Permanent</i><br/><i>score = cosine similarity</i><br/><i>UPSERT if sim > 0.9</i>"]
    end

    CONSOL -->|"zone: episodic<br/>tag: action_taken"| EPISODIC
    CONSOL -->|"zone: semantic<br/>tag: lesson"| SEMANTIC

    NEXT["Next daemon cycle"]
    RECALL["recall_for_task()<br/><i>Prune episodic (TTL)<br/>then search both zones</i>"]

    EPISODIC --> RECALL
    SEMANTIC --> RECALL
    RECALL -->|"4 episodic + 4 semantic<br/>memories injected"| NEXT

    style EPISODIC fill:#2d6a4f,stroke:#1b4332,color:#fff
    style SEMANTIC fill:#6a040f,stroke:#370617,color:#fff
```

---

## Skill lifecycle

```mermaid
flowchart TD
    NEED["Need identified<br/><i>during planning</i>"]
    RECALL["recall_for_task()<br/><i>semantic search over<br/>indexed skills</i>"]
    FOUND{"Found?"}
    READ["read_skill<br/><i>use it in the run</i>"]
    SEARCH["search_remote_skills()<br/><i>skillshub.wtf +<br/>skillsdirectory.com</i>"]
    INSTALL["install_skill()<br/><i>download SKILL.md<br/>from GitHub</i>"]
    INDEX["SkillIndex<br/><i>vectorise and store<br/>summary</i>"]
    MEMORIZE["consolidate_run()<br/><i>memorise 'I learned<br/>skill X for Y'</i>"]
    NEXT["Next run<br/><i>skill found by recall<br/>automatically</i>"]

    NEED --> RECALL
    RECALL --> FOUND
    FOUND -- "Yes" --> READ
    FOUND -- "No" --> SEARCH
    SEARCH --> INSTALL
    INSTALL --> INDEX
    INDEX --> READ
    READ --> MEMORIZE
    MEMORIZE --> NEXT
```
