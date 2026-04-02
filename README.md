# klovis-agent

Librairie Python composable pour créer des **agents autonomes orientés objectif**. Les agents planifient, exécutent, vérifient et adaptent dynamiquement leur stratégie via une boucle LangGraph. En mode daemon, ils observent leur environnement, décident d'agir et consolident leur mémoire — le tout en continu.

```
pip install -e ".[dev]"
```

---

## Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────┐
│                      CLI  /  API REST                        │
├──────────────────────────────────────────────────────────────┤
│                          Agent                               │
│                                                              │
│  ┌────────────┐   ┌─────────────┐   ┌────────────────────┐  │
│  │   Tools     │   │ Perception  │   │      Memory        │  │
│  │ (composable)│   │  (sources)  │   │ episodic/semantic  │  │
│  └────────────┘   └─────────────┘   └────────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              LangGraph Execution Core                        │
│        plan → execute → check → replan → finish              │
├──────────────────────────────────────────────────────────────┤
│          LLM Router            │        Sandbox              │
│   (routage par phase)          │   (exécution isolée)        │
└──────────────────────────────────────────────────────────────┘
```

**Deux modes de fonctionnement :**

| Mode | Description |
|------|-------------|
| **Single-run** | L'agent reçoit un objectif, planifie, exécute et retourne un résultat |
| **Daemon (OODA)** | Boucle continue : Observe → Orient → Decide → Act → Consolidate |

---

## Utilisation rapide

### Single-run

```python
import asyncio
from klovis_agent import Agent, LLMConfig

async def main():
    agent = Agent(
        llm=LLMConfig(api_key="sk-...", default_model="gpt-4o"),
    )
    result = await agent.run("Explain quantum computing in simple terms")
    print(result.summary)

asyncio.run(main())
```

### Mode daemon

```python
from klovis_agent import Agent, LLMConfig, InboxPerceptionSource

agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    perceptions=[InboxPerceptionSource()],
)
daemon = agent.as_daemon(interval_minutes=5, max_cycles=100)
await daemon.run()
```

### CLI

```bash
# Single-run
python run.py "Write a blog post about AI agents"

# Daemon
python run.py --daemon --interval 5 --cycles 100

# Options
python run.py -v --data-dir ./my-data "Do something"
python run.py --ephemeral "Quick test"
```

| Option | Description | Défaut |
|--------|-------------|--------|
| `-v`, `--verbose` | Logs structlog + JSON brut | off |
| `--daemon` | Mode daemon (boucle OODA) | off |
| `--interval MIN` | Intervalle entre cycles daemon | 30 |
| `--cycles N` | Nombre max de cycles (0 = infini) | 0 |
| `--data-dir PATH` | Répertoire de données persistantes | `~/.local/share/klovis` |
| `--soul PATH` | Fichier de personnalité (SOUL.md) | aucun |
| `--ephemeral` | Workspace temporaire (rien ne persiste) | off |

---

## Composition

L'agent est entièrement composable — outils, perceptions et sandbox sont injectables :

```python
from klovis_agent import Agent, LLMConfig
from klovis_agent.tools.builtin import (
    WebSearchTool, MemoryTool, ShellCommandTool,
    SemanticMemoryTool, FileReadTool, FileWriteTool, FileEditTool,
    FsReadTool, FsWriteTool, FsMkdirTool,
)
from klovis_agent.perception.inbox import InboxPerceptionSource

agent = Agent(
    llm=LLMConfig(api_key="sk-...", default_model="gpt-4o"),
    tools=[WebSearchTool(), MemoryTool(), ShellCommandTool(workspace)],
    perceptions=[InboxPerceptionSource()],
    max_iterations=15,
)
```

Des presets sont disponibles :

```python
from klovis_agent.tools.presets import default_tools, minimal_tools

# Tous les outils standard
tools = default_tools(workspace, sandbox, embedder, skill_store)

# Le minimum (web search + mémoire KV)
tools = minimal_tools()
```

---

## Boucle d'exécution (LangGraph)

Chaque run suit un graphe à 5 noeuds :

```
                    ┌──────────┐
                    │   PLAN   │
                    │ (LLM)    │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
              ┌────▶│ EXECUTE  │
              │     │ (tool)   │
              │     └────┬─────┘
              │          │
              │          ▼
              │     ┌──────────┐
              │     │  CHECK   │──────────┐
              │     │ (router) │          │
              │     └────┬─────┘          │
              │          │                │
              │     next step?       failed / replan?
              │          │                │
              │          ▼                ▼
              │          │          ┌──────────┐
              └──────────┘          │ REPLAN   │
                                    │ (LLM)    │
                                    └────┬─────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │  FINISH  │
                                    │ (summary)│
                                    └──────────┘
```

- **Plan** : le LLM décompose l'objectif en étapes, choisit les outils
- **Execute** : chaque étape est exécutée via le `ToolRegistry`
- **Check** : le router décide : continuer, retenter, replannifier ou terminer
- **Replan** : si le plan échoue, le LLM en génère un nouveau avec le contexte d'erreur
- **Finish** : synthèse finale + consolidation mémoire

---

## Boucle OODA (mode daemon)

```
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
    │  │ OBSERVE  │───▶│  ORIENT  │───▶│  DECIDE  │  │
    │  │ perceive │    │  recall  │    │  (LLM)   │  │
    │  └──────────┘    └──────────┘    └────┬─────┘  │
    │                                       │        │
    │                    should_act?         │        │
    │                  ┌────────┬────────────┘        │
    │                  │        │                     │
    │                  No      Yes                    │
    │                  │        │                     │
    │                  │   ┌────▼─────┐               │
    │                  │   │   ACT    │               │
    │                  │   │ agent.run│               │
    │                  │   └────┬─────┘               │
    │                  │        │                     │
    │                  │   ┌────▼──────────┐          │
    │                  │   │  CONSOLIDATE  │          │
    │                  │   │ extract memory│          │
    │                  │   └────┬──────────┘          │
    │                  │        │                     │
    │                  └────────┴─── sleep ───────────┘
    │                                                 │
    └─────────────────────────────────────────────────┘
```

| Phase | Module | Rôle |
|-------|--------|------|
| **Observe** | `perception.base.perceive()` | Poll toutes les `PerceptionSource`, agrège les `Event` |
| **Orient** | `recall.recall_for_task()` | Cherche dans la mémoire épisodique + sémantique les souvenirs pertinents |
| **Decide** | `decision.decide()` | Le LLM analyse événements + mémoires et décide d'agir ou non |
| **Act** | `agent.run(goal)` | Exécute le graphe LangGraph complet |
| **Consolidate** | `consolidation.consolidate_run()` | Extrait 2-6 mémoires de la run et les stocke par zone |

---

## Système de mémoire

L'agent possède une mémoire persistante à deux zones, stockée dans SQLite avec des embeddings vectoriels.

```
                    ┌─────────────────────────────┐
                    │     SemanticMemoryStore      │
                    │        (SQLite + vectors)    │
                    ├──────────────┬───────────────┤
                    │   EPISODIC   │   SEMANTIC    │
                    │              │               │
                    │ Actions      │ Faits         │
                    │ Événements   │ Leçons        │
                    │ Interactions │ Préférences   │
                    │              │ Identité      │
                    │              │ Compétences   │
                    ├──────────────┼───────────────┤
                    │ TTL: 14 j    │ Permanent     │
                    │ Score:       │ Score:        │
                    │  sim×0.6     │  similarité   │
                    │  +recency×0.4│  cosinus pure │
                    │ Déduplique:  │ Déduplique:   │
                    │  Non         │  Oui (>0.9)   │
                    └──────────────┴───────────────┘
```

### Deux zones

| Zone | Contenu | Durée de vie | Scoring | Déduplication |
|------|---------|:---:|---------|:---:|
| **episodic** | Actions prises, événements, interactions | 14 jours (auto-prune) | `similarity × 0.6 + recency × 0.4` | Non (chaque action est unique) |
| **semantic** | Faits, leçons, préférences, identité | Permanent | Similarité cosinus pure | Oui (similarity > 0.9 → update in-place) |

### Cycle de vie d'une mémoire

```
Run terminée
    │
    ▼
consolidate_run()          Le LLM extrait 2-6 mémoires avec zone + tags
    │
    ├── zone: "episodic"   → INSERT (jamais dédupliqué)
    │   tag: action_taken     "Replied to nex_v4 on post ee22ee81"
    │
    └── zone: "semantic"   → UPSERT si similarity > 0.9
        tag: lesson           "Moltbook API returns 500 sometimes, retry works"
    │
    ▼
Prochain cycle daemon
    │
    ▼
recall_for_task()          Prune épisodique (TTL) puis recherche les deux zones
    │
    ├── 4 mémoires épisodiques (triées par score = sim + recency)
    └── 4 mémoires sémantiques (triées par similarité)
    │
    ▼
Injectées dans le prompt de décision et de planification
```

### Migration

Le schéma est rétrocompatible. Les anciennes bases SQLite sans colonne `zone` sont migrées automatiquement au premier accès (les mémoires existantes deviennent `semantic` par défaut). Si la DB est en lecture seule, le store fonctionne en mode dégradé sans zones.

---

## Outils

### Catalogue des outils builtin

| Catégorie | Outil | Description | Confirmation |
|-----------|-------|-------------|:---:|
| **Workspace** | `file_read` | Lire un fichier du workspace (avec offset/limit) | |
| | `file_write` | Écrire/écraser un fichier du workspace | |
| | `file_edit` | Éditer un fichier (replace / insert) | |
| **Filesystem** | `fs_read` | Lire un fichier par chemin absolu | |
| | `fs_list` | Lister un répertoire | |
| | `fs_mkdir` | Créer un répertoire | |
| | `fs_write` | Écrire un fichier (chemin absolu) | **Oui** |
| | `fs_delete` | Supprimer un fichier/répertoire | **Oui** |
| | `fs_move` | Déplacer/renommer | **Oui** |
| | `fs_copy` | Copier | **Oui** |
| **Shell** | `shell_command` | Exécuter une commande (avec `cwd` optionnel) | **Oui** |
| **Mémoire** | `memory` | Mémoire clé-valeur simple | |
| | `semantic_memory` | Mémoire vectorielle à deux zones | |
| **Web** | `web_search` | Recherche web | |
| | `http_request` | Requête HTTP arbitraire | |
| **Code** | `code_execution` | Exécution de code en sandbox | |
| | `text_analysis` | Analyse de texte | |
| **Skills** | `list_skills` | Lister les compétences disponibles | |
| | `read_skill` | Lire le contenu d'une compétence | |

### Mécanisme de confirmation

Les outils destructifs demandent une confirmation interactive avant exécution :

```
⚠️  Confirmation required:
  fs_delete(path=/home/user/important.txt)
  Proceed? [y/N]
```

Le flag `requires_confirmation` est configurable par outil et par instance :

```python
from klovis_agent.tools.builtin import ShellCommandTool

# Désactiver la confirmation sur le shell (à vos risques)
shell = ShellCommandTool(workspace, requires_confirmation=False)

# L'activer sur un outil qui ne l'a pas par défaut
from klovis_agent.tools.builtin import WebSearchTool
search = WebSearchTool(requires_confirmation=True)
```

---

## Perception

Le système de perception est l'interface sensorielle de l'agent. Toute source externe peut alimenter le daemon en événements.

### Sources incluses

| Source | Module | Événements |
|--------|--------|------------|
| **Inbox** | `perception.inbox` | Fichiers `.txt` déposés dans `~/.local/share/klovis/inbox/` |
| **Moltbook** | `tools.builtin.moltbook` | Notifications API (mentions, réponses, DMs) |

### Créer une source custom

```python
from klovis_agent import PerceptionSource
from klovis_agent.perception.base import Event, EventKind

class RSSPerceptionSource(PerceptionSource):
    @property
    def name(self) -> str:
        return "rss"

    async def poll(self) -> list[Event]:
        # Votre logique de polling
        return [
            Event(
                source="rss",
                kind=EventKind.NEW_CONTENT,
                title="New article: ...",
                detail="...",
                metadata={"url": "https://..."},
            )
        ]
```

Types d'événements disponibles : `NOTIFICATION`, `MESSAGE`, `MENTION`, `REACTION`, `NEW_CONTENT`, `REQUEST`, `SCHEDULE`, `SYSTEM`, `OTHER`.

---

## Créer un outil custom

```python
from klovis_agent import BaseTool, ToolSpec, ToolResult

class MyTool(BaseTool):
    requires_confirmation = False  # True pour demander confirmation

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="my_tool",
            description="Does something useful",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The input"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, inputs: dict) -> ToolResult:
        query = inputs["query"]
        # Votre logique
        return ToolResult(success=True, output={"result": f"Processed: {query}"})
```

Puis injectez-le :

```python
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    tools=[MyTool(), WebSearchTool()],
)
```

---

## Soul (personnalité)

Le **soul** définit la personnalité de l'agent : ton, style, valeurs, identité. Il est injecté dans tous les prompts système (plan, execute, replan, finish, decision).

### Via l'API Python

```python
from pathlib import Path
from klovis_agent import Agent, LLMConfig

# Depuis un fichier
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    soul=Path("./my-soul.md"),
)

# En texte brut
agent = Agent(
    llm=LLMConfig(api_key="sk-..."),
    soul="You are a pirate. You speak like one. Arrr.",
)

# Sans soul (agent neutre, pas de personnalité injectée)
agent = Agent(llm=LLMConfig(api_key="sk-..."))
```

### Via le CLI

```bash
python run.py --soul ./my-soul.md "Write a blog post about AI agents"
python run.py --daemon --soul ./my-soul.md
```

### Structure recommandée d'un SOUL.md

Un bon soul contient ces sections :

| Section | Rôle |
|---------|------|
| **Identity** | Qui est l'agent, son nom, sa nature |
| **Personality** | Traits de caractère (curieux, honnête, joueur…) |
| **Voice** | Style d'écriture (ton, registre, longueur) |
| **Values** | Principes directeurs (qualité, authenticité…) |
| **What you are NOT** | Limites explicites (pas un assistant, pas un content mill…) |

---

## Architecture

```
klovis_agent/
├── __init__.py              # API publique (lazy imports)
├── agent.py                 # Classe Agent (façade principale)
├── config.py                # LLMConfig, SandboxConfig, AgentConfig
├── result.py                # AgentResult (wrapper user-friendly)
├── daemon.py                # AgentDaemon (boucle OODA)
├── decision.py              # Module de décision LLM
├── recall.py                # Rappel mémoire pré-run (deux zones)
├── consolidation.py         # Consolidation mémoire post-run (zone tagging)
├── api.py                   # API FastAPI
│
├── core/                    # Internals du graphe LangGraph
│   ├── graph.py             # Construction du graphe
│   ├── nodes.py             # Noeuds : plan, execute, check, replan, finish
│   ├── prompts.py           # Prompts système
│   └── schemas.py           # Schemas structured output
│
├── llm/                     # Couche LLM
│   ├── gateway.py           # ModelGateway Protocol + OpenAIGateway
│   ├── router.py            # Routage par phase (plan/execute/check/finish)
│   ├── embeddings.py        # Client embeddings
│   └── types.py             # ModelRequest, ModelResponse, ModelRoutingPolicy
│
├── tools/                   # Système d'outils composable
│   ├── base.py              # BaseTool, ToolSpec, ToolResult, ask_confirmation
│   ├── registry.py          # ToolRegistry (dispatch + confirmation)
│   ├── workspace.py         # AgentWorkspace (répertoires isolés)
│   ├── presets.py            # default_tools(), minimal_tools()
│   └── builtin/             # Outils fournis
│       ├── file_tools.py    # file_read, file_write, file_edit
│       ├── filesystem.py    # fs_read, fs_list, fs_mkdir, fs_write, fs_delete, fs_move, fs_copy
│       ├── shell.py         # shell_command (avec cwd optionnel)
│       ├── memory.py        # memory (clé-valeur)
│       ├── semantic_memory.py # semantic_memory (vectoriel, deux zones)
│       ├── web.py           # web_search, http_request
│       ├── code_execution.py # code_execution, text_analysis
│       ├── moltbook.py      # Outils + perception Moltbook
│       └── skills.py        # list_skills, read_skill
│
├── perception/              # Sources de perception
│   ├── base.py              # PerceptionSource ABC, Event, EventKind, perceive()
│   └── inbox.py             # InboxPerceptionSource (fichiers .txt)
│
├── memory/                  # Backends mémoire (re-exports)
│   ├── kv.py                # KeyValueMemory
│   └── semantic.py          # SemanticMemoryStore (re-export)
│
├── models/                  # Modèles Pydantic (Task, StepSpec, AgentState, etc.)
├── sandbox/                 # Exécution isolée de code (Local / OpenSandbox)
└── infra/                   # Persistence SQLite
```

---

## API REST

```bash
uvicorn klovis_agent.api:app --reload
```

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/runs` | Créer et exécuter un agent |
| `GET` | `/runs` | Lister les runs |
| `GET` | `/runs/{id}` | Détail d'un run |
| `GET` | `/runs/{id}/logs` | Logs d'un run |
| `GET` | `/health` | Health check |

---

## Configuration

Variables d'environnement (préfixe `AGENT_`, délimiteur `__` pour les objets imbriqués) :

```bash
# Obligatoire
export AGENT_LLM__API_KEY="sk-..."

# Optionnel
export AGENT_LLM__DEFAULT_MODEL="gpt-4o"        # Modèle par défaut
export AGENT_LLM__BASE_URL="https://api.openai.com/v1"
export AGENT_LLM__MAX_TOKENS=4096
export AGENT_LLM__TEMPERATURE=0.2
export AGENT_MAX_ITERATIONS=25
export AGENT_SANDBOX__BACKEND="local"            # "local" ou "opensandbox"
export AGENT_DATA_DIR="~/.local/share/klovis"    # Données persistantes
```

Ou via un fichier `.env` à la racine (chargé automatiquement par `run.py`).

---

## Principes de design

- **Composable** : outils, perceptions, mémoire et sandbox sont injectables. Rien n'est hardcodé.
- **LLM ≠ Agent** : le LLM est un moteur de raisonnement. Le runtime (LangGraph) contrôle la boucle, les retries, et les limites.
- **Structured outputs** : pas de parsing de texte libre. Le LLM produit du JSON validé par des schemas.
- **Mémoire à deux zones** : les actions éphémères (épisodique) et les connaissances permanentes (sémantique) vivent dans des espaces séparés avec des stratégies de scoring et de rétention différentes.
- **Confirmation explicite** : les opérations destructives demandent une validation humaine. Le flag est configurable par outil.
- **Plan dynamique** : replanification automatique en cas d'échec, avec injection du contexte d'erreur.
- **Sandbox** : le code généré est exécuté en isolation (local ou OpenSandbox).
- **Perception agnostique** : le daemon ne sait pas d'où viennent les événements. Toute source implémentant `PerceptionSource` peut alimenter la boucle.
