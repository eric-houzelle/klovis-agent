# Memoire de l'agent

Ce document decrit comment la memoire fonctionne dans `klovis-agent`, quand elle est utilisee, pourquoi, comment elle est structuree, et comment l'agent peut la re-organiser.

## Objectif

La memoire sert a:

- garder la continuite entre les runs
- eviter de repeter les memes actions
- capitaliser sur les preferences utilisateur, strategies, lecons, identite
- donner du contexte utile au planner avant d'agir

## Vue d'ensemble

Le systeme combine **2 couches**:

1. Memoire **key-value** (simple et deterministe)
2. Memoire **semantique vectorielle** (recherche par similarite)

La memoire semantique est elle-meme separee en:

- `episodic`: actions/evenements recents (TTL)
- `semantic`: connaissances persistantes (deduplication)

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory System                          │
│                                                             │
│  ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │   Key-Value       │    │   Semantic (vectorielle)       │ │
│  │   store.json      │    │   semantic.db + embeddings     │ │
│  │                   │    │                                │ │
│  │  set / get /      │    │  ┌────────────┐ ┌───────────┐ │ │
│  │  delete / list    │    │  │  episodic   │ │ semantic  │ │ │
│  │                   │    │  │  TTL 14j    │ │ permanent │ │ │
│  │  flags, prefs,    │    │  │  recence +  │ │ cosinus   │ │ │
│  │  etat explicite   │    │  │  similarite │ │ dedup 0.9 │ │ │
│  └──────────────────┘    │  └────────────┘ └───────────┘ │ │
│                           └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 1) Memoire key-value

Implementation: `memory` tool.

- operations: `set`, `get`, `delete`, `list`
- usage: flags, preferences simples, etat explicite
- stockage: JSON sur disque (`store.json`)

Reference code: `klovis_agent/tools/builtin/memory.py`

## 2) Memoire semantique vectorielle

Implementation: `semantic_memory` tool + `SemanticMemoryStore`.

- stockage: SQLite (`semantic.db`) + embeddings
- retrieval: similarite cosinus
- zones:
  - `episodic`: score = similarite + recence, purge automatique
  - `semantic`: similarite pure, deduplication (upsert si tres proche)

Reference code: `klovis_agent/tools/builtin/semantic_memory.py`

## Quand la memoire est lue/ecrite

### Avant l'execution (recall)

Avant de planifier, l'agent:

1. purge les episodic expires
2. recherche memories pertinentes en `episodic` + `semantic`
3. injecte le resultat dans `task.context.recalled_memories`

Le planner recoit ce contexte dans son prompt (`Context: ...`), ce qui influence la strategie.

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Tache   │────>│ Purge TTL    │────>│ Recall       │────>│ Planner  │
│  recue   │     │ (episodic)   │     │ (search both │     │ (+context│
│          │     │              │     │  zones)      │     │  memoire)│
└──────────┘     └──────────────┘     └──────────────┘     └──────────┘
```

References:

- `klovis_agent/recall.py`
- `klovis_agent/agent.py`
- `klovis_agent/core/nodes.py`

### Apres l'execution (consolidation)

En fin de run, l'agent:

1. resume ce qui s'est passe
2. extrait 2-6 memories via LLM
3. classe chaque memory (`zone`, `type`, `tags`)
4. stocke dans SQLite

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Run     │────>│ Resume LLM   │────>│ Extraction   │────>│ Store    │
│  termine │     │ (ce qui s'est│     │ 2-6 memories │     │ SQLite   │
│          │     │  passe)      │     │ zone/type/   │     │ + embed  │
│          │     │              │     │ tags          │     │          │
└──────────┘     └──────────────┘     └──────────────┘     └──────────┘
```

References:

- `klovis_agent/consolidation.py`
- `klovis_agent/agent.py`

## Structure des donnees

### Schema SQLite (`memories`)

```
┌────────────────────────────────────────────────────────┐
│                    memories (table)                     │
├──────────────┬─────────────┬───────────────────────────┤
│ id           │ INTEGER PK  │ auto-increment            │
│ content      │ TEXT        │ texte memorise            │
│ metadata     │ TEXT (JSON) │ type, tags, category, ... │
│ embedding    │ TEXT (JSON) │ vecteur float[]           │
│ zone         │ TEXT        │ "semantic" | "episodic"   │
│ created_at   │ REAL        │ timestamp creation        │
│ accessed_at  │ REAL        │ timestamp dernier acces   │
│ access_count │ INTEGER     │ nombre de lectures        │
└──────────────┴─────────────┴───────────────────────────┘
```

### Champ `metadata` (JSON)

```
┌─────────────────────────────────────────────────────────┐
│                    metadata (JSON)                       │
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │  Noyau stable (memory_type)                 │        │
│  │                                             │        │
│  │  type: mission | state | preference | fact  │        │
│  │        lesson | strategy | action |         │        │
│  │        identity | other                     │        │
│  │                                             │        │
│  │  tags: ["planning", "api", ...]             │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │  Cases dynamiques (taxonomie libre)         │        │
│  │                                             │        │
│  │  category:    libre (fallback → type)       │        │
│  │  subcategory: libre, optionnel              │        │
│  │  namespace:   libre (ex: nom de projet)     │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### `memory_type` (noyau stable)

Valeurs autorisees:

- `mission`, `state`, `preference`, `fact`, `lesson`, `strategy`, `action`, `identity`, `other`

Ce champ reste volontairement borne pour garder des filtres robustes.

### Cases dynamiques

Au-dessus du noyau, on ajoute une taxonomie libre:

- `category` (libre)
- `subcategory` (libre)
- `namespace` (libre, ex: nom de projet)

Normalisation:

- stockee en minuscule
- si `category` absente, elle herite de `memory_type`

## Operations du tool `semantic_memory`

### Operations historiques

- `remember`: memoriser un contenu
- `recall`: recherche semantique
- `forget`: suppression par ID
- `stats`: stats globales

### Operations de restructuration

- `cases`: liste les categories dynamiques existantes (`category`) avec compteur
- `reclassify`: met a jour la structure metadata d'une ou plusieurs memories

```
┌─────────────────────────────────────────────────────────────────┐
│                   Operations disponibles                        │
│                                                                 │
│  CRUD                          Introspection / Restructuration  │
│  ──────────────                ──────────────────────────────── │
│  remember  ──> store           stats      ──> compteurs/recent  │
│  recall    ──> search cosinus  cases      ──> categories list   │
│  forget    ──> delete by ID    reclassify ──> update metadata   │
└─────────────────────────────────────────────────────────────────┘
```

## Filtres disponibles dans `recall`

Filtres par structure stable:

- `zone`
- `memory_type`
- `memory_types`

Filtres par cases dynamiques:

- `category`
- `categories`
- `subcategory`
- `namespace`

```
recall(query)
  │
  ├── zone filter?         ──> episodic | semantic | both
  ├── memory_type filter?  ──> mission, fact, ...
  ├── memory_types filter? ──> [mission, preference]
  ├── category filter?     ──> "workflow"
  ├── categories filter?   ──> ["workflow", "api"]
  ├── subcategory filter?  ──> "planning"
  └── namespace filter?    ──> "autonomous-agent"
        │
        v
  results triees par score (similarite + recence pour episodic)
```

## Comment l'agent peut se restructurer "tout seul"

Oui, il peut, avec ses tools.

Workflow recommande:

```
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌────────────┐     ┌─────────┐
  │ 1.stats │────>│ 2.cases │────>│ 3.recall│────>│4.reclassify│────>│ 5.check │
  │ volume/ │     │ observer│     │ auditer │     │ renommer/  │     │ stats/  │
  │ bruit   │     │ taxo    │     │ cible   │     │ fusionner  │     │ cases   │
  └─────────┘     └─────────┘     └─────────┘     └────────────┘     └─────────┘
```

1. `semantic_memory` -> `stats` pour detecter volume/bruit
2. `semantic_memory` -> `cases` pour observer les cases actuelles
3. `semantic_memory` -> `recall` cible (par zone/type/category) pour auditer
4. `semantic_memory` -> `reclassify` pour renommer/fusionner/reorganiser
5. re-check avec `stats`/`cases`

Exemple d'operation `reclassify`:

```json
{
  "operation": "reclassify",
  "memory_ids": [12, 15, 19],
  "category": "workflow",
  "subcategory": "memory",
  "namespace": "autonomous-agent",
  "merge_tags": true
}
```

## Exemples utiles

Memoriser avec case dynamique:

```json
{
  "operation": "remember",
  "content": "Use `list_skills` before complex API work.",
  "zone": "semantic",
  "memory_type": "strategy",
  "category": "workflow",
  "subcategory": "planning",
  "namespace": "autonomous-agent",
  "tags": ["skills", "planning"]
}
```

Recall sur une case specifique:

```json
{
  "operation": "recall",
  "content": "How to reorganize memory categories",
  "zone": "semantic",
  "category": "workflow",
  "namespace": "autonomous-agent",
  "k": 8
}
```

Lister les cases:

```json
{
  "operation": "cases"
}
```

## Cycle de vie complet d'une memory

```
  Tache recue
      │
      v
  ┌────────────────┐
  │ Purge episodic  │  (TTL 14 jours)
  │ expires         │
  └───────┬────────┘
          v
  ┌────────────────┐
  │ Recall          │  search cosinus sur les 2 zones
  │ (pre-planning)  │  → injecte dans contexte planner
  └───────┬────────┘
          v
  ┌────────────────┐
  │ Planner + Exec  │  utilise le contexte memoire
  └───────┬────────┘
          v
  ┌────────────────┐
  │ Consolidation   │  LLM extrait 2-6 memories
  │ (post-run)      │  zone + type + tags
  └───────┬────────┘
          v
  ┌────────────────┐
  │ Store SQLite    │  dedup semantic (sim > 0.9 → upsert)
  │ + embedding     │  episodic toujours insere
  └────────────────┘
```

## Ce que le systeme fait deja automatiquement

- purge TTL de `episodic`
- dedup sur `semantic`
- injection de memories dans le contexte de planification
- consolidation post-run

## Limites actuelles

- `memory_type` ne se cree pas dynamiquement (liste fixe, inconnu => `other`)
- pas encore de "merge intelligent" automatique entre categories proches
- pas de workflow natif de rollback pour `reclassify` (a ajouter si besoin)

## Bonnes pratiques

- garder `memory_type` stable et utiliser `category/subcategory/namespace` pour la souplesse
- preferer des categories courtes et coherentes (`workflow`, `api`, `user-preferences`, etc.)
- utiliser `namespace` pour eviter les collisions multi-projets
- lancer regulierement `cases` + `stats` pour controler la qualite de la taxonomie
