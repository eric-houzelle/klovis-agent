# Au-delà du chatbot : async, perceptions et boucles de réflexion pour un agent orienté objectifs

**Sous-titre :** Comment **klovis-agent** sépare l’écoute du monde (perceptions parallèles, bus d’événements) de l’exécution (LangGraph), puis fige l’expérience en mémoire à deux vitesses — sans confondre “longue réponse” et “raisonnement structuré”.

---

## Le problème que l’async résout (et celui qu’il crée)

Un agent “branché au monde” vit deux temporalités en même temps :

1. **Le monde envoie des signaux** de façon irrégulière : webhooks, bot Discord, notifications GitHub, fichiers déposés dans une inbox, APIs sociales, etc.
2. **L’agent exécute** une tâche qui peut prendre du temps : plusieurs tours LLM, outils, I/O réseau.

Si tout est traité de façon **séquentielle et bloquante**, on obtient vite l’un de ces défauts : soit un **polling** avec un `sleep` fixe (latence artificielle, réveils inutiles), soit une **surdité** pendant l’exécution (événements perdus ou traités trop tard).

**klovis-agent** part d’un choix simple : **asyncio comme colonne vertébrale**, avec une séparation nette entre **producteurs d’événements** (les perceptions) et **consommateur décisionnel** (la boucle du daemon).

Concrètement, le bus est une **`asyncio.Queue`** : chaque perception peut `await put(event)` sans connaître le reste du système ; la boucle centrale consomme avec `get` / `drain` sans mélanger cette mécanique avec la logique des outils.

---

## Perceptions : plusieurs oreilles, une seule interface

Les perceptions ne sont pas un détail d’intégration. Elles portent une idée forte : **le daemon ne sait pas d’où viennent les événements** — il ne fait qu’en consommer.

Une source implémente une abstraction du type **`PerceptionSource`** : nom, rythme (`poll_interval` pour le mode poll), et une méthode qui renvoie des **`Event`**. Chaque événement porte une **taxonomie** (message, mention, requête, notification, nouveau contenu, etc.), un titre, un détail, des métadonnées. Ce vocabulaire commun permet de **normaliser** Discord, GitHub, Moltbook ou une inbox fichier : même objet mental pour le module de décision.

Le gain async est direct : **chaque source peut vivre dans sa propre tâche**, à son propre rythme. L’inbox peut être lente, Discord peut être plus “push”, GitHub peut suivre un intervalle différent. On évite la “mega boucle” où un retard sur une intégration bloque toutes les autres.

---

## Le bus : attendre intelligemment, puis regrouper

Le bus d’événements n’est pas seulement une file. Il implémente un compromis très “runtime” :

- on **bloque jusqu’au premier événement** (ou jusqu’à un timeout) ;
- puis on **récupère goulûment** les événements qui arrivent dans une courte fenêtre.

Intérêt produit : transformer un flot de micro-signaux en **lots** exploitables pour le LLM (moins de décisions du type “un event = un run complet”), tout en restant **réactif** au premier signal. C’est une forme de **micro-batching** naturelle au monde async, sans scheduler externe obligatoire.

Pendant qu’un objectif est exécuté via le graphe principal, **les listeners continuent**. Quand l’action se termine, on **re-vide** le bus pour traiter ce qui s’est accumulé. C’est le cœur de la promesse “rien n’est perdu” : l’async sert ici à **ne pas couper l’écoute** pendant l’exécution.

---

## Trois temporalités de réflexion (et pourquoi ce n’est pas redondant)

### 1) LangGraph : penser en graphe, pas en prompt unique

Quand l’agent “fait une tâche”, la réflexion est structurée en **nœuds** : plan → exécution → vérification / routage → replan éventuel → finition.

Ce n’est pas une conversation libre : c’est une **machine à états** où le LLM est invoqué **à des moments précis** avec des responsabilités précises. Planifier n’est pas la même opération que vérifier, ni que replanifier après une erreur. Ça réduit les modes dégénérés du type “je réécris tout depuis zéro à chaque tour”.

### 2) Décision daemon : réfléchir à *s’il faut agir*

En mode continu, une autre boucle intervient : on agrège des événements, on injecte du contexte (mémoire, parfois des directives persistantes), puis un module de **décision** répond à une question différente du planificateur : *est-ce que ça vaut un run complet ?*

C’est à la fois une **garde cognitive** et une **garde de coût** : éviter de lancer une exécution multi-étapes pour du bruit.

### 3) Consolidation : réflexion *après coup* (mémoire)

Après un run, une phase de **consolidation** demande explicitement une distillation : le modèle extrait un petit nombre de souvenirs **structurés** (contenu, tags, zone mémoire, type : leçon, action, préférence, etc.). C’est une réflexion **rétrospective** : transformer une trace d’exécution en **connaissances réutilisables**, pas un copier-coller de logs.

En résumé :

- **prospective** : plan / replan ;
- **intra-run** : vérification / routage ;
- **post-run** : consolidation mémoire ;
- **méso-couche** (daemon) : décider d’agir.

---

## Mémoire : l’idée en une phrase

Un agent a besoin à la fois de se souvenir **de ce qui vient d’arriver** (« j’ai déjà répondu à ce fil ») et de **ce qui reste vrai dans le temps** (« cette API renvoie parfois une 500 »). Si tout vit dans le même sac, tu mélanges l’historique de la journée avec les leçons durablement utiles.

**klovis-agent** coupe donc la mémoire en **deux zones** dans le même magasin (SQLite + embeddings). Ce qui change, ce n’est pas seulement l’étiquette : ce sont la **durée de vie**, la **façon de classer** les résultats, et la **déduplication**.

---

### Où ça vit techniquement

Les souvenirs sont des lignes dans une base locale (`SemanticMemoryStore`), avec un vecteur par entrée. Chaque entrée a au minimum :

- un **texte** (le contenu du souvenir) ;
- une **zone** : `episodic` ou `semantic` ;
- des **métadonnées** : entre autres des **tags** libres (liste de chaînes) et un champ **`type`** qui classe le souvenir.

Les **`type`** reconnus par défaut (consolidation + stockage) sont au nombre de neuf. En pratique, tu peux les lire comme des tiroirs :

- **`mission`** — ce que l’agent est censé accomplir sur le long terme  
- **`state`** — état courant utile à se remémorer  
- **`preference`** — goûts ou habitudes de l’utilisateur  
- **`fact`** — fait stable (nom, ID, comportement d’une API…)  
- **`lesson`** — leçon tirée d’une erreur ou d’un succès  
- **`strategy`** — approche qui a marché et qu’on peut réutiliser  
- **`action`** — action concrète menée pendant un run  
- **`identity`** — qui est l’agent (pseudo, profil, style…)  
- **`other`** — fourre-tout typé quand rien d’autre ne colle  

Les **tags** servent à affiner sans multiplier les types (ex. `action_taken`, `moltbook`, `skill_acquired`). Ce n’est pas magique : c’est surtout utile au **rappel** et à la **lecture humaine** des traces.

---

### Zone « épisodique » — le journal des actions

**À quoi ça sert ?** À répondre à des questions du genre : *qu’est-ce que j’ai fait récemment ?*, *est-ce que j’ai déjà traité ce message ?*, *où en était-on la dernière fois ?*

**Durée de vie.** Les souvenirs épisodiques **expirent** : au-delà d’environ **14 jours**, ils sont **supprimés** automatiquement lors du prochain rappel (`prune_episodic`). L’idée est simple : une action sur un fil social ou un ticket n’a souvent plus de valeur au bout de deux semaines ; garder tout pourrait encombrer la recherche.

**Classement quand on cherche.** Pour l’épisodique, le score ne suit pas seulement la similarité avec la requête. On mélange **similarité cosinus** et **récence** : en pratique, `score ≈ 0,6 × similarité + 0,4 × poids de récence` (la récence décroit quand le souvenir vieillit). Résultat : parmi des souvenirs à peu près aussi pertinents, les **plus récents** remontent un peu plus.

**Écriture.** Les actions typiques vont ici (souvent `type: action` et un tag du genre `action_taken`). On n’essaie pas de fusionner deux actions distinctes comme si c’était la même phrase : le journal, ce sont des **lignes d’événements**, pas une encyclopédie.

---

### Zone « sémantique » — le savoir qui reste

**À quoi ça sert ?** Aux leçons, préférences, identité, faits utiles au-delà de la fenêtre des deux semaines — tout ce qui répond plutôt à : *que sais-je ?*, *comment je dois me comporter ?*

**Durée de vie.** Pas d’expiration automatique : la zone sémantique est pensée comme **persistante**.

**Classement quand on cherche.** Ici, le score de tri est la **similarité seule** (cosinus). On ne mélange pas avec la récence : un fait vieux mais juste reste pertinent si la question s’y rapporte.

**Déduplication.** Si un nouveau souvenir est **très** proche d’un ancien (similarité au-dessus d’un seuil fixé à **0,9**), le système **met à jour** l’entrée existante au lieu d’empiler des doublons. Ça évite dix variantes de « l’API X est capricieuse ».

---

### Au moment du rappel (« recall ») — ce que le planificateur voit

Avant de planifier, le module `recall_for_task` prépare un **petit bloc de texte** injecté dans le contexte. L’ordre des opérations est facile à suivre :

1. **Nettoyer** les épisodes trop vieux (pruning).
2. **Encoder** l’objectif courant en vecteur (une requête = un embedding).
3. **Chercher en parallèle dans les deux zones**, avec des plafonds par défaut : jusqu’à **4** souvenirs épisodiques et **4** sémantiques **les plus utiles** pour cet objectif.
4. Ignorer ce qui est trop faible : un plancher de **similarité minimale** (par défaut **0,30**) évite de remplir le prompt de bruit.
5. Ajouter un **troisième fil**, à part : jusqu’à **4** entrées sémantiques récentes dont le `type` est l’un de **`mission`**, **`state`**, **`preference`**, **`strategy`**. Dans le texte de rappel, ça apparaît sous une rubrique du style **« Persistent directives »** — ce sont les **lignes directrices** qu’on veut voir même si elles ne matchent pas fortement la requête du moment.
6. Si un index de skills existe : jusqu’à **3** skills **pertinents** pour l’objectif (rappel séparé : « charge `read_skill` avant d’agir »).

Dans le texte généré pour le modèle, tu retrouves donc des sections lisibles du genre :

- *Recent actions & events (episodic memory)*  
- *Permanent knowledge (semantic memory)*  
- *Persistent directives (mission/state/preferences)*  
- éventuellement *Relevant installed skills*

Chaque ligne peut afficher **tags**, **similarité**, et pour l’épisodique un **score** combinant pertinence et récence.

---

### Après la tâche (« consolidation ») — comment les souvenirs naissent

Quand un run se termine, un passage dédié demande au modèle : **qu’est-ce qui mérite d’être retenu ?** La réponse n’est pas du texte libre : c’est un **JSON** validé par schéma, avec **entre 2 et 6** souvenirs. Chaque souvenir a obligatoirement :

- `content` — une phrase courte, **autonome** (compréhensible sans relire tout le run) ;
- `tags` ;
- `zone` — `episodic` ou `semantic` ;
- `type` — un des neuf types listés plus haut.

Le prompt de consolidation insiste sur une règle de bon sens : **au moins un** souvenir épisodique qui trace **ce qui a été fait** (pour limiter les répétitions inutiles aux cycles suivants). Le reste peut être des faits ou des leçons en sémantique.

Ensuite, chaque souvenir est **embedé** et **écrit** dans le store avec métadonnées (tags, `type`, identifiant de run). C’est la phase « sommeil » : la trace brute des outils devient une **mémoire réutilisable**.

---

### En résumé : épisodique vs sémantique

**Épisodique** — Tu te poses : *qu’ai-je fait ?*, *qu’est-ce qui s’est passé ?* Ça expire au bout d’environ **14 jours**. À la recherche, on mélange **similarité** et **récence**. On n’agrège pas les « presque doublons » comme en sémantique : c’est un **journal**. Les types qu’on y voit souvent : surtout **`action`**.

**Sémantique** — Tu te poses : *que sais-je ?*, *quelle règle suivre ?* C’est **persistant**. À la recherche, tri par **similarité seule**. Si un nouveau souvenir ressemble fortement à un ancien (**similarité > 0,9**), l’ancien est **mis à jour** plutôt que dupliqué. Les types courants : **`lesson`**, **`fact`**, **`preference`**, **`identity`**, **`mission`**, **`strategy`**, **`state`**.

Les **skills** indexées restent un complément du même pipeline de rappel : elles apparaissent à côté des souvenirs pour rappeler **quelle doc ouvrir** avant une action délicate — ce n’est pas la même table, mais la même **logique de contexte** avant de planifier.

---

## Exemple réel (sortie terminal) : Discord, skill e-mail, introspection mémoire

Voici un scénario capturé en **mode daemon** (`uv run python run.py --daemon`), avec les sources **inbox, moltbook, discord, github**. Il illustre bien la pile : **perception** → **requête structurée** → **graphe multi-étapes** → **mémoire** → **skill** → **replans** → **consolidation**.

### Étape 0 — Préparer le terrain (hors Discord)

Avant la séquence Discord, un run **one-shot** en CLI peut enregistrer une information durable (par ex. *« voici ma clé API de messagerie agent, garde-la pour un envoi futur »*). Le terminal montre alors un plan en une étape (« store in persistent memory »), une exécution réussie, puis **`Memorized 4 new insights`** : la consolidation post-run a déjà distillé l’épisode en souvenirs réutilisables pour les cycles suivants.

*(Note sécurité pour tout article public : ne jamais publier de vraies clés API ni de tokens — masque ou réécrit l’exemple.)*

### Étape 1 — Sur Discord : « trouve une SKILL pour envoyer des mails »

Tu peux envoyer ce type de message au bot : l’événement entre dans le même pipeline que les autres perceptions. L’agent peut enchaîner **recherche distante de skills** (`search_remote_skills`), **installation** (`install_skill` avec confirmation si besoin), **lecture** (`read_skill`) — c’est exactement le cycle « documentation exécutable » décrit plus haut : la skill devient une *contrainte opératoire* avant les appels HTTP ou SDK.

### Étape 2 — Sur Discord : « envoie-moi un e-mail avec un résumé de ta mémoire actuelle »

C’est là que le journal devient parlant. Le daemon affiche un **cycle** où la perception résume l’intention utilisateur, puis une ligne du type **`Request from discord:`** suivie du **goal** reformulé.

Le runtime **enrichit automatiquement** le message Discord : un bloc *« Recent conversation context (oldest -> newest) »* et des **exigences de réponse** (langue, réponse utilisateur finale claire, etc.). Ce n’est pas du « prompt engineering » décoratif : ça stabilise le comportement quand le canal mélange plusieurs sujets (par ex. fragments de doc Moltbook dans l’historique).

Ensuite, le terminal montre un **plan explicite** — typiquement dans cet ordre logique :

1. **Lire la skill** liée à la messagerie (ex. *agentmail*) — *« I'm reading the agentmail docs… »*
2. **Récupérer la mémoire épisodique** — souvenirs récents d’actions.
3. **Récupérer la mémoire sémantique** — faits, leçons, identité.
4. **Récupérer les directives persistantes** — mission / état / préférences / stratégies.
5. **Composer** le corps de l’e-mail à partir de ces blocs.
6. **Envoyer** vers l’adresse demandée.

On voit alors la **boucle de réflexion intra-run** telle quelle : ce n’est pas un long monologue, c’est une **décomposition en étapes nommées**, chacune avec un statut (succès, échec outil, retry).

### Quand ça coince : replanifier au lieu de « halluciner un succès »

Dans la capture réelle, l’étape d’envoi échoue d’abord (par ex. tentative via `http_request` sans authentification correcte → **401**). Le graphe déclenche alors un **`Replanning (v2, …)`** avec une *reason* qui cite l’erreur : la stratégie change — recherche / installation de skill, relecture de doc, composition locale, nouvel envoi « via l’API de la skill ».

On observe aussi la friction **humain + async** : une **confirmation** pour installer une skill depuis une URL (`Proceed? [y/N]`), puis parfois un **échec de téléchargement** du `SKILL.md` (URL ou réseau). L’agent retente, relit ce qui est déjà installé, recompose. C’est une leçon d’architecture : **les outils échouent** ; le graphe **ré-alloue** l’attention (replan) au lieu de s’arrêter net sur le premier 401.

### Fin de run et coût cognitif

Le journal peut se terminer par un statut du type **`Partially done (30 iterations)`** : le modèle **assume l’échec partiel** (ex. brouillon d’e-mail prêt, envoi bloqué par auth / API) plutôt que de prétendre que l’e-mail est parti. Juste après : **`Memorized 6 new insights`** — la consolidation capte l’épisode (tentatives, erreurs, ce qui a été appris sur la stack mail).

Sur un run « riche », une ligne **`Cycle tokens: … (66 LLM call(s))`** montre le prix de cette richesse : beaucoup de **petits appels** (décision, plan, étapes, replans, consolidation). C’est le contre-pied du « un seul gros prompt » : **traçabilité** et **comportement de système**, au prix d’une discipline sur les budgets et le bruit des perceptions (sinon tu paies des cycles sur du vide).

### Ce que cet exemple démontre

| Idée | Ce qu’on voit dans le terminal |
|------|-------------------------------|
| **Perceptions** | `Request from discord` + goal enrichi (historique + règles de réponse). |
| **Async / daemon** | Entre-temps, d’autres cycles peuvent rester **Idle** sur du bruit (ex. notifications GitHub « 0 commit ») — garde de coût au niveau *décider d’agir*. |
| **Mémoire** | Étapes dédiées épisodique / sémantique / directives avant de rédiger le mail. |
| **Skills** | `read_skill` / recherche / install comme prérequis à l’action réseau. |
| **Replan** | Nouveau plan après erreur outil, raison affichée. |
| **Consolidation** | `Memorized N new insights` après la vague d’exécution. |

---

## Ce que ces choix changent concrètement

- **Async + bus + micro-batch** : concurrence des entrées sans explosion du nombre de runs.
- **Perceptions normalisées** : brancher des sources hétérogènes sans réécrire le cœur.
- **Graphe d’exécution** : réflexion **bornée** et **inspectable** (itérations, transitions).
- **Consolidation structurée** : mémoire qui apprend des **insights**, pas des archives illisibles.

---

## Limites honnêtes

L’async simplifie beaucoup de choses, mais il impose de penser **timeouts**, erreurs par tâche, et comportement sous charge (une perception instable ne doit pas faire tomber tout le daemon). Le batching du bus est un compromis : trop agressif, tu retardes la réaction ; trop fin, tu multiplies les décisions LLM.

Le projet est encore en **alpha** ; l’API et certains détails peuvent évoluer. La qualité en production dépend surtout des garde-fous opérationnels : confirmations sur actions destructrices, sandbox pour le code, budgets tokens, filtrage des sources.

---

## Ressources

- Dépôt : [github.com/eric-houzelle/klovis-agent](https://github.com/eric-houzelle/klovis-agent)
- Package PyPI : `klovis-agent`
- Diagrammes Mermaid (GitHub / Notion / mermaid.live) : fichier `assets/diagrams.md` dans le dépôt

---

**À propos :** **klovis-agent** est une bibliothèque Python (Apache 2.0) pour agents orientés objectifs — planification, exécution d’outils, mode daemon réactif, mémoire persistante et acquisition de skills. Auteur : Eric Houzelle.

---

### Note pour Medium (pas du Markdown natif)

Medium **n’est pas** un éditeur Markdown : c’est un **texte riche** (blocs). À la **saisie**, l’éditeur reconnaît souvent des **raccourcis de style Markdown** (`#` / `##` / `###`, `**gras**`, listes `*`, `>`, blocs code avec trois backticks, `---` + Entrée pour un séparateur, etc.) et les convertit en formatage — le comportement exact peut varier selon la plateforme et les mises à jour.

En **collant** tout un article `.md`, le résultat est **souvent partiel** : gros paragraphes, tableaux, liens complexes ou mise en page fine peuvent se dégrader. Les **tableaux Markdown** ne sont en général **pas** gérés comme sur GitHub.

**Pratiques courantes :** coller par sections et corriger à la main ; ou passer par une **conversion Markdown → HTML** (ex. Pandoc) puis coller le HTML si ton flux le permet ; ou utiliser un outil / extension dédié « Markdown → Medium ». Le **sous-titre** du haut de ce fichier vaut mieux dans le **sous-titre / kicker** natif de Medium que comme ligne `**…**` si tu veux un rendu propre.
