# Shared sentence review collection

The GitHub Pages interface is `site/annotate/`; this directory mirrors the independently deployed Sites backend source at commit `5f936d7a5495b82d0b82daf9de26e5f06617c2a2`. Backend changes require a separate Sites publication; GitHub Pages deployment does not deploy the API.

Production: https://sotu-annotation-collection.tirogerl.chatgpt.site
Owner dashboard: /admin (ChatGPT owner sign-in).

The service is publicly reachable for reviewers, enabled with owner approval. Reviewers submit anonymously using unguessable assignment tokens. Owner APIs still require trusted Sites owner identity or the server-held admin secret. Private links scope access to one assignment; tokens are stored only as SHA-256 hashes. A link grants access to its holder and should be shared privately.

D1 stores assignments, labels, optional notes, revisions and timestamps. `data/study.json` freezes the 1,000 blind sentences and sample hash, without metadata, partitions or predictions. The schema is managed by committed Drizzle migrations; Sites applies hosted migrations during publication. Never alter schema inside request handlers. QA round `qa` is for synthetic service checks and is omitted from study exports.

Runtime variables: `FRONTEND_URL` (main site `/annotate/` URL); secrets `OWNER_EMAIL`, `ADMIN_SECRET`. Configure these with Sites environment tools; never commit values. Sites supplies verified identity headers. `LOCAL_PREVIEW=true` permits loopback origins only for local checks. Never enable it in production.

Test the API core without network or dependencies on Node 22.13+ (SQLite built in):

```bash
node --test tests/collection.test.mjs
```

Install from the committed lockfile, then build using the Sites skill's build helper. Follow the supplied starter README for local preview and applying the local migration. Use dummy local credentials. The production source repository is registered by `.openai/hosting.json`; retain that project ID and obtain a fresh short-lived write credential. Publish with the Sites source helper and exact saved source commit/archive. The skill describes credential handling and deployment; do not push GitHub source and assume the API updated.

After deployment, verify `/api/health`, create a `qa` assignment, save/resume/retry an answer, confirm a reviewer cannot call owner exports, and confirm A/B exports contain no QA labels. Do not use A/B assignments for synthetic checks. Export both rounds periodically as JSONL backups and import with `python -m sotu.import_reviews` as described in the annotation protocol. Individual answer changes are revision-checked, but the service does not retain an immutable history of every edit. Preserve exports and reviewer code assignments for research provenance.
