const LABELS = new Set(["negative", "neutral", "positive", "ambiguous"]);
const MAX_BODY = 12000;
export const json = (data, status = 200) => new Response(JSON.stringify(data), {status, headers:{"Content-Type":"application/json; charset=utf-8", "Cache-Control":"no-store"}});
export async function digest(value) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)))].map(x=>x.toString(16).padStart(2,"0")).join("");
}
function token() { return Array.from(crypto.getRandomValues(new Uint8Array(32)), x=>x.toString(16).padStart(2,"0")).join(""); }
function bearer(request) { return request.headers.get("authorization")?.match(/^Bearer ([A-Za-z0-9_-]{30,160})$/)?.[1] || ""; }
export async function isAdmin(request, env) {
  const credential = bearer(request);
  if (credential && env.ADMIN_SECRET && await digest(credential) === await digest(env.ADMIN_SECRET)) return true;
  const email = request.headers.get("oai-authenticated-user-email");
  const userId = request.headers.get("oai-authenticated-user-id");
  if (!email || !userId || !env.OWNER_EMAIL || email.toLowerCase() !== env.OWNER_EMAIL.toLowerCase()) return false;
  // Cookie-authenticated writes originate from the same-origin owner dashboard.
  return request.method === "GET" || request.headers.get("origin") === new URL(request.url).origin;
}
async function body(request) {
  if (!request.headers.get("content-type")?.startsWith("application/json")) throw new Error("json_body");
  if (Number(request.headers.get("content-length")) > MAX_BODY) throw new Error("body_size");
  const value = await request.text();
  if (new TextEncoder().encode(value).length > MAX_BODY) throw new Error("body_size");
  const parsed = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("json_body");
  return parsed;
}
export async function createAssignment(db, study, {reader, count, title=""}, env) {
  if (!["a","b","qa"].includes(reader) || !Number.isInteger(count) || count < 1 || count > 1000 || typeof title !== "string" || title.length > 80) return json({error:"Choose reader A or B, a batch size from 1 to 1000, and a short assignment name."},400);
  if (title.trim() && reader !== "qa") {
    const other = await db.prepare("SELECT id FROM review_assignments WHERE LOWER(title) = LOWER(?) AND reader != ? AND reader IN ('a','b') LIMIT 1").bind(title.trim(),reader).first();
    if (other) return json({error:"This reviewer code already has the other round. Assign independent rounds to different people."},409);
  }
  for (let attempt=0; attempt<3; attempt++) {
    const occupied = await db.prepare("SELECT item_key FROM review_items WHERE reader = ?").bind(reader).all();
    const assigned = new Set(occupied.results.map(x=>x.item_key));
    const chosen = study.items.filter(x=>!assigned.has(x.key)).slice(0,count);
    if (chosen.length < count) return json({error:"Not enough unassigned sentences in this reader round.",available:chosen.length},409);
    const id = crypto.randomUUID(), secret = token(), hash = await digest(secret), now = new Date().toISOString();
    try {
      const statements = [db.prepare("INSERT INTO review_assignments (id, token_hash, reader, title, created_at) VALUES (?, ?, ?, ?, ?)").bind(id,hash,reader,title.trim() || `Reader ${reader.toUpperCase()}`,now),
        ...chosen.map((item,i)=>db.prepare("INSERT INTO review_items (reader, item_key, assignment_id, ordinal) VALUES (?, ?, ?, ?)").bind(reader,item.key,id,i))];
      // One atomic D1 batch; the primary key prevents concurrent double assignment.
      await db.batch(statements);
      return json({id,reader,title:title.trim() || `Reader ${reader.toUpperCase()}`,count:chosen.length,
        review_url: env.FRONTEND_URL + "#invite=" + secret},201);
    } catch (error) {
      if (!String(error).includes("UNIQUE") || attempt === 2) throw error;
    }
  }
}
export async function saveAnswer(db, assignment, value) {
  if (!value || typeof value.key !== "string" || !LABELS.has(value.sentiment) || typeof value.notes !== "string" || value.notes.length > 2000 || !Number.isSafeInteger(value.revision) || value.revision < 0) return json({error:"Choose a valid label and keep notes below 2,000 characters."},400);
  const updated = await db.prepare("UPDATE review_items SET sentiment = ?, notes = ?, revision = revision + 1, updated_at = ? WHERE assignment_id = ? AND item_key = ? AND revision = ? RETURNING item_key, sentiment, notes, revision, updated_at")
    .bind(value.sentiment,value.notes,new Date().toISOString(),assignment.id,value.key,value.revision).first();
  if (updated) return json({answer:updated});
  const current = await db.prepare("SELECT item_key, sentiment, notes, revision, updated_at FROM review_items WHERE assignment_id = ? AND item_key = ?").bind(assignment.id,value.key).first();
  if (!current) return json({error:"This sentence is not part of your assignment."},403);
  if (current.sentiment === value.sentiment && current.notes === value.notes) return json({answer:current});
  return json({error:"This answer changed in another tab or device. Load the latest answer before editing.",current},409);
}
export async function collection(request, {db, study, env}) {
  const url = new URL(request.url);
  const origin = request.headers.get("origin");
  const frontend = new URL(env.FRONTEND_URL).origin;
  const allowed = !origin || origin === frontend || origin === url.origin || (env.LOCAL_PREVIEW === "true" && /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin));
  let response;
  try {
    if (!allowed) response = json({error:"Origin not allowed."},403);
    else if (request.method === "OPTIONS") response = new Response(null,{status:204});
    else if (url.pathname === "/api/health" && request.method === "GET") {
      const ready = await db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'review_items'").first();
      response = ready ? json({ok:true,dataset_hash:study.hash,items:study.items.length}) : json({error:"Collection database is not ready."},503);
    } else if (url.pathname.startsWith("/api/admin/")) {
      if (!await isAdmin(request, env)) response = json({error:"Owner sign-in required."},401);
      else if (url.pathname === "/api/admin/assignments" && request.method === "POST") response = await createAssignment(db,study,await body(request),env);
      else if (url.pathname === "/api/admin/assignments" && request.method === "GET") {
        const rows = await db.prepare("SELECT a.id, a.reader, a.title, a.created_at, COUNT(i.item_key) AS total, SUM(CASE WHEN i.sentiment IS NOT NULL THEN 1 ELSE 0 END) AS labeled FROM review_assignments a JOIN review_items i ON i.assignment_id = a.id WHERE a.reader IN ('a','b') GROUP BY a.id ORDER BY a.created_at DESC").all();
        response = json({assignments:rows.results,totals:{a:1000,b:1000},dataset_hash:study.hash});
      } else if (url.pathname === "/api/admin/export" && request.method === "GET") {
        const reader = url.searchParams.get("reader");
        if (!["a","b"].includes(reader)) response = json({error:"Choose reader a or b."},400);
        else {
          const rows = await db.prepare("SELECT item_key, sentiment, notes FROM review_items WHERE reader = ?").bind(reader).all();
          const labels = new Map(rows.results.map(r=>[r.item_key,r]));
          const content = study.items.map(item=>JSON.stringify({key:item.key,text:item.text,sentiment:labels.get(item.key)?.sentiment || null, notes:labels.get(item.key)?.notes || ""})).join("\n") + "\n";
          response = new Response(content,{headers:{"Content-Type":"application/x-ndjson; charset=utf-8","Content-Disposition":`attachment; filename="sentiment_reader_${reader}.jsonl"`,"Cache-Control":"no-store"}});
        }
      } else response = json({error:"Not found."},404);
    } else {
      const credential = bearer(request);
      const assignment = credential ? await db.prepare("SELECT id, reader, title FROM review_assignments WHERE token_hash = ?").bind(await digest(credential)).first() : null;
      if (!assignment) response = json({error:"This review link is missing or invalid. Ask the study owner for your link."},401);
      else if (url.pathname === "/api/session" && request.method === "GET") {
        const rows = await db.prepare("SELECT item_key, sentiment, notes, revision, updated_at FROM review_items WHERE assignment_id = ? ORDER BY ordinal").bind(assignment.id).all();
        const byKey = new Map(study.items.map(x=>[x.key,x.text]));
        response = json({assignment:{id:assignment.id,title:assignment.title},dataset_hash:study.hash,
          items:rows.results.map(r=>({key:r.item_key,text:byKey.get(r.item_key),sentiment:r.sentiment,notes:r.notes,revision:r.revision}))});
      } else if (url.pathname === "/api/answer" && request.method === "POST") response = await saveAnswer(db,assignment,await body(request));
      else response = json({error:"Not found."},404);
    }
  } catch (error) {
    if (error instanceof SyntaxError || ["json_body","body_size"].includes(error.message)) response = json({error:"Invalid or oversized JSON request."},400);
    else { console.error("Collection operation failed", error.message); response = json({error:"Could not save right now. Your local draft is safe; please retry."},503); }
  }
  const headers = new Headers(response.headers);
  headers.set("Vary","Origin"); headers.set("X-Content-Type-Options","nosniff");
  if (allowed && origin) headers.set("Access-Control-Allow-Origin",origin);
  headers.set("Access-Control-Allow-Methods","GET, POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers","Authorization, Content-Type");
  headers.set("Cache-Control","no-store");
  return new Response(response.body,{status:response.status,headers});
}
