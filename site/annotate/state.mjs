export const LABELS = ["negative","neutral","positive","ambiguous"];
export function validateSession(data) {
  if (!data?.assignment?.id || typeof data.assignment.title !== "string" || typeof data.dataset_hash !== "string" || !Array.isArray(data.items) || !data.items.length) throw new Error("The review assignment is incomplete.");
  const seen = new Set();
  for (const item of data.items) {
    if (typeof item.key !== "string" || seen.has(item.key) || typeof item.text !== "string" || ![null,...LABELS].includes(item.sentiment) || typeof item.notes !== "string" || !Number.isSafeInteger(item.revision) || item.revision < 0) throw new Error("The review assignment has invalid sentences.");
    seen.add(item.key);
  }
  return data;
}
export const countSaved = items => items.filter(item=>LABELS.includes(item.sentiment)).length;
export function nextPendingIndex(items, from=-1) {
  for (let offset=1;offset<=items.length;offset++) {const i=(from+offset)%items.length;if(items[i].sentiment===null)return i;}
  return -1;
}
export function makeAnswer(item,draft) {
  const value = draft || {sentiment:item.sentiment,notes:item.notes,revision:item.revision};
  if (!LABELS.includes(value.sentiment) || typeof value.notes !== "string" || value.notes.length>2000 || !Number.isSafeInteger(value.revision) || value.revision<0) throw new Error("Choose a label before saving.");
  return {key:item.key,sentiment:value.sentiment,notes:value.notes,revision:value.revision};
}
export function safeDrafts(value,items) {
  if (!value || typeof value!=="object" || Array.isArray(value)) return {};
  const keys = new Set(items.map(item=>item.key)), out={};
  for (const [key,draft] of Object.entries(value)) {
    if(keys.has(key) && draft && [null,...LABELS].includes(draft.sentiment) && typeof draft.notes==="string" && draft.notes.length<=2000 && Number.isSafeInteger(draft.revision) && draft.revision>=0)out[key]={sentiment:draft.sentiment,notes:draft.notes,revision:draft.revision};
  }
  return out;
}
