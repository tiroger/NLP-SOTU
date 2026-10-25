"use client";
import { useEffect, useState } from "react";
type Assignment = {id:string; reader:string;title:string;total:number;labeled:number;created_at:string};
export default function Dashboard() {
  const [assignments,setAssignments] = useState<Assignment[]>([]);
  const [reader,setReader] = useState("a"), [count,setCount] = useState(50), [title,setTitle] = useState("");
  const [busy,setBusy] = useState(false), [error,setError] = useState(""), [invite,setInvite] = useState(""), [copied,setCopied] = useState(false);
  const load = async () => {const r=await fetch("/api/admin/assignments");const data=await r.json();if(!r.ok)throw new Error(data.error);setAssignments(data.assignments);};
  useEffect(()=>{load().catch(e=>setError(e.message));},[]);
  async function create(e:React.FormEvent) {
    e.preventDefault();setBusy(true);setError("");setInvite("");setCopied(false);
    try {const r=await fetch("/api/admin/assignments",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reader,count,title})});const data=await r.json();if(!r.ok)throw new Error(data.error);setInvite(data.review_url);await load();}
    catch(e:any){setError(e.message);}finally{setBusy(false);}
  }
  return <main><p className="eyebrow">State of the Union · Human annotation</p><h1>Independent readings.<br/>One shared collection.</h1><p>Create a personal link for each reviewer. Their work saves to this collection and stays separate from other reviewers’ answers. Use a different person for each round on the same sentence.</p>
    <div className="panel"><h2>Create a reviewer link</h2><form className="form" onSubmit={create}>
      <label>Reviewer code<input required maxLength={80} placeholder="e.g. Reader 01" value={title} onChange={e=>setTitle(e.target.value)}/></label>
      <label>Independent round<select value={reader} onChange={e=>setReader(e.target.value)}><option value="a">Round A</option><option value="b">Round B</option></select></label>
      <label>Sentences<input type="number" min={1} max={1000} required value={count} onChange={e=>setCount(Number(e.target.value))}/></label>
      <button className="primary" disabled={busy}>{busy?"Creating…":"Create link"}</button></form>
      <p>Sentences are allocated without overlap within each round. Fifty is a manageable first batch. Reviewer codes can be pseudonyms.</p>
      {invite&&<div className="notice" role="status"><strong>Copy this personal link now.</strong><p>The access link is shown once. Share it with only the intended reviewer.</p><p><a href={invite} target="_blank" rel="noreferrer">{invite}</a></p><button onClick={async()=>{try{await navigator.clipboard.writeText(invite);setCopied(true);}catch{setError("Select and copy the link manually.");}}}>{copied?"Copied":"Copy link"}</button></div>}
      {error&&<p role="alert" className="error">{error}</p>}</div>
    <div className="panel"><h2>Collection progress</h2><div className="actions">{["a","b"].map(r=><a className="button" key={r} href={`/api/admin/export?reader=${r}`}>Export round {r.toUpperCase()} labels</a>)}<button onClick={()=>load().catch(e=>setError(e.message))}>Refresh</button></div>
      <p>Exports include all 1,000 study items per round, preserving blank labels for incomplete assignments. They can be imported into the existing analysis pipeline.</p>
      <div className="scroll"><table><thead><tr><th>Reviewer</th><th>Round</th><th>Submitted</th><th>Created</th></tr></thead><tbody>{assignments.map(a=><tr key={a.id}><td>{a.title}</td><td>{a.reader.toUpperCase()}</td><td>{a.labeled} / {a.total}</td><td>{new Date(a.created_at).toLocaleDateString()}</td></tr>)}</tbody></table></div>{!assignments.length&&<p>No assignments yet. Create the first reviewer link above.</p>}</div>
    <p><a href="https://tiroger.github.io/NLP-SOTU/annotate/">Annotation page</a> · <a href="https://github.com/tiroger/NLP-SOTU/blob/main/docs/annotation.md">Study protocol</a></p></main>;
}
