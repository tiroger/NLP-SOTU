import {LABELS,validateSession,countSaved,nextPendingIndex,makeAnswer,safeDrafts} from "./state.mjs";
const $=id=>document.getElementById(id);
let config,credential,session,index=0,drafts={},storageKey="",saving=false,reviewing=false,conflict=null,storageAvailable=true;
const labelSelect=$("sentiment");
function message(text,error=false){$("message").textContent=text;$("message").classList.toggle("error",error);$("message").hidden=!text;}
function store(){try{localStorage.setItem(storageKey,JSON.stringify(drafts));return storageAvailable=true;}catch{return storageAvailable=false;}}
function currentDraft(){return drafts[session.items[index].key];}
function updateDraft(){
  if(!session||saving)return;
  const item=session.items[index],prior=currentDraft();
  drafts[item.key]={sentiment:labelSelect.value || null,notes:$("notes").value,revision:prior?.revision ?? item.revision};
  const persisted=store();$("save-state").textContent=persisted?"Draft kept on this device. Save & next submits your answer.":"Draft storage is unavailable. Keep this page open until your answer is submitted.";
  $("answer-state").textContent="Unsubmitted edits";$("answer-state").classList.remove("saved");
  $("save").disabled=!drafts[item.key].sentiment || saving || Boolean(conflict);
}
async function api(path,options={}){
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
  try{
    const response=await fetch(config.api+path,{...options,credentials:"omit",signal:controller.signal,headers:{...(options.body?{"Content-Type":"application/json"}:{}),Authorization:"Bearer "+credential,...options.headers}});
    if(!response.headers.get("content-type")?.includes("application/json"))throw new Error("The shared collection is not accessible yet. Ask the study owner to enable reviewer access.");
    const result=await response.json();
    if(!response.ok){const error=new Error(result.error||"The collection could not complete the request.");error.status=response.status;error.current=result.current;throw error;}
    return result;
  }finally{clearTimeout(timeout);}
}
function render(){
  const item=session.items[index],draft=currentDraft(),value=draft||item;
  $("progress-text").textContent=`${countSaved(session.items)} of ${session.items.length} saved`;
  $("progress").max=session.items.length;$("progress").value=countSaved(session.items);
  const done=countSaved(session.items)===session.items.length&&!reviewing&&!Object.keys(drafts).length;
  $("complete").hidden=!done;$("workspace").hidden=done;
  $("position").textContent=`Sentence ${index+1} of ${session.items.length}`;$("sentence").textContent=item.text;
  labelSelect.value=value.sentiment || "";labelSelect.disabled=saving;
  $("notes").value=value.notes;$("notes").disabled=saving;
  $("answer-state").textContent=draft?"Unsubmitted edits":item.sentiment?"Saved to collection":"Not submitted";
  $("answer-state").classList.toggle("saved",!draft&&Boolean(item.sentiment));
  $("save-state").textContent=draft?(storageAvailable?"Draft kept on this device. Save & next submits your answer.":"Keep this page open until your answer is submitted."):item.sentiment?"This answer is saved. You can revise it and save again.":"Choose a label, then save it.";
  $("save").disabled=!value.sentiment||saving||Boolean(conflict);
  $("save").textContent=saving?"Saving…":index===session.items.length-1?"Save answer":"Save & next →";
  $("previous").disabled=index===0||saving;$("skip").disabled=saving;$("jump").disabled=saving;
  $("jump").replaceChildren(...session.items.map((x,i)=>{const option=document.createElement("option");option.value=String(i);option.textContent=`Sentence ${i+1}${x.sentiment?" · saved":" · not submitted"}`;return option;}));$("jump").value=String(index);
  $("conflict").hidden=!conflict;
}
function navigate(to){if(saving)return;conflict=null;index=to;render();$("sentence").focus({preventScroll:true});}
async function start(token){
  credential=token;$("welcome").hidden=true;$("study").hidden=true;$("loading").hidden=false;message("");
  try{
    session=validateSession(await api("/api/session"));
    if(session.dataset_hash!==config.dataset_hash)throw new Error("This review uses a different study version. Ask the owner for an updated link.");
    storageKey=`sotu-review:${session.assignment.id}:${session.dataset_hash}`;
    try{drafts=safeDrafts(JSON.parse(localStorage.getItem(storageKey)||"{}"),session.items);}catch{drafts={};}
    index=session.items.findIndex(item=>drafts[item.key]);if(index<0)index=Math.max(0,nextPendingIndex(session.items));$("assignment-title").textContent=session.assignment.title;
    $("study").hidden=false;render();
  }catch(error){message(error.name==="AbortError"?"The collection took too long to respond. Please try opening your link again.":error.message,true);$("welcome").hidden=false;}
  finally{$("loading").hidden=true;}
}
$("annotation-form").addEventListener("submit",async event=>{
  event.preventDefault();if(saving||conflict)return;
  const item=session.items[index];let value;
  try{value=makeAnswer(item,currentDraft());}catch(error){message(error.message,true);return;}
  saving=true;render();message("");
  try{
    const data=await api("/api/answer",{method:"POST",body:JSON.stringify(value)});
    item.sentiment=data.answer.sentiment;item.notes=data.answer.notes;item.revision=data.answer.revision;
    delete drafts[item.key];store();
    if(index<session.items.length-1)index++;
    else {const next=nextPendingIndex(session.items,index);if(next>=0)index=next;else reviewing=false;}
  }catch(error){
    if(error.status===409&&error.current){conflict=error.current;$("conflict-text").textContent=`This answer changed in another tab or device. The saved label is “${conflict.sentiment}”. ${conflict.notes?"Saved notes: "+conflict.notes:"Your draft has been kept."}`;}
    else message(error.name==="AbortError"?"The collection took too long to respond. Your draft is kept on this device; retry to confirm it was saved.":error.message,true);
  }finally{saving=false;render();$("sentence").focus({preventScroll:true});}
});
labelSelect.addEventListener("change",updateDraft);$("notes").addEventListener("input",updateDraft);
$("previous").addEventListener("click",()=>navigate(Math.max(0,index-1)));
$("skip").addEventListener("click",()=>{const next=nextPendingIndex(session.items,index);navigate(next>=0?next:(index+1)%session.items.length);});
$("jump").addEventListener("change",event=>navigate(Number(event.target.value)));
$("review-answers").addEventListener("click",()=>{reviewing=true;navigate(0);});
$("use-saved").addEventListener("click",()=>{const item=session.items[index];Object.assign(item,{sentiment:conflict.sentiment,notes:conflict.notes,revision:conflict.revision});delete drafts[item.key];store();conflict=null;render();});
$("keep-draft").addEventListener("click",()=>{const item=session.items[index],draft=currentDraft();if(draft)draft.revision=conflict.revision;Object.assign(item,{sentiment:conflict.sentiment,notes:conflict.notes,revision:conflict.revision});store();conflict=null;render();});
$("open-link").addEventListener("submit",event=>{event.preventDefault();try{const url=new URL($("review-link").value);const invite=new URLSearchParams(url.hash.slice(1)).get("invite");if(!invite||!/^[A-Za-z0-9_-]{30,160}$/.test(invite))throw new Error("Paste the complete personal review link you received.");history.replaceState(null,"","#invite="+invite);start(invite);}catch(error){message(error.message,true);}});
document.addEventListener("keydown",event=>{
  if(!session||saving||$("workspace").hidden||event.metaKey||event.ctrlKey||event.altKey)return;
  const target=event.target,tag=target.tagName;
  if(tag==="TEXTAREA"||tag==="SELECT"||tag==="INPUT"||tag==="BUTTON")return;
  const n=Number(event.key);
  if(Number.isInteger(n)&&n>=1&&n<=4){event.preventDefault();labelSelect.value=LABELS[n-1];updateDraft();}
  else if(event.key==="Enter"&&!$("save").disabled){event.preventDefault();$("annotation-form").requestSubmit();}
});
try{const theme=localStorage.getItem("theme");if(theme)document.documentElement.dataset.theme=theme;}catch{}
$("theme").addEventListener("click",()=>{const dark=document.documentElement.dataset.theme?document.documentElement.dataset.theme==="dark":matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.dataset.theme=dark?"light":"dark";try{localStorage.setItem("theme",dark?"light":"dark");}catch{}});
try{
  const response=await fetch("config.json");if(!response.ok)throw new Error("The review page configuration could not be loaded.");config=await response.json();$("manage").href=config.manage;
  const invite=new URLSearchParams(location.hash.slice(1)).get("invite");
  if(invite&&/^[A-Za-z0-9_-]{30,160}$/.test(invite))await start(invite);
  else{$("loading").hidden=true;$("welcome").hidden=false;}
}catch(error){$("loading").hidden=true;message(error.message,true);}
