import {test} from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {collection} from '../lib/collection.mjs';
function fixture(){
 const sql=new DatabaseSync(':memory:');sql.exec(readFileSync(new URL('../drizzle/0000_damp_talon.sql',import.meta.url),'utf8'));
 const db={prepare(query){return {bind(...args){this.args=args;return this;},async first(){return sql.prepare(query).get(...(this.args||[]))||null;},async all(){return {results:sql.prepare(query).all(...(this.args||[]))};},run(){return sql.prepare(query).run(...(this.args||[]));}};},async batch(statements){sql.exec('BEGIN');try{const out=statements.map(s=>s.run());sql.exec('COMMIT');return out;}catch(e){sql.exec('ROLLBACK');throw e;}}};
 const study={hash:'frozen-study',items:Array.from({length:8},(_,i)=>({key:`s-${i}`,text:`Sentence ${i}`}))};
 const env={ADMIN_SECRET:'owner-secret-'.repeat(4),OWNER_EMAIL:'owner@example.org',FRONTEND_URL:'https://tiroger.github.io/NLP-SOTU/annotate/'};
 async function call(path,{token=env.ADMIN_SECRET,method='GET',body,origin='https://tiroger.github.io'}={}){return collection(new Request('https://api.example.org'+path,{method,headers:{...(token?{authorization:'Bearer '+token}:{}),origin,'content-type':'application/json'},...(body?{body:JSON.stringify(body)}:{})}),{db,study,env});}
 async function assign(reader,title,count=2){const r=await call('/api/admin/assignments',{method:'POST',body:{reader,title,count}});assert.equal(r.status,201);const data=await r.json();return {...data,token:new URL(data.review_url).hash.split('=')[1]};}
 return {call,assign,sql,study};
}
test('independent assignments, scoped access, central persistence and safe retries',async()=>{
 const {call,assign,sql}=fixture();
 assert.equal((await call('/api/admin/export?reader=a',{token:null})).status,401);
 assert.equal((await call('/api/session',{token:'x'.repeat(64)})).status,401);
 const a=await assign('a','reader-one'),other=await assign('a','reader-two');
 assert.equal((await call('/api/admin/assignments',{method:'POST',body:{reader:'b',title:'READER-ONE',count:1}})).status,409);
 const session=await (await call('/api/session',{token:a.token})).json();
 assert.equal(session.items.length,2);assert.deepEqual(Object.keys(session.items[0]).sort(),['key','notes','revision','sentiment','text']);
 assert.equal(session.reader,undefined);assert.equal(session.items[0].sentiment,null);
 const otherSession=await (await call('/api/session',{token:other.token})).json();assert.notEqual(otherSession.items[0].key,session.items[0].key);
 const answer={key:session.items[0].key,sentiment:'neutral',notes:'human note',revision:0};
 const save=await call('/api/answer',{token:a.token,method:'POST',body:answer});assert.equal(save.status,200);assert.equal((await save.json()).answer.revision,1);
 assert.equal((await call('/api/answer',{token:a.token,method:'POST',body:answer})).status,200);
 assert.equal((await call('/api/answer',{token:a.token,method:'POST',body:{...answer,sentiment:'negative'}})).status,409);
 assert.equal((await call('/api/answer',{token:other.token,method:'POST',body:answer})).status,403);
 assert.equal((await call('/api/answer',{token:a.token,method:'POST',body:{...answer,sentiment:'bad'}})).status,400);
 assert.equal((await call('/api/admin/export?reader=a',{token:a.token})).status,401);
 const resumed=await (await call('/api/session',{token:a.token})).json();assert.equal(resumed.items[0].sentiment,'neutral');
 const exported=(await (await call('/api/admin/export?reader=a')).text()).trim().split('\n').map(JSON.parse);assert.equal(exported.length,8);assert.equal(exported[0].notes,'human note');assert.equal(exported[2].sentiment,null);
 assert.ok(sql.prepare('SELECT token_hash FROM review_assignments LIMIT 1').get().token_hash!==a.token);
});
test('QA excluded, CORS restricted, preflight and storage health',async()=>{
 const {call,assign}=fixture();const qa=await assign('qa','test');
 await call('/api/answer',{token:qa.token,method:'POST',body:{key:'s-0',sentiment:'positive',notes:'synthetic',revision:0}});
 assert.deepEqual((await (await call('/api/admin/assignments')).json()).assignments,[]);
 assert.ok(!(await (await call('/api/admin/export?reader=a')).text()).includes('synthetic'));
 assert.equal((await call('/api/health')).status,200);
 assert.equal((await call('/api/health',{origin:'https://untrusted.example'})).status,403);
 const preflight=await call('/api/session',{method:'OPTIONS',token:null});assert.equal(preflight.status,204);assert.equal(preflight.headers.get('access-control-allow-origin'),'https://tiroger.github.io');
});
