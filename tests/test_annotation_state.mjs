import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validateSession,countSaved,nextPendingIndex,makeAnswer,safeDrafts} from '../site/annotate/state.mjs';
const items=[{key:'one',text:'Sentence',sentiment:'neutral',notes:'',revision:1},{key:'two',text:'Other',sentiment:null,notes:'',revision:0}];
test('progress and pending navigation wrap without skipping unsubmitted items',()=>{assert.equal(countSaved(items),1);assert.equal(nextPendingIndex(items,1),1);assert.equal(nextPendingIndex([items[0]]),-1);});
test('payload preserves draft revision for concurrent-edit detection',()=>{const draft={sentiment:'ambiguous',notes:'context',revision:0};assert.deepEqual(makeAnswer(items[0],draft),{key:'one',...draft});assert.throws(()=>makeAnswer(items[1]));});
test('reject corrupt sessions and filter corrupt drafts',()=>{const data={assignment:{id:'study',title:'Review'},dataset_hash:'hash',items};assert.equal(validateSession(data),data);assert.throws(()=>validateSession({...data,items:[items[0],items[0]]}));assert.deepEqual(safeDrafts({one:{sentiment:'negative',notes:'',revision:0},two:{sentiment:'bad'},missing:{sentiment:'neutral',notes:'',revision:0}},items),{one:{sentiment:'negative',notes:'',revision:0}});});
