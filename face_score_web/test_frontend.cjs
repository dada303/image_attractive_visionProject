const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes=new Map(),jobs=[],predictions=[],urls=new Map();let count=0;
const el=id=>{if(!nodes.has(id))nodes.set(id,{hidden:false,disabled:false,value:'',textContent:'',files:[],handlers:{},selectedOptions:[{textContent:'EfficientNet-B0'}],videoWidth:640,videoHeight:480,addEventListener(e,f){this.handlers[e]=f;},removeAttribute(k){delete this[k];},replaceChildren(){},add(){},close(){this.handlers.close?.();}});return nodes.get(id);};
const camera={type:'image/png',size:100};
const ctx=vm.createContext({console,AbortController,setTimeout,Image:class{naturalWidth=400;naturalHeight=400;async decode(){}},Option:class{},
 document:{getElementById:el,addEventListener(){},body:{append(){}},createElement(){return {getContext(){return {drawImage(){}};},toBlob(fn){fn(camera);},click(){},remove(){}};}},window:{addEventListener(){}},navigator:{},
 URL:{createObjectURL(b){const u='blob:'+ ++count;urls.set(u,b);return u;},revokeObjectURL(u){urls.delete(u);}},
 fetch:async()=>({ok:true,json:async()=>({models:[{id:'efficientnet_b0',label:'EfficientNet-B0',ready:true}]})}),
 cropImage:(blob,name,signal)=>new Promise((resolve,reject)=>jobs.push({blob,name,signal,resolve,reject})),
 predictImage:async(...args)=>{predictions.push(args);return {score:3.25,model:'EfficientNet-B0'};}
});
vm.runInContext(fs.readFileSync(__dirname+'/frontend/js/app.js','utf8').replace(/^import .*;\r?\n/,''),ctx);
const tick=()=>new Promise(r=>setImmediate(r));
const finish=(i,tag)=>{const blob={type:'image/png',size:80,tag};jobs[i].resolve({blob,token:'signed-'+tag,faceCount:1});return blob;};
(async()=>{
 await tick();const original={type:'image/jpeg',size:100};
 let p=ctx.selectPhoto(original,'a.jpg');assert.equal(el('analyze').disabled,true);
 const crop=finish(0,'first');await p;assert.equal(predictions.length,0);assert.equal(urls.get(el('photo').src),crop);
 await el('analyze').handlers.click();assert.equal(predictions[0][0],crop);assert.equal(predictions[0][2],'signed-first');
 const a=ctx.selectPhoto(original,'A.jpg');assert.equal(el('result').hidden,true);assert.equal(el('photo').src,undefined);
 const b=ctx.selectPhoto(original,'B.jpg');const bBlob=finish(2,'B');await b;finish(1,'A');await a;assert.equal(urls.get(el('photo').src),bBlob);
 el('capture').handlers.click();assert.equal(jobs[3].blob,camera);finish(3,'camera');await tick();assert.equal(predictions.length,1);await el('analyze').handlers.click();assert.equal(predictions.length,2);
 p=ctx.selectPhoto(original,'no-face.jpg');jobs[4].reject(new Error('얼굴을 인식하지 못했습니다.'));await p;assert.equal(el('analyze').disabled,true);assert.equal(el('photo').src,undefined);assert.equal(predictions.length,2);
 p=ctx.selectPhoto(original,'model-change.jpg');el('model-select').handlers.change();const last=finish(5,'last');await p;assert.equal(urls.get(el('photo').src),last);
 await ctx.selectPhoto({type:'text/plain',size:10},'bad.txt');assert.equal(el('analyze').disabled,true);assert.equal(el('photo').src,undefined);
 console.log('PASS: click-only inference, exact preview Blob, reset, A/B race, camera, no-face, model switch, invalid input');
})().catch(e=>{console.error(e);process.exitCode=1;});
