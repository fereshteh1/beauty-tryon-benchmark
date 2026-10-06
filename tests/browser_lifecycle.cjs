// Synthetic DOM fixtures; no real user image, browser session or GPU inference.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const {spawnSync} = require('child_process');

const extraction = spawnSync('python3', ['-c', String.raw`
import ast,json,sys
tree=ast.parse(open(sys.argv[1]).read())
scripts={}
for fn in tree.body:
    if isinstance(fn,ast.FunctionDef) and fn.name in {'pick_image','show_transient'}:
        candidates=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        scripts[fn.name]=max(candidates,key=len)
print(json.dumps(scripts))
`, path.join(__dirname,'../engine/photo_preview.py')], {encoding:'utf8'});
assert.equal(extraction.status,0,extraction.stderr);
const scripts = JSON.parse(extraction.stdout);
const picker = scripts.pick_image.replace('LABEL',JSON.stringify('fixture'));
const preview = scripts.show_transient.replace('PAYLOAD',JSON.stringify(Buffer.from('fixture').toString('base64')));

function browser(failImage=false) {
  const nodes=[],timers=[],revoked=[];
  const document={body:{append(...items){nodes.push(...items);}},createElement(tag){
    const node={tag,removed:false,style:{},files:[],append(...items){this.children=items;},
      remove(){this.removed=true;},removeAttribute(name){delete this[name];}};
    if(tag==='img') Object.defineProperty(node,'src',{set(value){this.url=value;
      queueMicrotask(()=>failImage?this.onerror?.():this.onload?.());}});
    return node;
  }};
  const window={addEventListener(){},removeEventListener(){}};
  const context=vm.createContext({document,window,
    URL:{createObjectURL(){return 'blob:memory-only';},revokeObjectURL(url){revoked.push(url);}},
    Blob,Uint8Array,atob,setTimeout(fn,ms){timers.push({fn,ms});return timers.length;},clearTimeout(){},
    FileReader:class {readAsDataURL(file){this.result='data:image/png;base64,VEVTVA==';this.onload();}abort(){}}
  });
  return {context,nodes,timers,revoked,window};
}

(async()=>{
  for(const action of ['success','cancel','oversize','timeout']) {
    const b=browser(); const result=vm.runInContext(picker,b.context);
    const [,input,cancel]=b.nodes[0].children;
    if(action==='success'){input.files=[{size:4}];input.onchange();}
    if(action==='cancel')cancel.onclick();
    if(action==='oversize'){input.files=[{size:26*1024*1024}];input.onchange();}
    if(action==='timeout')b.timers[0].fn();
    const response=await result;
    assert(b.nodes[0].removed);assert.equal(input.value,'');
    if(action==='success')assert.equal(response.data,'VEVTVA==');else assert(response.error);
  }
  for(const action of ['close','expiry','replacement','image_failure']) {
    const b=browser(action==='image_failure');
    assert.equal(await vm.runInContext(preview,b.context),action!=='image_failure');
    if(action==='image_failure'){assert(b.nodes[0].removed);continue;}
    const [,image,close]=b.nodes[0].children;
    assert.equal(image.url,'blob:memory-only');assert.equal(b.timers[0].ms,300000);
    if(action==='close')close.onclick();
    if(action==='expiry')b.timers[0].fn();
    if(action==='replacement') {
      await vm.runInContext(preview,b.context);
      assert(b.nodes[0].removed);assert(!b.nodes[1].removed);
      b.window.__beautyDisposePreview();
    }
    assert(b.nodes[0].removed);assert.equal(b.window.__beautyDisposePreview,undefined);
    assert(b.revoked.length>=1);
  }
  console.log('PASS: four picker paths and four preview lifecycle paths.');
})().catch(error=>{console.error(error);process.exitCode=1;});
