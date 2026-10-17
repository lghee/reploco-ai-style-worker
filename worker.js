/** REPLOCO AI Style Finder Worker · Cloudflare Vectorize + Google Gemini.
 * No persistent customer-photo storage. Only product embeddings in Vectorize.
 */
const encoder = new TextEncoder();
const TYPES = new Set(['tops','bottoms','bags','shoes','watches','accessories','other']);
function respond(data,status=200){return new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'}})}
function safeError(err){return {ok:false,error:err?.publicMessage||'AI worker error'}}
function failure(message,status=400){const e=new Error(message);e.status=status;e.publicMessage=message;throw e}
async function authenticate(request,env,raw){
  if(!env.REPLOCO_SHARED_SECRET || !env.GEMINI_API_KEY || !env.VECTORIZE)failure('Worker not configured',503);
  const ts=request.headers.get('x-reploco-time');const signature=request.headers.get('x-reploco-signature')||'';
  if(!/^\d{10}$/.test(ts||'')||Math.abs(Date.now()/1000-Number(ts))>120||!/^[a-f0-9]{64}$/i.test(signature))failure('Unauthorized',401);
  const key=await crypto.subtle.importKey('raw',encoder.encode(env.REPLOCO_SHARED_SECRET),{name:'HMAC',hash:'SHA-256'},false,['sign']);
  const signed=new Uint8Array(await crypto.subtle.sign('HMAC',key,encoder.encode(ts+'.'+raw)));
  let diff=0;for(let i=0;i<32;i++){diff|=signed[i]^parseInt(signature.slice(i*2,i*2+2),16)}
  if(diff!==0)failure('Unauthorized',401);
}
async function gemini(env,model,suffix,body){
  const resp=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:${suffix}`,{
    method:'POST',headers:{'Content-Type':'application/json','x-goog-api-key':env.GEMINI_API_KEY},body:JSON.stringify(body)
  });
  const result=await resp.json().catch(()=>null);
  if(!resp.ok){if(resp.status===429)failure('Gemini quota reached',429);failure('Gemini model request failed',502)}
  if(!result || typeof result!=='object')failure('Empty model response',502);
  return result;
}
function validImage(input){
  if(!input || !['image/png','image/jpeg'].includes(input.mime)|| typeof input.image!=='string' ||input.image.length>5*1024*1024 || !/^[A-Za-z0-9+/=]+$/.test(input.image)) failure('Invalid image',400);
  return {inline_data:{mime_type:input.mime,data:input.image}};
}
async function embedding(env,parts){
  const result=await gemini(env,'gemini-embedding-2','embedContent',{
    model:'models/gemini-embedding-2',content:{parts},outputDimensionality:768
  });
  const values=result.embedding?.values;
  if(!Array.isArray(values)||values.length!==768||!values.every(Number.isFinite))failure('Invalid embedding dimensions',502);
  return values;
}
function parseJsonFromText(value){
  const raw=String(value||'').replace(/^```(?:json)?\s*/i,'').replace(/```\s*$/,'').trim();
  try{return JSON.parse(raw)}catch{return null}
}
async function vision(env,image,prompt){
  const model=env.VISION_MODEL||'gemini-2.5-flash';
  const r=await gemini(env,model,'generateContent',{
    contents:[{role:'user',parts:[{text:prompt},image]}],
    generationConfig:{responseMimeType:'application/json',temperature:0.1,maxOutputTokens:1500}
  });
  const text=(r.candidates?.[0]?.content?.parts||[]).map(p=>p.text||'').join('');
  return parseJsonFromText(text);
}
function normalizeKind(k){return TYPES.has(k)?k:'other'}
function cleanBox(box){
  if(!Array.isArray(box)||box.length!==4||!box.every(x=>typeof x==='number'&&Number.isFinite(x)))return null;
  let [y1,x1,y2,x2]=box.map(v=>Math.max(0,Math.min(1000,Math.round(v))));
  if(y2-y1<20||x2-x1<20||y2<=y1||x2<=x1)return null;
  return [y1,x1,y2,x2];
}
async function detect(env,req){
  const image=validImage(req);
  const prompt=`Find only clearly visible fashion items in this image: tops/outerwear, bottoms, bags, shoes, watches, accessories. Never identify people, celebrities, faces, gender identity, or brand without clear visible proof. Return a JSON object {"items":[{"kind":"tops|bottoms|bags|shoes|watches|accessories|other","label":"short Korean garment/material/color description","box":[ymin,xmin,ymax,xmax]}]}. box coordinates are 0-1000 normalized, each item cropped tightly. At most 8 distinct clearly visible items. Return {"items":[]} if none.`;
  const result=await vision(env,image,prompt);
  const items=[];
  for(const a of Array.isArray(result?.items)?result.items:[]){
    const box=cleanBox(a.box);if(!box)continue;
    items.push({kind:normalizeKind(a.kind),label:String(a.label||'스타일').slice(0,70),box});if(items.length>=8)break;
  }
  return {ok:true,items};
}
async function findMatches(env,vec,kind,topK=30){
  const opts={topK:Math.max(1,Math.min(50,Number(topK)||30)),returnMetadata:'all'};
  if(TYPES.has(kind)&&kind!=='other')opts.filter={kind};
  let found=await env.VECTORIZE.query(vec,opts);
  // Missing taxonomy classifications should not make otherwise relevant products invisible.
  if(opts.filter && (!found.matches || !found.matches.length))found=await env.VECTORIZE.query(vec,{topK:opts.topK,returnMetadata:'all'});
  const seen=new Set();const matches=[];
  for(const m of found.matches||[]){
    const id=Number(m.metadata?.product_id||0);
    if(!Number.isSafeInteger(id)||id<=0||seen.has(id))continue;
    seen.add(id);matches.push({product_id:id,score:typeof m.score==='number'?m.score:0});
  }
  return matches;
}
async function search(env,req){
  let parts=[];
  if(req.image) parts=[validImage(req)];
  else if(typeof req.query==='string'&&req.query.trim().length>=2&&req.query.length<=160)parts=[{text:req.query.trim()}];
  else failure('Image or text query required',400);
  const vec=await embedding(env,parts);
  return {ok:true,matches:await findMatches(env,vec,normalizeKind(req.kind),req.topK)};
}
function productKey(id){if(!Number.isSafeInteger(id)||id<=0)failure('Invalid product',400);return 'reploco-p-'+id}
function imageHostAllowed(env,url){
  const list=String(env.ALLOWED_IMAGE_HOSTS||'reploco.com,img.reploco.com').split(',').map(x=>x.trim().toLowerCase()).filter(Boolean);
  const u=new URL(url);return u.protocol==='https:' && !u.username && !u.password && !u.port && list.some(host=>u.hostname.toLowerCase()===host);
}
async function index(env,req){
  const id=Number(req.product_id);const key=productKey(id);
  if(typeof req.image_url!=='string'||req.image_url.length>2000||!imageHostAllowed(env,req.image_url))failure('Product image host not allowlisted',400);
  const response=await fetch(req.image_url,{redirect:'manual',headers:{'Accept':'image/jpeg,image/png'}});
  if(!response.ok || response.status>=300)failure('Product image unavailable',502);
  const mime=(response.headers.get('content-type')||'').split(';')[0].trim().toLowerCase();
  if(!['image/jpeg','image/png'].includes(mime))failure('Only JPG or PNG supported for indexing',400);
  if(Number(response.headers.get('content-length')||0)>4*1024*1024)failure('Product image too large',400);
  const data=new Uint8Array(await response.arrayBuffer());if(!data.length||data.byteLength>4*1024*1024)failure('Product image too large',400);
  let binary='';for(let i=0;i<data.length;i+=32768)binary+=String.fromCharCode(...data.subarray(i,i+32768));
  const vector=await embedding(env,[{inline_data:{mime_type:mime,data:btoa(binary)}}]);
  // Idempotent: upsert replaces prior product vector instead of creating duplicates.
  await env.VECTORIZE.upsert([{id:key,values:vector,metadata:{product_id:id,kind:normalizeKind(req.kind),model:'gemini-embedding-2'}}]);
  return {ok:true,product_id:id};
}
async function remove(env,req){const id=Number(req.product_id);await env.VECTORIZE.deleteByIds([productKey(id)]);return {ok:true}}
async function similar(env,req){
  const id=Number(req.product_id);const found=await env.VECTORIZE.getByIds([productKey(id)]);
  const v=Array.isArray(found)?found[0]:null;if(!v?.values) return {ok:true,matches:[]};
  const all=await findMatches(env,Array.from(v.values),normalizeKind(req.kind),Math.min(50,(req.topK||30)+1));
  return {ok:true,matches:all.filter(m=>m.product_id!==id)};
}
async function outfit(env,req){
  const label=String(req.label||'').slice(0,120).trim();
  const kind=normalizeKind(req.kind);if(label.length<2)failure('Style description required',400);
  // Deterministic style-based cross-category search, not fabricated product recommendations.
  const related={tops:['bottoms','shoes','bags'],bottoms:['tops','shoes','bags'],bags:['tops','shoes','bottoms'],shoes:['tops','bottoms','bags'],watches:['tops','bags','shoes'],accessories:['tops','bags','shoes'],other:['tops','bottoms','bags']}[kind];
  const groups=[];
  for(const k of related){
    const text=`한국 패션 쇼핑몰에서 ${label} 스타일과 조화로운 ${ {tops:'상의',bottoms:'바지 및 하의',bags:'가방',shoes:'신발'}[k] } 디자인, 컬러, 실루엣`;
    const vec=await embedding(env,[{text}]);
    groups.push({kind:k,matches:(await findMatches(env,vec,k,15)).slice(0,10)});
  }
  return {ok:true,groups};
}
export default {
  async fetch(request,env){
    try{
      if(request.method!=='POST')return respond({ok:false,error:'POST required'},405);
      const url=new URL(request.url);const route=url.pathname.replace(/^\/+|\/+$/g,'');
      if(!['health','detect','search','index','delete','similar','outfit'].includes(route))return respond({ok:false,error:'Route not found'},404);
      const raw=await request.text();if(raw.length>6*1024*1024)failure('Payload too large',413);
      await authenticate(request,env,raw);
      const body=JSON.parse(raw||'{}');
      let result;
      switch(route){
        case 'health':result={ok:true,model:'gemini-embedding-2',dimensions:768};break;
        case 'detect':result=await detect(env,body);break;
        case 'search':result=await search(env,body);break;
        case 'index':result=await index(env,body);break;
        case 'delete':result=await remove(env,body);break;
        case 'similar':result=await similar(env,body);break;
        case 'outfit':result=await outfit(env,body);break;
      }
      return respond(result);
    }catch(e){return respond(safeError(e),e.status||502)}
  }
};
