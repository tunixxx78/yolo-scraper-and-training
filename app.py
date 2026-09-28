import base64,csv,hashlib,io,json,re,time
from pathlib import Path
from urllib.parse import urlparse
import imagehash,requests,streamlit as st
from PIL import Image

GOOGLE_API='https://www.googleapis.com/customsearch/v1'
OPENVERSE_API='https://api.openverse.org/v1/images/'
COMMONS_API='https://commons.wikimedia.org/w/api.php'
SERPAPI='https://serpapi.com/search.json'
UA='YoloDatasetCollector/4.3 (local YOLO dataset research tool; Python requests)'

def safe_name(s): return re.sub(r'[^A-Za-z0-9._-]+','_',s.strip()).strip('_') or 'dataset'
def norm(s): return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def ollama_tags(base):
 r=requests.get(base.rstrip('/')+'/api/tags',timeout=10);r.raise_for_status();return [m.get('name','') for m in r.json().get('models',[])]
def parse_json(s,fallback):
 s=(s or '').strip();s=re.sub(r'^```(?:json)?\s*|\s*```$','',s,flags=re.I)
 try:return json.loads(s)
 except Exception:
  m=re.search(r'\{.*\}',s,re.S)
  if m:
   try:return json.loads(m.group(0))
   except:pass
 return fallback

def chat_json(base,model,prompt,images=None,timeout=180,schema=None):
 msg={'role':'user','content':prompt}
 if images:msg['images']=images
 payload={'model':model,'messages':[msg],'stream':False,'format':schema if schema else 'json','options':{'temperature':0.0}}
 r=requests.post(base.rstrip('/')+'/api/chat',json=payload,timeout=timeout);r.raise_for_status()
 return parse_json(r.json().get('message',{}).get('content',''),{})

def analyze_target(base,model,equipment,category,required,excluded):
 p=f'''You are disambiguating an exact physical equipment model for an image dataset.
Exact model: {equipment}
User category: {category}
Required context: {required}
Excluded meanings: {excluded}
Return JSON only:
{{"canonical_name":"...","category":"...","family":"...","aliases":["..."],"positive_context":["..."],"negative_context":["..."]}}
Never shorten an alias to an ambiguous generic word. Preserve manufacturer names, model numbers, variant suffixes, and other discriminating identifiers in aliases. Generate aliases only when they refer to the same requested physical equipment model.'''
 d=chat_json(base,model,p)
 return {'canonical_name':d.get('canonical_name') or equipment,'category':category or d.get('category') or 'military equipment','family':d.get('family') or '',
         'aliases':[x for x in d.get('aliases',[]) if isinstance(x,str) and len(x.strip())>2][:12],
         'positive_context':list(dict.fromkeys(([x.strip() for x in required.split(',') if x.strip()]+[x for x in d.get('positive_context',[]) if isinstance(x,str)])))[:16],
         'negative_context':list(dict.fromkeys(([x.strip() for x in excluded.split(',') if x.strip()]+[x for x in d.get('negative_context',[]) if isinstance(x,str)])))[:20]}

def search_identity_context(equipment,info):
 """Return strong generic physical-context terms for ambiguous/model-like names."""
 equipment=(equipment or '').strip()
 cat=(info.get('category') or '').strip()
 fam=(info.get('family') or '').strip()
 aliases=[str(x).strip() for x in info.get('aliases',[]) if str(x).strip()]
 pos=[str(x).strip() for x in info.get('positive_context',[]) if str(x).strip()]
 toks=re.findall(r'[A-Za-z0-9]+',equipment)
 ambiguous=(len(toks)<=3 or len(equipment)<=14 or bool(re.search(r'\d',equipment)))
 ctx=[]
 for x in [fam,cat]+aliases+pos:
  if x and x.lower()!=equipment.lower() and x.lower() not in [c.lower() for c in ctx]:ctx.append(x)
 return ambiguous,ctx[:10]

def enforce_search_context(queries,equipment,info,quoted=True):
 """Never search an ambiguous identifier naked; bind it to family/category/manufacturer context."""
 ambiguous,ctx=search_identity_context(equipment,info)
 if not ambiguous:return list(dict.fromkeys(queries))
 out=[]
 # Strong context-first seeds.
 for c in ctx[:4]:
  out.append(f'"{equipment}" "{c}"' if quoted else f'{equipment} {c}')
 for q in queries:
  q=(q or '').strip()
  if not q:continue
  # Any ambiguous query must contain exact user target plus at least one strong context term.
  if equipment.lower() not in q.lower():
   q=(f'"{equipment}" ' if quoted else f'{equipment} ')+q
  if ctx and not any(c.lower() in q.lower() for c in ctx):
   q=f'"{equipment}" "{ctx[0]}"' if quoted else f'{equipment} {ctx[0]}'
  out.append(q)
 return list(dict.fromkeys(out))

def build_queries(equipment,info):
 names=[equipment,info['canonical_name']]+info['aliases']; names=list(dict.fromkeys(x.strip() for x in names if x and x.strip()))
 # Reject dangerous alias collapse: every alias must retain either a digit from exact model or >=2 meaningful exact tokens.
 exact_tokens=[t for t in re.findall(r'[A-Za-z0-9]+',equipment) if len(t)>=2]; digits=''.join(re.findall(r'\d+',equipment))
 good=[]
 for n in names:
  nt=norm(n)
  hits=sum(1 for t in exact_tokens if norm(t) in nt)
  if n==equipment or n==info['canonical_name'] or (digits and digits in ''.join(re.findall(r'\d+',n))) or hits>=2: good.append(n)
 names=list(dict.fromkeys(good))[:8]
 cat=info['category']; pos=info['positive_context'] or [cat]
 views=['photo','side view','front view','rear view','three quarter view','field','exercise','museum','army']
 neg=' '.join('-'+re.sub(r'\s+','-',x.strip()) for x in info['negative_context'][:10] if x.strip() and ' ' not in x.strip())
 qs=[]
 for n in names:
  contexts=list(dict.fromkeys([cat]+pos[:5]))
  for c in contexts:
   qs.append(f'"{n}" "{c}" {neg}'.strip())
  for v in views: qs.append(f'"{n}" "{cat}" {v} {neg}'.strip())
 return enforce_search_context(list(dict.fromkeys(qs))[:100],equipment,info,quoted=True)[:100]

def build_backend_queries(equipment,info):
 # Each backend gets syntax it actually understands. Negative terms are applied locally.
 names=[equipment,info['canonical_name']]+info['aliases']
 names=list(dict.fromkeys(x.strip() for x in names if x and x.strip()))
 exact_tokens=[t for t in re.findall(r'[A-Za-z0-9]+',equipment) if len(t)>=2]
 digits=''.join(re.findall(r'\d+',equipment))
 good=[]
 for n in names:
  nt=norm(n);hits=sum(1 for t in exact_tokens if norm(t) in nt)
  if n==equipment or n==info['canonical_name'] or (digits and digits in ''.join(re.findall(r'\d+',n))) or hits>=2:good.append(n)
 names=list(dict.fromkeys(good))[:8]
 cat=info['category'];pos=info['positive_context'] or [cat]
 views=['photo','side view','front view','rear view','three quarter view','field','exercise','museum','army']
 neg=' '.join('-'+re.sub(r'\s+','-',x.strip()) for x in info['negative_context'][:10] if x.strip() and ' ' not in x.strip())
 google=[];openverse=[];commons=[]
 for n in names:
  contexts=list(dict.fromkeys([cat]+pos[:5]))
  for c in contexts:
   google.append(f'"{n}" "{c}" {neg}'.strip())
   openverse.append(f'{n} {c}'.strip())
  for v in views:
   google.append(f'"{n}" "{cat}" {v} {neg}'.strip())
   if v in ('photo','side view','front view','field','exercise','museum'):
    openverse.append(f'{n} {cat} {v}'.strip())
  # Commons search works best with short literal equipment names.
  commons += [n, f'{n} {cat}', f'{n} tank' if 'tank' in norm(cat) else f'{n} {cat}']
 google=enforce_search_context(list(dict.fromkeys(google)),equipment,info,quoted=True)
 openverse=enforce_search_context(list(dict.fromkeys(openverse)),equipment,info,quoted=False)
 commons=enforce_search_context(list(dict.fromkeys(commons)),equipment,info,quoted=False)
 return {'google':google[:100],'openverse':openverse[:60],'commons':commons[:40]}

def result(item):
 base={'creator':'','license':'unknown','license_url':'','source':'','title':'','snippet':'','page_url':'','query':'','width':'','height':''};base.update(item);return base

def search_google(q,key,cx,pages=3):
 if not key or not cx:return []
 out=[]
 for start in range(1,min(91,pages*10)+1,10):
  r=requests.get(GOOGLE_API,params={'key':key,'cx':cx,'q':q,'searchType':'image','num':10,'start':start,'safe':'off','imgType':'photo'},timeout=30)
  if r.status_code!=200: raise RuntimeError(f'Google API {r.status_code}: {r.text[:300]}')
  for x in r.json().get('items',[]):
   im=x.get('image',{})
   out.append(result({'provider':'google','title':x.get('title',''),'snippet':x.get('snippet',''),'image_url':x.get('link',''),'page_url':im.get('contextLink',''),'source':x.get('displayLink',''),'query':q,'width':im.get('width',''),'height':im.get('height','')}))
 return out

def search_serpapi(q,key,pages=1,budget=None):
 if not key:return []
 out=[]
 cache_dir=Path('.search_cache')/'serpapi';cache_dir.mkdir(parents=True,exist_ok=True)
 for ijn in range(max(1,pages)):
  cache_key=hashlib.sha256((q+'|'+str(ijn)).encode()).hexdigest()
  cache_file=cache_dir/(cache_key+'.json')
  d=None
  if cache_file.exists():
   try:
    d=json.loads(cache_file.read_text(encoding='utf-8'))
    if budget is not None:budget['cache_hits']=budget.get('cache_hits',0)+1
   except Exception:d=None
  if d is None:
   if budget is not None and budget.get('remaining',0)<=0:break
   r=requests.get(SERPAPI,params={'engine':'google_images','q':q,'api_key':key,'ijn':ijn,'safe':'off','device':'desktop','tbs':'itp:photos','no_cache':'false'},headers={'User-Agent':UA,'Accept':'application/json'},timeout=60)
   if budget is not None:
    budget['remaining']=max(0,budget.get('remaining',0)-1);budget['used']=budget.get('used',0)+1
   if r.status_code!=200:raise RuntimeError(f'SerpApi HTTP {r.status_code}: {(r.text or "")[:500]}')
   d=r.json()
   if d.get('error'):raise RuntimeError('SerpApi error: '+str(d.get('error'))[:500])
   try:cache_file.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
   except Exception:pass
  rows=d.get('images_results',[]) or []
  for x in rows:
   u=x.get('original') or x.get('thumbnail')
   if not u:continue
   out.append(result({'provider':'serpapi','title':x.get('title',''),'snippet':x.get('source','') or '','image_url':u,'page_url':x.get('link',''),'source':x.get('source',''),'query':q,'width':x.get('original_width',''),'height':x.get('original_height','')}))
  if not rows:break
 return out

def http_debug(r):
 return {'status_code':r.status_code,'final_url':r.url,'content_type':r.headers.get('content-type',''),'rate_limit_remaining':r.headers.get('x-ratelimit-remaining',''),'retry_after':r.headers.get('retry-after',''),'response_excerpt':(r.text or '')[:500]}

def search_openverse(q,debug=False):
 # Anonymous Openverse requests currently allow at most 20 results per page.
 # Fetch several pages politely instead of requesting page_size=50, and back off
 # when the service asks us to slow down.
 out=[]
 debug_pages=[]
 max_pages=3  # up to 60 raw results per query without authentication
 headers={'User-Agent':UA,'Api-User-Agent':UA,'Accept':'application/json'}

 for page in range(1,max_pages+1):
  r=None
  for attempt in range(3):
   r=requests.get(OPENVERSE_API,params={'q':q,'page_size':20,'page':page},headers=headers,timeout=30)
   page_dbg=http_debug(r);page_dbg.update({'page':page,'attempt':attempt+1})
   debug_pages.append(page_dbg)
   if r.status_code==429:
    # Respect Retry-After when present. Keep the wait bounded for an interactive app.
    try:wait=float(r.headers.get('retry-after','') or (2 ** attempt))
    except Exception:wait=float(2 ** attempt)
    time.sleep(min(max(wait,1.0),15.0))
    continue
   break

  if r is None:
   break
  if r.status_code==429:
   # Do not hammer Cloudflare/Openverse. Keep any results already collected.
   break
  if r.status_code!=200:
   raise RuntimeError(f"Openverse HTTP {r.status_code}: {(r.text or '')[:500]}")

  try:d=r.json()
  except Exception as e:
   raise RuntimeError(f"Openverse non-JSON response: {(r.text or '')[:500]}") from e

  rows=d.get('results',[])
  for x in rows:
   u=x.get('url') or x.get('thumbnail')
   if u:
    out.append(result({'provider':'openverse','title':x.get('title',''),'snippet':x.get('description','') or '','image_url':u,'page_url':x.get('foreign_landing_url',''),'creator':x.get('creator',''),'license':x.get('license',''),'license_url':x.get('license_url',''),'source':x.get('source',''),'query':q,'width':x.get('width',''),'height':x.get('height','')}))

  # Stop if this was the last page.
  if len(rows)<20 or not d.get('next'):
   break

  # Anonymous API is rate limited. A short pause prevents bursts across pages.
  time.sleep(0.75)

 dbg={
  'backend':'Openverse',
  'query':q,
  'ok':True,
  'parsed_items':len(out),
  'pages_requested':len(debug_pages),
  'last_request':debug_pages[-1] if debug_pages else {},
 }
 return (out,dbg) if debug else out

def search_commons(q,debug=False):
 p={'action':'query','format':'json','formatversion':2,'generator':'search','gsrsearch':q,'gsrnamespace':6,'gsrlimit':50,'maxlag':5,'utf8':1,'prop':'imageinfo','iiprop':'url|mime|size|extmetadata','iiurlwidth':1600}
 r=requests.get(COMMONS_API,params=p,headers={'User-Agent':UA,'Api-User-Agent':UA,'Accept':'application/json'},timeout=30)
 dbg=http_debug(r);dbg.update({'backend':'Wikimedia Commons','query':q})
 if r.status_code!=200:
  raise RuntimeError(
   f"Wikimedia HTTP {r.status_code}. URL={r.url}. "
   f"Content-Type={r.headers.get('content-type','')}. "
   f"Response={(r.text or '')[:700]}"
  )
 try:d=r.json()
 except Exception as e:raise RuntimeError(f"Wikimedia non-JSON response: {(r.text or '')[:500]}") from e
 if 'error' in d:raise RuntimeError('Wikimedia API error: '+json.dumps(d['error'],ensure_ascii=False)[:800])
 pages=d.get('query',{}).get('pages',[])
 if isinstance(pages,dict):pages=list(pages.values())
 dbg['api_page_count']=len(pages);out=[]
 for pg in pages:
  ii=(pg.get('imageinfo') or [{}])[0];u=ii.get('thumburl') or ii.get('url');e=ii.get('extmetadata',{});val=lambda k:re.sub('<[^>]+>','',str(e.get(k,{}).get('value','')))
  if u:out.append(result({'provider':'commons','title':pg.get('title',''),'image_url':u,'page_url':ii.get('descriptionurl',''),'creator':val('Artist'),'license':val('LicenseShortName'),'license_url':val('LicenseUrl'),'source':val('Credit'),'snippet':val('ImageDescription'),'query':q,'width':ii.get('width',''),'height':ii.get('height','')}))
 return (out,dbg) if debug else out

def test_backend(name,q,key='',cx=''):
 t0=time.time()
 try:
  if name=='Openverse':items,dbg=search_openverse(q,True)
  elif name=='Wikimedia Commons':items,dbg=search_commons(q,True)
  elif name=='Google':
   if not key or not cx:return {'backend':'Google','ok':False,'error':'API key and Search Engine ID (cx) are required.'}
   items=search_google(q,key,cx,1);dbg={'backend':'Google','query':q,'status_code':200,'api_result_count':len(items)}
  elif name=='SerpApi Google Images':
   if not key:return {'backend':'SerpApi Google Images','ok':False,'error':'SerpApi API key is required.'}
   items=search_serpapi(q,key,1);dbg={'backend':'SerpApi Google Images','query':q,'status_code':200,'api_result_count':len(items)}
  dbg.update({'ok':True,'parsed_items':len(items),'elapsed_ms':round((time.time()-t0)*1000),'sample_titles':[x.get('title','') for x in items[:5]]})
  return dbg
 except Exception as e:return {'backend':name,'query':q,'ok':False,'error':f'{type(e).__name__}: {e}','elapsed_ms':round((time.time()-t0)*1000)}

def metadata_score(item,equipment,info):
 """Soft, generic metadata ranking. It never requires the exact model name."""
 text=norm(' '.join([item.get('title',''),item.get('snippet',''),item.get('page_url',''),item.get('source','')]))
 exact=norm(equipment);canonical=norm(info.get('canonical_name',''))
 aliases=[norm(x) for x in info.get('aliases',[]) if x]
 category=norm(info.get('category',''));family=norm(info.get('family',''))
 positive=[norm(x) for x in info.get('positive_context',[]) if x]
 negative=[norm(x) for x in info.get('negative_context',[]) if x]
 score=0
 if exact and exact in text:score+=30
 if canonical and canonical in text:score+=24
 score+=max([18 if a and a in text else 0 for a in aliases] or [0])
 if family and family in text:score+=10
 if category and category in text:score+=8
 score+=sum(3 for x in positive if x and x in text)
 score-=sum(10 for x in negative if x and x in text)
 return score

def metadata_candidate(item,equipment,info):
 """Only discard strong negative metadata matches; ambiguous metadata is kept for vision."""
 text=norm(' '.join([item.get('title',''),item.get('snippet',''),item.get('page_url',''),item.get('source','')]))
 negatives=[norm(x) for x in info.get('negative_context',[]) if x]
 positives=[norm(equipment),norm(info.get('canonical_name','')),norm(info.get('family','')),norm(info.get('category',''))]+[norm(x) for x in info.get('positive_context',[]) if x]
 neg=sum(1 for x in negatives if x and x in text);pos=sum(1 for x in positives if x and x in text)
 return not (neg>=2 and pos==0)

def search_all(backend_queries,settings,status,max_candidates):
 out=[];seen=set();stats={}
 plans=[]
 if settings['google']:plans.append(('Google',backend_queries['google'],lambda q:search_google(q,settings['gkey'],settings['gcx'],settings['pages'])))
 if settings.get('serpapi'):
  sb={'remaining':settings.get('serp_budget',3),'used':0,'cache_hits':0};plans.append(('SerpApi Google Images',backend_queries['google'],lambda q:search_serpapi(q,settings['serpkey'],settings['serppages'],sb)))
 if settings['ov']:plans.append(('Openverse',backend_queries['openverse'],search_openverse))
 if settings['wc']:plans.append(('Wikimedia Commons',backend_queries['commons'],search_commons))
 for backend,queries,fn in plans:
  returned=0;unique_added=0
  status.write(f'### {backend}')
  for i,q in enumerate(queries):
   status.write(f'{backend} {i+1}/{len(queries)}: {q}')
   try:batch=fn(q)
   except Exception as e:
    status.error(f'{backend} query failed: {type(e).__name__}: {e}');continue
   returned+=len(batch)
   # Avoid anonymous Openverse burst throttling between separate search queries.
   if backend=='Openverse': time.sleep(1.0)
   for x in batch:
    u=x.get('image_url','')
    if u and u not in seen:
     seen.add(u);out.append(x);unique_added+=1
   if len(out)>=max_candidates:break
  # Count strict metadata passes now, so UI shows where results disappear.
  passed=sum(1 for x in out if x.get('provider','').lower() in backend.lower() and metadata_candidate(x,settings['equipment'],settings['info']))
  stats[backend]={'returned':returned,'unique_added':unique_added,'metadata_candidates':passed}
  if backend=='SerpApi Google Images':stats[backend].update({'new_requests':sb.get('used',0),'local_cache_hits':sb.get('cache_hits',0),'budget_remaining':sb.get('remaining',0)})
  status.write(f'{backend}: returned {returned}, unique added {unique_added}, metadata candidates {passed}')
  if len(out)>=max_candidates:break
 return out,stats

def download(item,max_bytes=40_000_000):
 # Download politely. Wikimedia explicitly prefers scaled thumbnails over originals
 # for automated clients; search_commons() therefore stores a ~1600 px thumburl.
 headers={'User-Agent':UA,'Api-User-Agent':UA,'Referer':item.get('page_url',''),'Accept':'image/avif,image/webp,image/*,*/*;q=0.8'}
 last_error=None
 for attempt in range(4):
  try:
   r=requests.get(item['image_url'],headers=headers,timeout=35,stream=True)
   if r.status_code==429:
    try: wait=float(r.headers.get('Retry-After','') or (5*(2**attempt)))
    except Exception: wait=float(5*(2**attempt))
    r.close();time.sleep(min(max(wait,5.0),60.0));continue
   r.raise_for_status();b=bytearray()
   for c in r.iter_content(262144):
    if not c:continue
    b.extend(c)
    if len(b)>max_bytes:
     r.close();raise ValueError('too large')
   r.close();im=Image.open(io.BytesIO(b));im.load()
   # Small pacing delay for Wikimedia image delivery to avoid bursty automated downloads.
   if item.get('provider')=='commons':time.sleep(0.8)
   return im.convert('RGB'),bytes(b)
  except requests.HTTPError as e:
   last_error=e
   if getattr(e.response,'status_code',None) in (429,503):
    time.sleep(min(5*(2**attempt),60));continue
   raise
 if last_error:raise last_error
 raise RuntimeError('Image download failed after retries')

def validate(base,model,equipment,info,im,item=None):
 cp=im.copy();cp.thumbnail((1536,1536));buf=io.BytesIO();cp.save(buf,'JPEG',quality=88);enc=base64.b64encode(buf.getvalue()).decode()
 item=item or {};meta=" | ".join(str(item.get(k,'') or '') for k in ('title','snippet','page_url','source'))
 schema={'type':'object','properties':{'is_photograph':{'type':'boolean'},'target_visible':{'type':'boolean'},'target_prominence':{'type':'number','minimum':0,'maximum':1},'target_occlusion':{'type':'number','minimum':0,'maximum':1},'contains_target_family':{'type':'boolean'},'family':{'type':'string'},'family_confidence':{'type':'number','minimum':0,'maximum':1},'exact_variant':{'type':'string'},'exact_variant_confidence':{'type':'number','minimum':0,'maximum':1},'variant_status':{'type':'string','enum':['confirmed','uncertain','wrong']},'usable_for_training':{'type':'boolean'},'view':{'type':'string'},'reason':{'type':'string'}},'required':['is_photograph','target_visible','target_prominence','target_occlusion','contains_target_family','family','family_confidence','exact_variant','exact_variant_confidence','variant_status','usable_for_training','view','reason']}
 prompt=("Verify this candidate for an object-detection dataset.\nTARGET OBJECT: "+equipment+"\nCATEGORY: "+str(info['category'])+"\nFAMILY: "+str(info['family'])+"\nSearch metadata (secondary evidence only): "+meta[:1800]+"\nThe TARGET OBJECT itself must be visibly present in the pixels. Metadata can support identity but can NEVER prove target presence. People, crew, buildings, scenery, signs, or other context are not the target unless explicitly requested. Set target_visible=false if the requested object is absent, tiny/background-only, almost fully hidden, or merely mentioned in text/metadata. target_prominence: 0=absent/tiny, 1=clear/substantial/easy to localize. target_occlusion: 0=fully visible, 1=fully hidden. Do not require printed model text. If the correct family is visibly present and reliable metadata explicitly names the exact requested model, metadata may support exact identity only AFTER target_visible=true. Use uncertain for a visible correct family whose exact variant cannot be distinguished; wrong for unrelated or positively different equipment.")
 return chat_json(base,model,prompt,[enc],schema=schema)

def visual_prefilter(base,model,equipment,info,item,timeout=120):
 """Fail-closed pixel-only family/category gate. Exact model/variant recognition is deferred."""
 u=item.get('thumbnail_url') or item.get('image_url')
 if not u:return {'target_visible':False,'target_prominence':0.0,'reason':'no image URL'}
 try:
  r=get_with_backoff(u,provider=item.get('provider',''),timeout=45)
  im=Image.open(io.BytesIO(r.content)).convert('RGB')
  im.thumbnail((768,768));buf=io.BytesIO();im.save(buf,'JPEG',quality=75)
  enc=base64.b64encode(buf.getvalue()).decode()
 except Exception as e:return {'target_visible':False,'target_prominence':0.0,'reason':f'download failed: {e}'}
 schema={'type':'object','properties':{'target_visible':{'type':'boolean'},'target_prominence':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},'required':['target_visible','target_prominence','reason']}
 category=str(info.get('category','') or '');family=str(info.get('family','') or '')
 physical=family or category or equipment
 prompt=("STRICT PIXEL-ONLY PRESENCE GATE. "
  f"Final requested model/name: {equipment}. Physical family/category to look for: {physical}. "
  "Do NOT decide the exact model or variant here. Inspect image pixels only; ignore title, filename, URL, caption and all metadata. "
  "target_visible=true ONLY when a real physical object matching that family/category is clearly visible. "
  "A shared model name printed on an unrelated consumer product is irrelevant. "
  "FAIL CLOSED: if the image is people-only, an event/meeting, electronics/audio gear, medicine, documents, diagrams, buildings/scenery, or any unrelated product/object, set target_visible=false. "
  "If uncertain whether the requested physical family/category is present, set target_visible=false. "
  "target_prominence: 0=absent/tiny/background-only; 1=clear substantial object easy to annotate.")
 try:
  v=chat_json(base,model,prompt,[enc],timeout=timeout,schema=schema)
  if not isinstance(v,dict) or v.get('target_visible') not in (True,False):
   return {'target_visible':False,'target_prominence':0.0,'reason':'invalid vision response'}
  return v
 except Exception as e:return {'target_visible':False,'target_prominence':0.0,'reason':f'vision failed: {e}'}

def run_visual_prefilter(base,model,equipment,info,candidates,min_prominence,max_checks,status,progress=None):
 """Filter only the highest-ranked candidates so preview/collection does not waste work on obvious context images."""
 ranked=candidates[:max_checks]
 passed=[];rejected=[];cache=st.session_state.setdefault('visual_prefilter_cache',{})
 for i,item in enumerate(ranked,1):
  key=hashlib.sha256((equipment+'|'+str(info.get('category',''))+'|v4.11|'+str(item.get('image_url',''))).encode()).hexdigest()
  if key in cache:v=cache[key]
  else:
   v=visual_prefilter(base,model,equipment,info,item);cache[key]=v
  visible=(isinstance(v,dict) and v.get('target_visible') is True);prom=float(v.get('target_prominence',0) or 0) if isinstance(v,dict) else 0.0
  item=dict(item);item['_prefilter_visible']=visible;item['_prefilter_prominence']=prom;item['_prefilter_reason']=v.get('reason','')
  if visible and prom>=min_prominence:passed.append(item)
  else:rejected.append(item)
  if progress is not None:progress.progress(i/max(1,len(ranked)),text=f'Visual pre-filter {i}/{len(ranked)} | passed {len(passed)} | removed {len(rejected)}')
 # Candidates outside the checked budget remain after the checked/passed set so collection can continue if needed.
 return passed+candidates[max_checks:],rejected,{'checked':len(ranked),'passed':len(passed),'removed':len(rejected)}

def preview_candidates(candidates,equipment,info,limit=30):
 ranked=[x for x in candidates if metadata_candidate(x,equipment,info)]
 ranked=sorted(ranked,key=lambda x:metadata_score(x,equipment,info),reverse=True)
 return ranked[:limit]

def metadata_exact_match(item,equipment):
 """Generic exact-model evidence check. No equipment names/classes are hard-coded."""
 raw=(equipment or '').strip()
 if not raw:return False
 text=" ".join(str(item.get(k,'') or '') for k in ('title','snippet','page_url','source'))
 # Normalize punctuation/spacing while preserving alphanumeric model identifiers.
 target_tokens=re.findall(r'[a-z0-9]+',raw.lower())
 text_tokens=re.findall(r'[a-z0-9]+',text.lower())
 if not target_tokens:return False
 # Require the complete requested token sequence, not a shortened family name.
 n=len(target_tokens)
 return any(text_tokens[i:i+n]==target_tokens for i in range(max(0,len(text_tokens)-n+1)))


def collect(equipment,target,model,base,outdir,settings,progress,status,info,queries,candidates):
 candidates=[x for x in candidates if metadata_candidate(x,equipment,info)];candidates=sorted(candidates,key=lambda x:metadata_score(x,equipment,info),reverse=True)
 root=Path(outdir).resolve()/safe_name(equipment);acc=root/'accepted';review=root/'review'
 for d in (acc,review):d.mkdir(parents=True,exist_ok=True)
 fields=['filename','bucket','equipment','provider','query','title','page_url','image_url','license','license_url','source','sha256','phash','target_visible','target_prominence','target_occlusion','family_confidence','exact_variant','exact_variant_confidence','variant_status','metadata_exact_match','view','reason']
 seen=[];counts={'accepted':0,'review':0,'rejected':0,'downloaded':0,'duplicates':0,'failures':0,'wrong_family':0,'unusable':0,'variant_uncertain':0,'low_confidence':0}
 with (root/'metadata.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for idx,item in enumerate(candidates,1):
   if counts['accepted']>=target:break
   progress.progress(min(1.,counts['accepted']/target),text=f"Accepted {counts['accepted']}/{target} | review {counts['review']} | rejected {counts['rejected']} | checked {idx}/{len(candidates)}")
   try:
    im,raw=download(item);counts['downloaded']+=1;ph=imagehash.phash(im)
    if any(ph-x<=4 for x in seen):counts['duplicates']+=1;continue
    seen.append(ph);v=validate(base,model,equipment,info,im,item)
    fc=float(v.get('family_confidence',0) or 0);vc=float(v.get('exact_variant_confidence',0) or 0)
    visible=(isinstance(v,dict) and v.get('target_visible') is True);prom=float(v.get('target_prominence',0) or 0) if isinstance(v,dict) else 0.0;occ=float(v.get('target_occlusion',0) or 0)
    family=bool(v.get('contains_target_family'));usable=bool(v.get('usable_for_training',True))
    usable=usable and visible and prom>=settings.get('min_prominence',0.35) and occ<=settings.get('max_occlusion',0.80)
    variant=str(v.get('exact_variant','') or '');vstatus=str(v.get('variant_status','uncertain')).lower()
    meta_exact=metadata_exact_match(item,equipment)
    # Accept either a visually confirmed exact variant OR a strong visual family match
    # backed by exact-model search metadata. Visible text inside the image is NOT required.
    visual_exact=(vstatus=='confirmed' and vc>=settings['variant_conf'])
    provenance_exact=(meta_exact and fc>=settings['family_conf'] and vstatus!='wrong')
    if family and usable and fc>=settings['family_conf'] and (visual_exact or provenance_exact):
     bucket='accepted'
    elif family and usable and fc>=settings['review_conf'] and vstatus in ('confirmed','uncertain'):
     bucket='review'
     if vstatus=='uncertain':counts['variant_uncertain']+=1
    else:
     bucket='rejected'
     if not family:counts['wrong_family']+=1
     elif not usable:counts['unusable']+=1
     else:counts['low_confidence']+=1
    counts[bucket]+=1;fn=''
    if bucket in ('accepted','review'):
     fn=f"{bucket}_{counts[bucket]:06d}.jpg";im.save((acc if bucket=='accepted' else review)/fn,'JPEG',quality=95)
    w.writerow({'filename':fn,'bucket':bucket,'equipment':equipment,**{k:item.get(k,'') for k in ['provider','query','title','page_url','image_url','license','license_url','source']},'sha256':hashlib.sha256(raw).hexdigest(),'phash':str(ph),'target_visible':visible,'target_prominence':prom,'target_occlusion':occ,'family_confidence':fc,'exact_variant':variant,'exact_variant_confidence':vc,'variant_status':vstatus,'metadata_exact_match':meta_exact,'view':v.get('view',''),'reason':v.get('reason','')});f.flush()
   except Exception as e:counts['failures']+=1;status.caption(f'Skip: {type(e).__name__}: {e}')
 progress.progress(min(1.,counts['accepted']/target),text=f"Finished: {counts['accepted']}/{target} accepted")
 return root,len(candidates),counts

st.set_page_config(page_title='YOLO Dataset Collector v4.11',layout='wide');st.title('YOLO Dataset Collector v4.11')
st.caption('Backend search → soft metadata ranking → family/category visual pre-filter → exact-model Qwen3-VL verification.')
with st.sidebar:
 st.header('Ollama');base=st.text_input('Ollama URL','http://localhost:11434');model=st.text_input('Vision model','huihui_ai/qwen3-vl-abliterated:8b');outdir=st.text_input('Output directory','./datasets')
 if st.button('Test Ollama'):
  try:
   ms=ollama_tags(base)
   if model not in ms:st.warning('Connected, selected model not installed.');st.code('\n'.join(ms))
   else:
    d=chat_json(base,model,'Return JSON only: {"status":"ready"}');st.success(f'Ollama READY — {model} — inference: {d.get("status","response received")}')
  except Exception as e:st.error(str(e))
 st.header('Search backend')
 use_google=st.checkbox('Google Programmable Image Search',False);gkey=st.text_input('Google API key',type='password');gcx=st.text_input('Google Search Engine ID (cx)');pages=st.slider('Google pages/query',1,9,3)
 use_serpapi=st.checkbox('SerpApi Google Images',False);serpkey=st.text_input('SerpApi API key',type='password');serppages=st.slider('SerpApi pages/query',1,3,1);serp_budget=st.slider('Maximum SerpApi searches per run',1,10,3);st.caption(f'Hard cap: {serp_budget} new SerpApi requests per run. Cached identical searches are reused locally.')
 use_ov=st.checkbox('Openverse fallback',False);use_wc=st.checkbox('Wikimedia Commons fallback',True);mult=st.slider('Candidate multiplier',5,50,15)
 st.header('Validation');family_conf=st.slider('Family confidence',0.,1.,0.75,.05);variant_conf=st.slider('Exact model confidence',0.,1.,0.70,.05);review_conf=st.slider('Review threshold',0.,1.,0.45,.05);min_prominence=st.slider('Minimum target prominence',0.,1.,0.35,.05);max_occlusion=st.slider('Maximum target occlusion',0.,1.,0.80,.05);use_visual_prefilter=st.checkbox('Visual pre-filter before preview',True);prefilter_prominence=st.slider('Preview minimum target prominence',0.,1.,0.50,.05);prefilter_checks=st.slider('Preview images to visually check',10,100,40,10)

c1,c2=st.columns([3,1]);equipment=c1.text_input('Exact equipment/model',placeholder='e.g. exact manufacturer and model');target=c2.number_input('Accepted images wanted',1,5000,100,10)
c3,c4=st.columns(2);category=c3.text_input('Physical category (optional — Qwen can infer it)',value='');required=c4.text_input('Required context (comma-separated, optional)',value='')
excluded=st.text_input('Excluded meanings (comma-separated, optional)',value='toy,model,miniature,game,illustration,drawing,diagram')
st.info('v4.11 uses soft metadata ranking: exact model text is not required before vision. The preview visual pre-filter checks only the inferred family/category; exact variant identification is deferred to final validation. No equipment model or class is hard-coded.')
st.warning('Google Custom Search JSON API requires an existing API key + Programmable Search Engine ID. Google states the JSON API is closed to new customers; existing customers can use it during the transition period. Broad-web images may be copyrighted—verify rights before reuse.')

if 'v41_search' not in st.session_state: st.session_state.v41_search=None

if st.button('Test search backends'):
 st.subheader('Search backend diagnostics')
 tq=(equipment.strip()+' '+category.strip()).strip() or 'equipment model'
 st.caption('Test query: '+tq)
 tests=[]
 if use_ov:tests.append(test_backend('Openverse',tq))
 if use_wc:tests.append(test_backend('Wikimedia Commons',equipment.strip() or tq))
 if use_serpapi:tests.append(test_backend('SerpApi Google Images',tq,serpkey.strip()))
 if use_google:tests.append(test_backend('Google',tq,gkey.strip(),gcx.strip()))
 if not tests:st.error('No search backend selected.')
 for t in tests:
  if t.get('ok'):st.success(f"{t['backend']}: connected; parsed {t.get('parsed_items',0)} result(s); HTTP {t.get('status_code','?')}")
  else:st.error(f"{t['backend']}: {t.get('error','Unknown error')}")
  st.json(t)

if st.button('1. Analyze target & Search preview',type='primary',disabled=not equipment.strip()):
 try:
  if not (use_google or use_serpapi or use_ov or use_wc): st.error('No search backend selected. Enable SerpApi, Google, Openverse, or Wikimedia Commons.');st.stop()
  if use_google and (not gkey.strip() or not gcx.strip()): st.error('Google search is enabled: enter both Google API key and Search Engine ID (cx).');st.stop()
  if use_serpapi and not serpkey.strip(): st.error('SerpApi is enabled: enter your SerpApi API key.');st.stop()
  with st.status('Analyzing and searching…',expanded=True) as s:
   info=analyze_target(base,model,equipment.strip(),category.strip(),required.strip(),excluded.strip());s.write('Target: '+json.dumps(info,ensure_ascii=False))
   backend_queries=build_backend_queries(equipment.strip(),info);s.write(f"Backend queries: Google {len(backend_queries['google'])}, Openverse {len(backend_queries['openverse'])}, Commons {len(backend_queries['commons'])}.")
   settings={'google':use_google,'gkey':gkey.strip(),'gcx':gcx.strip(),'pages':pages,'serpapi':use_serpapi,'serpkey':serpkey.strip(),'serppages':serppages,'serp_budget':serp_budget,'ov':use_ov,'wc':use_wc,'equipment':equipment.strip(),'info':info}
   candidates,stats=search_all(backend_queries,settings,s,max(int(target)*int(mult),100));s.write(f'{len(candidates)} unique raw candidates found.')
   # Rank by metadata first, then visually inspect the best candidates before preview.
   candidates=[x for x in candidates if metadata_candidate(x,equipment.strip(),info)];candidates=sorted(candidates,key=lambda x:metadata_score(x,equipment.strip(),info),reverse=True)
   if use_visual_prefilter and candidates:
    pf=st.progress(0.0)
    candidates,removed,pfstats=run_visual_prefilter(base.strip(),model.strip(),equipment.strip(),info,candidates,float(prefilter_prominence),min(int(prefilter_checks),len(candidates)),s,pf)
    pf.empty();stats['Visual pre-filter']=pfstats
    s.write(f"Visual pre-filter: checked {pfstats['checked']}, passed {pfstats['passed']}, removed {pfstats['removed']}.")
   st.session_state.v41_search=(equipment.strip(),info,backend_queries,candidates,stats)
  st.success('Preview ready. Inspect the results below before starting downloads.')
 except Exception as e:st.exception(e)

if st.session_state.v41_search:
 eq,info,backend_queries,candidates,stats=st.session_state.v41_search
 if eq==equipment.strip():
  st.subheader('Search preview')
  if use_visual_prefilter:st.caption('Preview is visually pre-filtered: the requested physical target must be visible and meet the preview prominence threshold.')
  prev=candidates[:30]
  st.write(f'{len(candidates)} candidates remain after metadata ranking / optional visual pre-filter; showing {len(prev)}.')
  cols=st.columns(5)
  for i,x in enumerate(prev):
   with cols[i%5]:
    try:st.image(x['image_url'],caption=(x.get('title') or '')[:80],use_container_width=True)
    except:st.write('Preview unavailable')
    st.caption(x.get('provider','')+' · '+urlparse(x.get('page_url','')).netloc)
  st.subheader('Backend statistics')
  st.json(stats)
  with st.expander('Generated backend-specific search queries'):
   for backend,qs in backend_queries.items():
    st.markdown(f'**{backend.title()}**')
    st.code('\n'.join(qs))
  if st.button('2. Start verified collection'):
   p=st.progress(0.);s=st.status('Downloading and verifying…',expanded=True)
   settings={'family_conf':family_conf,'variant_conf':variant_conf,'review_conf':review_conf,'min_prominence':min_prominence,'max_occlusion':max_occlusion,'use_visual_prefilter':use_visual_prefilter,'prefilter_prominence':prefilter_prominence,'prefilter_checks':prefilter_checks}
   try:
    root,n,c=collect(eq,int(target),model.strip(),base.strip(),outdir.strip(),settings,p,s,info,backend_queries,candidates);s.update(label='Complete',state='complete');st.success(f'Saved: {root}');st.write(f"Strict candidates {n} | Downloaded {c['downloaded']} | Accepted {c['accepted']} | Review {c['review']} | Rejected {c['rejected']} | Duplicates {c['duplicates']} | Failures {c['failures']}")
   except Exception as e:s.update(label='Failed',state='error');st.exception(e)

