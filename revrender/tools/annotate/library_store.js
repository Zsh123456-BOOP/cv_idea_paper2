/* Shared offline persistence; per-part drafts avoid localStorage's small quota. */
'use strict';
window.ReviewLibraryStore=(()=>{
  let opening,pending=Promise.resolve(),serial=0;
  const embedded=window.parent&&window.parent!==window;
  function remote(action,key,value){return new Promise((resolve,reject)=>{
    const id='storage-'+Date.now()+'-'+(++serial);
    const timer=setTimeout(()=>{window.removeEventListener('message',receive);reject(Error('自动保存连接超时'));},30000);
    const receive=e=>{if(e.source!==window.parent||e.data?.type!=='review-storage-result'||e.data.id!==id)return;clearTimeout(timer);window.removeEventListener('message',receive);e.data.error?reject(Error(e.data.error)):resolve(e.data.value);};
    window.addEventListener('message',receive);window.parent.postMessage({type:'review-storage',id,action,key,value},'*');
  });}

  function db(){return opening||=(new Promise((resolve,reject)=>{
    const r=indexedDB.open('paper2-review-library',1);
    r.onupgradeneeded=()=>r.result.createObjectStore('progress');
    r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  }));}
  async function get(key){if(embedded)return remote('get',key);const d=await db();return new Promise((resolve,reject)=>{
    const r=d.transaction('progress').objectStore('progress').get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  });}
  function put(key,value){const snapshot=structuredClone(value);
    const task=pending.catch(()=>{}).then(async()=>{if(embedded)return remote('put',key,snapshot);const d=await db();await new Promise((resolve,reject)=>{
      const t=d.transaction('progress','readwrite');t.objectStore('progress').put(snapshot,key);
      t.oncomplete=resolve;t.onerror=()=>reject(t.error);t.onabort=()=>reject(t.error||Error('保存被中断'));
    });});pending=task;return task;
  }
  const flush=()=>pending;
  if(!embedded&&window.addEventListener)window.addEventListener('message',async e=>{
    const m=e.data;if(m?.type!=='review-storage'||!window.REVIEW_LIBRARY)return;
    if(![...document.querySelectorAll('iframe')].some(f=>f.contentWindow===e.source))return;
    if(typeof m.key!=='string'||!m.key.startsWith('pixel-review-'+window.REVIEW_LIBRARY.id+'-'))return;
    try{const value=m.action==='get'?await get(m.key):m.action==='put'?await put(m.key,m.value):undefined;e.source.postMessage({type:'review-storage-result',id:m.id,value},'*');}
    catch(error){e.source.postMessage({type:'review-storage-result',id:m.id,error:error.message},'*');}
  });
  return {get,put,flush};
})();
