/* Shared offline persistence; per-part drafts avoid localStorage's small quota. */
'use strict';
window.ReviewLibraryStore=(()=>{
  let opening,pending=Promise.resolve();
  function db(){return opening||=(new Promise((resolve,reject)=>{
    const r=indexedDB.open('paper2-review-library',1);
    r.onupgradeneeded=()=>r.result.createObjectStore('progress');
    r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  }));}
  async function get(key){const d=await db();return new Promise((resolve,reject)=>{
    const r=d.transaction('progress').objectStore('progress').get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  });}
  function put(key,value){const snapshot=structuredClone(value);
    const task=pending.catch(()=>{}).then(async()=>{const d=await db();await new Promise((resolve,reject)=>{
      const t=d.transaction('progress','readwrite');t.objectStore('progress').put(snapshot,key);
      t.oncomplete=resolve;t.onerror=()=>reject(t.error);t.onabort=()=>reject(t.error||Error('保存被中断'));
    });});pending=task;return task;
  }
  const flush=()=>pending;
  return {get,put,flush};
})();
