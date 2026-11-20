'use strict';
const $ = id => document.getElementById(id);
let token = '', documents = [], labels = [], analysis = null, result = null, manual = [], choices = new Map();
let busy = false;
function showError(error) { $('error').textContent = error.message || 'Processing stopped.'; $('error').hidden = false; }
function clearError() { $('error').hidden = true; $('error').textContent = ''; }
async function api(path, body, binary = false) {
  const response = await fetch('/api/' + path, {method:'POST', headers:{'Content-Type':'application/json','X-CasePatch-Token':token}, body:JSON.stringify(body)});
  if (!response.ok) { const value = await response.json(); throw new Error(value.error || 'Processing stopped.'); }
  return binary ? response.blob() : response.json();
}
async function work(action) {
  if (busy) return;
  busy = true; clearError(); $("operation").hidden=false; $("operation").textContent="Working locally…";
  const buttons = [...document.querySelectorAll('button')];
  const prior = buttons.map(button => button.disabled);
  buttons.forEach(button => button.disabled = true);
  try { await action(); } catch (error) { showError(error); }
  finally { $("operation").hidden=true; busy = false; buttons.forEach((button,i) => { if(button.isConnected) button.disabled = prior[i]; }); refreshActions(); }
}
function refreshActions() { $('analyze').disabled = !documents.length || busy; $('preview').disabled = !analysis || busy; $('export').disabled = !result || !$('confirmed').checked || busy; }
function invalidateOutput() { result = null; $('output').hidden = true; $('confirmed').checked = false; refreshActions(); }
async function discardAnalysis() {
  if (analysis) await api('clear', {case_id:analysis.case_id});
  analysis = null; choices.clear(); $('groups').replaceChildren(); $('review').hidden = true; invalidateOutput();
}
function renderDocuments() {
  $('documents').replaceChildren();
  documents.forEach((doc,i) => {
    const li = document.createElement('li');
    const text = document.createElement('span'); text.textContent = `${i+1}. ${labels[i]} · ${doc.format.toUpperCase()}`;
    const remove = document.createElement('button'); remove.className = 'quiet'; remove.textContent = 'Remove'; remove.setAttribute('aria-label',`Remove document ${i+1}`);
    remove.addEventListener('click', () => work(async()=>{await discardAnalysis();documents.splice(i,1);labels.splice(i,1);manual=[];renderDocuments();}));
    li.append(text,remove); $('documents').append(li);
  });
  $('document-count').textContent = documents.length ? `${documents.length} document${documents.length===1?'':'s'} added.` : 'No documents added.';
  refreshActions();
}
async function addDocuments(newDocuments,newLabels) {
  if(documents.length + newDocuments.length > 20) throw new Error('Add at most 20 documents per case.');
  await discardAnalysis(); manual=[]; documents.push(...newDocuments); labels.push(...newLabels);renderDocuments();
}
$('files').addEventListener('change',()=>work(async()=>{
  const newDocs=[],newLabels=[];
  for(const file of $('files').files) {
    if(file.size > 2000000) throw new Error('Each file must be smaller than 2 MB.');
    const extension=file.name.split('.').pop().toLowerCase();
    if(!['txt','json','jsonl'].includes(extension)) throw new Error('Use text, JSON, or JSONL. PDFs, images, and other binaries are unsupported.');
    let text;
    try { text = new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer()); } catch(_) {throw new Error('Use UTF-8 text files.');}
    if(text.includes('\u0000')) throw new Error('Binary files are unsupported.');
    let content=text, format='text';
    if(extension!=='txt') {
      content=extension==='jsonl'?text.split(/\r?\n/).filter(line=>line.trim()).map(line=>CasePatchJSON.parse(line)):CasePatchJSON.parse(text);
      if(!content || typeof content!=='object') throw new Error('JSON must contain an object or list.');
      format='json';
    }
    newDocs.push({format,content});newLabels.push(file.name);
  }
  await addDocuments(newDocs,newLabels);$('files').value='';
}));
$('add-paste').addEventListener('click',()=>work(async()=>{
  const content=$('paste').value;
  if(!content.trim()) throw new Error('Paste some text first.');
  await addDocuments([{format:'text',content}],['Pasted text']);$('paste').value='';
}));
$('example').addEventListener('click',()=>work(async()=>{
  const response=await fetch('/api/example'); if(!response.ok)throw new Error('The example could not load.');
  const value=await response.json();await discardAnalysis();documents=value.documents;labels=['Fictional bank records','Fictional call transcript'];manual=[];renderDocuments();
}));
function groupLabel(group) { return `${group.kind}: ${group.values.join(' / ')}`; }
function selectOptions(select,kind,current,own) {
  const options=new Map();
  for(const group of analysis.groups) if(group.kind===kind) options.set(group.id,groupLabel(group));
  for(const [sid,choice] of choices) if(choice.group.startsWith('split-') && analysis.groups.some(group=>group.kind===kind&&group.occurrences.some(o=>o.id===sid))) options.set(choice.group,`Separate occurrence ${sid}`);
  for(const group of analysis.groups) if(group.kind===kind && group.occurrences.some(o=>choices.get(o.id).group==='confirmed-'+group.id)) options.set('confirmed-'+group.id,`Confirmed identity: ${group.values.join(' / ')}`);
  if(!options.has(current))options.set(current,`Separate occurrence ${own}`);
  for(const [value,text] of options) {const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
  select.value=current;
}
function renderReview() {
  $('groups').replaceChildren();
  $('analysis-summary').textContent=`${analysis.characters.toLocaleString()} input characters · ${analysis.groups.length} detected values or linked groups · ${analysis.learned_detector?'Local name model and rules':'Limited rules mode; unstructured names can be missed'}`;
  if(analysis.issues.some(issue=>issue.startsWith('AMBIGUOUS_'))) $('analysis-summary').textContent+=' · Ambiguous name parts need a link or a Keep choice.';
  if(analysis.issues.includes('CONFLICTING_PERSON_IDS_REQUIRES_REVIEW')) $('analysis-summary').textContent+=' · Different person IDs share a name. Confirm or split those identities.';
  for(const group of analysis.groups) {
    const section=document.createElement('div');section.className='group';
    const heading=document.createElement('div');heading.className='group-heading';
    const name=document.createElement('strong');name.textContent=group.values.join(' / ');
    const type=document.createElement('span');type.className='kind';type.textContent=`${group.kind} · ${group.occurrences.length} occurrence${group.occurrences.length===1?'':'s'}`;
    heading.append(name,type);section.append(heading);
    const controls=document.createElement('div');controls.className='actions';
    if(group.occurrences.some(o=>o.identity_review)) {
      const warning=document.createElement('p');warning.textContent='Different customer or patient IDs share this name. Review the records before choosing one identity or splitting occurrences.';section.append(warning);
      const confirm=document.createElement('button');confirm.className='quiet';confirm.textContent='Confirm one identity';
      confirm.addEventListener('click',()=>{for(const occurrence of group.occurrences)choices.get(occurrence.id).group='confirmed-'+group.id;invalidateOutput();renderReview();});controls.append(confirm);
    }
    const toggle=document.createElement('button');toggle.className='quiet';
    const allSelected=group.occurrences.every(o=>choices.get(o.id).selected);toggle.textContent=allSelected?'Keep original':'Replace all';
    toggle.addEventListener('click',()=>{for(const occurrence of group.occurrences)choices.get(occurrence.id).selected=!allSelected;invalidateOutput();renderReview();});controls.append(toggle);
    if(group.occurrences.length>1) {const split=document.createElement('button');split.className='quiet';split.textContent='Split occurrences';split.addEventListener('click',()=>{for(const o of group.occurrences)choices.get(o.id).group='split-'+o.id;invalidateOutput();renderReview();});controls.append(split);}
    section.append(controls);
    const details=document.createElement('details');details.open=group.occurrences.some(o=>o.source.startsWith('ambiguous-')||o.identity_review);
    const summary=document.createElement('summary');summary.textContent='Review occurrences and identity links';details.append(summary);
    for(const occurrence of group.occurrences) {
      const row=document.createElement('div');row.className='occurrence';
      const label=document.createElement('label');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=choices.get(occurrence.id).selected;checkbox.setAttribute('aria-label',`Replace ${occurrence.id}`);
      checkbox.addEventListener('change',()=>{choices.get(occurrence.id).selected=checkbox.checked;invalidateOutput();});
      const context=document.createElement('span');context.className='context';context.textContent=`Document ${occurrence.document+1}: ${occurrence.context}`;label.append(checkbox,context);
      const link=document.createElement('select');link.setAttribute('aria-label',`Replacement identity for ${occurrence.id}`);selectOptions(link,group.kind,choices.get(occurrence.id).group,occurrence.id);
      link.addEventListener('change',()=>{choices.get(occurrence.id).group=link.value;invalidateOutput();});row.append(label,link);details.append(row);
    }
    section.append(details);$('groups').append(section);
  }
  $('review').hidden=false;
}
async function analyze() {
  await discardAnalysis();analysis=await api('analyze',{documents,manual});
  for(const group of analysis.groups)for(const occurrence of group.occurrences)choices.set(occurrence.id,{id:occurrence.id,group:group.id,selected:occurrence.selected});
  renderReview();
}
$('analyze').addEventListener('click',()=>work(analyze));
$('add-manual').addEventListener('click',()=>work(async()=>{
  const value=$('manual-value').value;if(value.length<2)throw new Error('Enter at least two exact source characters.');
  const proposed={value,kind:$('manual-kind').value};manual.push(proposed);
  try{await analyze();$('manual-value').value='';}catch(error){manual.pop();throw error;}
}));
function asText(doc) {return doc.format==='text'?doc.content:JSON.stringify(doc.content,null,2);}
function renderOutputDocument() {const index=Number($('output-document').value);$('before').textContent=asText(documents[index]);$('after').textContent=asText(result.documents[index]);}
$('preview').addEventListener('click',()=>work(async()=>{
  result=await api('transform',{case_id:analysis.case_id,decisions:[...choices.values()]});
  $('result-summary').textContent=`${result.report.replacement_count} occurrences replaced. ${result.report.kept_detected_occurrences} detected occurrences kept. Other source text and numeric values stay unchanged.`;
  $('output-document').replaceChildren();
  result.documents.forEach((doc,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`Document ${i+1}`;$('output-document').append(option);});
  $('confirmed').checked=false;renderOutputDocument();$('output').hidden=false;$('output').scrollIntoView({behavior:'smooth',block:'start'});
}));
$('output-document').addEventListener('change',renderOutputDocument);
$('confirmed').addEventListener('change',refreshActions);
$('export').addEventListener('click',()=>work(async()=>{
  if(!$('confirmed').checked)throw new Error('Review the output before exporting.');
  const blob=await api('export',{case_id:analysis.case_id,review_confirmed:true},true);
  const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='casepatch-review-required.zip';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
}));
$('clear').addEventListener('click',()=>work(async()=>{await discardAnalysis();documents=[];labels=[];manual=[];$('paste').value='';$('manual-value').value='';$('files').value='';$('before').textContent='';$('after').textContent='';renderDocuments();}));
(async()=>{try{const response=await fetch('/api/config');if(!response.ok)throw new Error('Could not connect to the local app.');const config=await response.json();token=config.csrf;$('model').textContent=config.learned_detector?'Local name detection is ready. Source data stays on this computer.':'Limited rules mode. Names in free text may be missed. Install the local name model for broader detection.';}catch(error){showError(error)}})();
