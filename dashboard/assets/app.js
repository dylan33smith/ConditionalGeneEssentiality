(function(){
  var SECTIONS = ['home','result','reframe','data','timeline','learnings','directions','status','limits','refs'];
  var LABELS = {home:'Overview',result:'The key result',reframe:'First-principles reframe',data:'Data',timeline:'Experiment timeline',
    learnings:"What we've learned",directions:'Future directions',status:'Status & next steps',
    limits:'Limitations & verification',refs:'References'};
  var nav = document.getElementById('nav');
  var links = Array.prototype.slice.call(nav.querySelectorAll('a'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('main > section'));

  function activate(id){
    if(SECTIONS.indexOf(id)<0) id='home';
    sections.forEach(function(s){ s.classList.toggle('active', s.id===id); });
    links.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href')==='#'+id); });
    var lbl=document.getElementById('pLabel'); if(lbl) lbl.textContent=LABELS[id]||id;
    window.scrollTo(0,0);
    var side=document.getElementById('side'); if(side) side.classList.remove('open');
  }

  function route(){
    var h=(location.hash||'#home').replace('#','');
    if(h.indexOf('exp-')===0){
      activate('timeline');
      var d=document.getElementById(h);
      if(d){ d.open=true; setTimeout(function(){ d.scrollIntoView({behavior:'smooth',block:'center'}); },60); }
      return;
    }
    activate(h);
  }
  window.addEventListener('hashchange', route);

  /* ---- timeline filters ---- */
  var filters=document.getElementById('filters');
  if(filters){
    filters.addEventListener('click', function(e){
      var b=e.target.closest('.chip'); if(!b) return;
      filters.querySelectorAll('.chip').forEach(function(c){c.classList.remove('active');});
      b.classList.add('active');
      var f=b.getAttribute('data-f');
      document.querySelectorAll('#tl > details.exp').forEach(function(d){
        d.classList.toggle('hide', f!=='all' && d.getAttribute('data-verdict')!==f);
      });
    });
  }
  var ea=document.getElementById('expandAll'), ca=document.getElementById('collapseAll');
  if(ea) ea.onclick=function(){document.querySelectorAll('#tl > details.exp:not(.hide)').forEach(function(d){d.open=true;});};
  if(ca) ca.onclick=function(){document.querySelectorAll('#tl > details.exp').forEach(function(d){d.open=false;});};

  /* ---- theme ---- */
  var tt=document.getElementById('themeToggle');
  function setTheme(t){ document.documentElement.setAttribute('data-theme',t); try{localStorage.setItem('dash-theme',t);}catch(e){} }
  try{ var saved=localStorage.getItem('dash-theme'); if(saved) setTheme(saved); }catch(e){}
  if(tt) tt.onclick=function(){ setTheme(document.documentElement.getAttribute('data-theme')==='light'?'dark':'light'); };

  /* ---- present mode ---- */
  var pt=document.getElementById('presentToggle');
  function present(on){
    document.body.classList.toggle('present',on);
    if(pt) pt.textContent = on ? '■ exit' : '▶ present';
  }
  if(pt) pt.onclick=function(){ present(!document.body.classList.contains('present')); };
  var pExit=document.getElementById('pExit'); if(pExit) pExit.onclick=function(){present(false);};
  function step(delta){
    var cur=SECTIONS.indexOf((location.hash||'#home').replace('#','').replace(/^exp-.*/,'timeline'));
    if(cur<0) cur=0;
    var ni=Math.max(0,Math.min(SECTIONS.length-1,cur+delta));
    location.hash='#'+SECTIONS[ni];
  }
  var pn=document.getElementById('pNext'), pp=document.getElementById('pPrev');
  if(pn) pn.onclick=function(){step(1);}; if(pp) pp.onclick=function(){step(-1);};
  document.addEventListener('keydown', function(e){
    if(!document.body.classList.contains('present')) return;
    if(e.key==='ArrowRight'||e.key==='PageDown'){step(1);}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){step(-1);}
    else if(e.key==='Escape'){present(false);}
  });

  /* ---- mobile nav ---- */
  var nt=document.getElementById('navToggle');
  if(nt) nt.onclick=function(){ document.getElementById('side').classList.toggle('open'); };

  route();
})();
