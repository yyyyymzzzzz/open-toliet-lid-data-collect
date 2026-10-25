window.R5=(()=>{
  const listeners=[];
  const state={connected:false,status:null,episodes:[],exports:[],evaluations:[]};
  const escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=(v,n=1)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});
  async function api(path,options={}){
    const response=await fetch(path,{signal:AbortSignal.timeout(path==='/api/motion/heartbeat'?1500:options.method==='POST'?12000:3000),...options,cache:'no-store',headers:{'Content-Type':'application/json','X-R5-Panel':'1',...(options.headers||{})}});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'请求失败');return result;
  }
  const post=(path,payload)=>api(path,{method:'POST',body:JSON.stringify(payload)});
  let toastTimer,busy=false,lastCollections=0;
  function toast(message,error=false){const t=document.getElementById('toast');if(!t)return;t.textContent=message;t.style.background=error?'#991b1b':'#0b1220';t.style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.style.display='none',5000);}
  function notify(){listeners.forEach(fn=>{try{fn(state);}catch(e){console.error(e);}});}
  function acceptStatus(status){
    const old=state.status;
    if(old?.now&&status?.now&&Date.parse(status.now)<Date.parse(old.now))return false;
    if(old?.service_instance_id===status?.service_instance_id&&Number.isFinite(old?.motion?.session_generation)&&Number.isFinite(status?.motion?.session_generation)&&status.motion.session_generation<old.motion.session_generation)return false;
    Object.assign(state,{connected:true,status,connectionError:null});notify();return true;
  }
  async function refresh(force=true){
    if(busy||location.protocol==='file:')return;busy=true;
    try{
      const status=await api('/api/status');acceptStatus(status);
      if(force||Date.now()-lastCollections>4000){
        try{const [episodes,exports,evaluations]=await Promise.all([api('/api/episodes'),api('/api/exports'),api('/api/evaluations')]);
        Object.assign(state,{episodes:episodes.episodes,exports:exports.exports,evaluations:evaluations.evaluations});lastCollections=Date.now();state.collectionError=null;}catch(e){state.collectionError=e.message;}
      }
    }catch(e){state.connected=false;state.connectionError=e.message;}
    finally{busy=false;notify();}
  }
  function selectTab(id){
    const sections=[...document.querySelectorAll('main > section.sec')];
    if(!sections.some(s=>s.id===id))id=sections[0]?.id;
    sections.forEach(s=>s.hidden=s.id!==id);
    document.querySelectorAll('.topnav a').forEach(a=>{a.classList.toggle('active',a.hash==='#'+id);a.setAttribute('aria-current',a.hash==='#'+id?'page':'false');});
    if(id)history.replaceState(null,'','#'+id);
  }
  function chart(element,series,{unit='',height=190}={}){
    const valid=series.map(s=>({...s,points:s.points.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]))}));
    const all=valid.flatMap(s=>s.points);if(!all.length){element.innerHTML='<div class="chart-empty">暂无可绘制的数据</div>';return;}
    const w=700,h=height,left=54,right=20,top=18,bottom=35;
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;all.forEach(([x,y])=>{xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);});
    if(xmax===xmin)xmax=xmin+1;if(ymax===ymin){ymax+=.5;ymin-=.5;}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
    const x=v=>left+(v-xmin)/(xmax-xmin)*(w-left-right),y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
    let svg=`<svg role="img" aria-label="${escape(unit)}折线图" viewBox="0 0 ${w} ${h}" style="width:100%;display:block">`;
    for(let i=0;i<4;i++){const v=ymin+(ymax-ymin)*i/3;svg+=`<line x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}" stroke="#e3e8f1"/><text x="${left-9}" y="${y(v)+4}" text-anchor="end" fill="#5f6b80" font-size="11">${fmt(v,2)}</text>`;}
    for(let i=0;i<5;i++){const v=xmin+(xmax-xmin)*i/4;svg+=`<text x="${x(v)}" y="${h-9}" text-anchor="middle" fill="#5f6b80" font-size="11">${fmt(v,1)}</text>`;}
    valid.forEach(s=>{svg+=`<polyline points="${s.points.map(p=>x(p[0]).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ')}" stroke="${escape(s.color)}" stroke-width="2" fill="none" stroke-linejoin="round"/>`;});
    svg+=`</svg><div class="chart-legend">${valid.map(s=>`<span><i style="background:${escape(s.color)}"></i>${escape(s.name)}</span>`).join('')}<span>${escape(unit)}</span></div>`;element.innerHTML=svg;
  }
  function on(fn){listeners.push(fn);fn(state);return()=>{const i=listeners.indexOf(fn);if(i>=0)listeners.splice(i,1);};}
  document.addEventListener('DOMContentLoaded',()=>{
    document.querySelectorAll('.topnav a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();selectTab(a.hash.slice(1));}));
    selectTab(location.hash.slice(1)||'capture');
    on(s=>{
      const status=s.status;document.getElementById('connection-badge').textContent=s.connected?'机器人已连接':'等待机器人连接';
      document.getElementById('updated-at').textContent=s.connected?new Date(status.now).toLocaleTimeString('zh-CN')+' 更新':'等待机器人连接';
      const metric=(id,value,unit)=>document.getElementById(id).innerHTML=escape(value)+`<small>${escape(unit)}</small>`;
      metric('hero-accepted',s.connected?fmt(status.summary.accepted,0):'—','条');
      const cams=Object.values(status?.observation?.cameras||{});const fps=cams.length?Math.min(...cams.map(c=>c.hz||0)):null;
      metric('hero-fps',s.connected?fmt(fps):'—','Hz');metric('hero-disk',s.connected?fmt(status.disks.data.free_bytes/2**30,0):'—','GiB');
      document.getElementById('hero-mode').textContent=s.connected?({IDLE:'待采集',RECORDING:'录制中',REVIEW:'待回看',ERROR:'需检查'}[status.mode]||status.mode):'待连接';
    });refresh();setInterval(()=>refresh(false),1000);
  });
  return{state,on,api,post,refresh,acceptStatus,toast,escape,fmt,chart,selectTab};
})();
