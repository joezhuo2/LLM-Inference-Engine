type Pt = number[];
type XY = [number, number];
type P3 = [number, number, number];
type BoxStyle = { edge?: string; top?: string; side?: string; front?: string; lw?: number };
type Kit = ReturnType<typeof plateKit>;
export type MachineInk = { bg: string; text: string; accent: string; warn: string };
const prototypeInk: MachineInk = { bg: '#191813', text: '#e7e2d6', accent: '#a9c3d9', warn: '#e8b7c2' };
const channels = (c: string) => c[0] === '#' ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) : (c.match(/[\d.]+/g) ?? []).map(Number);
const toward = (a: number[], b: number[]) => a.map((v, i) => b[i] - v);
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const base = channels(prototypeInk.bg), u = toward(base, channels(prototypeInk.text)), v = toward(base, channels(prototypeInk.accent));
const uu = dot(u, u), uv = dot(u, v), vv = dot(v, v), det = uu * vv - uv * uv;
let site = { bg: base, text: u, accent: v, warn: channels(prototypeInk.warn), shadow: [17, 23, 18] }, same = true;
const toned = new Map<string, string>();
export function setMachineInk(ink: MachineInk) {
  const bg = channels(ink.bg);
  same = (['bg', 'text', 'accent', 'warn'] as const).every((key) => channels(ink[key]).join() === channels(prototypeInk[key]).join());
  site = { bg, text: toward(bg, channels(ink.text)), accent: toward(bg, channels(ink.accent)), warn: channels(ink.warn), shadow: channels(ink.text) };
  toned.clear();
}
export function tone(color: string) {
  if (same) return color;
  const known = toned.get(color);
  if (known) return known;
  const [r, g, b, a] = channels(color), d = toward(base, [r, g, b]);
  const alpha = (vv * dot(d, u) - uv * dot(d, v)) / det, beta = (uu * dot(d, v) - uv * dot(d, u)) / det;
  const mapped = color.startsWith('rgba(17,23,18,') ? site.shadow : color === prototypeInk.warn ? site.warn : site.bg.map((s, i) => s + alpha * site.text[i] + beta * site.accent[i] + d[i] - alpha * u[i] - beta * v[i]);
  const rgb = mapped.map((c) => Math.round(Math.min(255, Math.max(0, c)))).join(',');
  const out = a === undefined ? `rgb(${rgb})` : `rgba(${rgb},${a})`;
  if (toned.size > 4096) toned.clear();
  toned.set(color, out);
  return out;
}
const palette = {line:'#a9ac96', low:'#737867', top:'#373b30', side:'#23271f', front:'#292c23', amber:'#a9c3d9', copper:'#6485a0', darkCopper:'#465966'};
function plateKit(ctx: CanvasRenderingContext2D, cx: number, cy: number, scale: number) {
  const p = (...[x, y, z]: number[]): XY => [cx + scale*(x-.86*y), cy + scale*(.42*x+.49*y-z)];
  function path(points: Pt[], edge: string | null, fill?: string | null, width=1) {
    ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));
    if(fill){ctx.closePath();ctx.fillStyle=tone(fill);ctx.fill()}
    if(edge){ctx.lineWidth=width;ctx.strokeStyle=tone(edge);ctx.stroke()}
  }
  function line(a: P3, b: P3, color=palette.line, w=.85){path([p(...a),p(...b)],color,null,w)}
  function box(x: number, y: number, z: number, w: number, d: number, h: number, style: BoxStyle={}) {
    const {edge=palette.line,top=palette.top,side=palette.side,front=palette.front,lw=.95}=style;
    const a=p(x,y,z+h),b=p(x+w,y,z+h),c=p(x+w,y+d,z+h),e=p(x,y+d,z+h);
    path([e,c,p(x+w,y+d,z),p(x,y+d,z)],edge,front,lw);
    path([b,c,p(x+w,y+d,z),p(x+w,y,z)],edge,side,lw);
    path([a,b,c,e],edge,top,lw);
  }
  function pad(x: number, y: number, z: number, w: number, d: number, h: number, activity=0){
    box(x,y,z,w,d,h,{
      top:mixInk(palette.top,palette.copper,activity), side:mixInk(palette.side,palette.darkCopper,activity),
      front:mixInk(palette.front,'#895a3c',activity), edge:mixInk(palette.line,palette.amber,activity)
    });
  }
  function prism(outline: Pt[], z: number, height: number, top: string, edge=palette.line, side=palette.side) {
    const upper=outline.map(([x,y])=>p(x,y,z+height));
    for(let i=0;i<outline.length;i++){
      const j=(i+1)%outline.length, a=outline[i], b=outline[j];
      if((b[0]-a[0])*.49-(b[1]-a[1])*.42>0)continue;
      path([upper[i],upper[j],p(...b,z),p(...a,z)],edge,side,.65);
    }
    path(upper,edge,top,.85);
  }
  function ring(x: number, y: number, z: number, r: number, fill: string | null, edge: string | null=palette.line, lw=.75){
    const c=p(x,y,z);
    ctx.save();ctx.transform(scale,scale*.42,-scale*.86,scale*.49,...c);
    ctx.beginPath();ctx.arc(0,0,r,0,Math.PI*2);
    if(fill){ctx.fillStyle=tone(fill);ctx.fill()}if(edge){ctx.strokeStyle=tone(edge);ctx.lineWidth=lw/scale;ctx.stroke()}ctx.restore();
  }
  function trace(points: Pt[], z: number, color=palette.low, w=.8){path(points.map(([x,y])=>p(x,y,z)),color,null,w)}
  function install(t: number, lift: number, draw: () => void, travelY=0){
    if(t<=0)return;
    if(t>=1){draw();return}
    const gap=1-soften(t);
    ctx.save();
    ctx.globalAlpha*=soften(clamp01(lift || travelY ? t/.20 : t));
    ctx.translate(-scale*.86*travelY*gap,scale*(.49*travelY-lift)*gap);
    draw();
    ctx.restore();
  }
  function seat(t: number, x: number, y: number, z: number, w: number, d: number, lift: number, draw: () => void){
    if(t<=0)return;
    if(t>=1){draw();return}
    const gap=1-soften(t);
    const dx=w*.5,dy=d*.5;
    const footprint=[[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]].map(([u,v])=>p(x+u,y+v,z));
    const shadow=soften(clamp01(t/.16))*(.16+.24*(1-gap));
    path(footprint,null,'rgba(17,23,18,'+shadow+')');
    install(t,lift,draw);
  }
  return {p,line,box,pad,path,prism,ring,trace,install,seat};
}
const clamp01 = (v: number) => Math.min(1,Math.max(0,v));
const ramp = (v: number, a: number, b: number) => clamp01((v-a)/(b-a));
function slot(age: number, start: number, end: number, index=0, count=1){
  return clamp01((age-start)*count/(end-start)-index);
}
function soften(t: number){return t*t*(3-2*t)}
function mixInk(from: string, to: string, t: number){
  if(t<=0)return from;
  if(t>=1)return to;
  const a=parseInt(from.slice(1),16), b=parseInt(to.slice(1),16);
  const channel=(shift: number)=>Math.round(((a>>shift)&255)*(1-t)+((b>>shift)&255)*t);
  return 'rgb('+channel(16)+','+channel(8)+','+channel(0)+')';
}
function transistor(k: Kit, age: number){
  const {box,line,prism,ring,trace,path,p}=k;
  const gateGap=1-soften(slot(age,-.05,.36)),zGate=17+gateGap*86;
  box(-103,-73,-32,206,146,32,{top:'#576057',side:'#333d39',front:'#414b46',edge:'#cecfc0'});
  for(let x=-99;x<99;x+=4)line([x,73,-31],[Math.min(103,x+13),73,-3],'#899285',.55);
  for(let y=-70;y<70;y+=5)line([103,y,-31],[103,Math.min(73,y+12),-3],'#748276',.6);
  for(const depth of [-27,-24,-19]){
    line([-103,73,depth],[103,73,depth],'#a5ac9c',.48);
    line([103,73,depth],[103,-73,depth],'#838e81',.48);
  }
  for(const x of [-79,31]){
    const well=[[x,-60],[x+49,-60],[x+49,73],[x,73]];
    prism(well,0,.8,'#758e7b','#b6c6ad');
    const cross=[[x,0],[x+49,0],[x+46,-8],[x+40,-13],[x+8,-13],[x+2,-8]];
    path(cross.map(([u,z])=>p(u,73,z)),'#c1c8ae','#677f6b',.8);
    for(let y=-54;y<70;y+=5)trace([[x+3,y],[x+46,y]],1,'#98ac93',.65);
  }
  for(let i=-2;i<=2;i++){
    const y=i*22;box(-35,y-2,1,70,4,6,{top:'#b2b7a5',edge:'#aeb5a0',side:'#7d8877'});
  }
  for(const side of [-1,1]){
    const x=side*56, spread=1-soften(slot(age,-.95,-.05,(side+1)/2,2));
    const zMetal=12+spread*34;
    for(let j=-2;j<=2;j++){
      const y=j*23;
      ring(x,y,1.4,6,'#222924','#d3bd94');
      ring(x,y,1.5,3.8,null,'#a09a81',.55);
      const plug=Array.from({length:12},(_,i)=>[x+4.3*Math.cos(i*Math.PI/6),y+4.3*Math.sin(i*Math.PI/6)]);
      prism(plug,zMetal-11,11,'#d4c4a4','#afa68b','#726b58');
      k.install(ramp(spread,.04,.18),0,()=>line([x,y,2],[x,y,zMetal-12],'#7b8270',.6));
    }
    const footprint=[[x-10,-63],[x+9,-63],[x+9,-34],[x+16,-27],[x+16,62],[x-14,62],[x-14,-21],[x-10,-26]];
    prism(footprint,zMetal,5,'#a6afa1','#d0d2bd','#717c70');
    for(const inset of [3,5])trace([[x-10+inset,-58],[x-10+inset,-26],[x-14+inset,-20],[x-14+inset,57]],zMetal+5.1,'#758173',.62);
    for(let j=-2;j<=2;j++)ring(x,j*23,zMetal+5.2,3.1,'#737e72','#d5d7c2');
  }
  const oxide=[[-24,-66],[19,-66],[25,-60],[25,59],[18,66],[-24,66]];
  prism(oxide,zGate-4,4,'#a3a89a','#c4c9b9','#677167');
  const gate=[[-17,-64],[15,-64],[21,-57],[21,55],[14,62],[-21,62],[-21,-58]];
  prism(gate,zGate,12,'#bd8055','#e8b186','#865d43');
  for(let y=-52;y<53;y+=4)trace([[-18,y],[17,y]],zGate+12.1,'#dfab7d',.65);
  const cap=[[-9,-39],[8,-39],[8,27],[15,34],[15,65],[-6,65],[-6,39],[-9,35]];
  prism(cap,zGate+12,4,'#e2b78e','#f2d2ac','#b08765');
  for(const y of [-31,17,54])ring(1,y,zGate+16.1,2.5,'#ba9476','#e9c9a4',.55);
  k.install(ramp(gateGap,.04,.20),0,()=>{
    for(const x of [-19,19])for(const y of [-56,55])line([x,y,6],[x,y,zGate-4],'#89917d',.65);
  });
}
const dieTileDropDuration=1.10;
const dieTileStart=(row: number, column: number)=>1.30+column*.30+row*.21;
const dieWaveEnd=dieTileStart(3,5)+dieTileDropDuration;
function array(k: Kit, age: number) {
  const {prism,box,ring,trace,pad}=k;
  const outline=[[-114,-72],[-104,-84],[114,-84],[114,84],[-114,84]];
  prism(outline,-12,12,'#33413a','#c3c9b2','#242e29');
  prism([[-97,-70],[92,-70],[98,-64],[98,69],[-97,69]],0,2,'#28372e','#909e86');
  for(let row=0;row<4;row++){
    const y=-61+row*32;
    for(let lane=0;lane<4;lane++){
      const x=-108+lane*2.1;
      trace([[x,y],[x,y+6],[x+5,y+11],[x+5,y+15]],.25,'#a39672',.6);
      ring(x,y,.3,.8,'#d3c39b',null);
    }
  }
  for(let i=0;i<40;i++){
    const x=-101+i*5.3;
    for(let edge=0;edge<2;edge++){
      const y=edge===0?-78:73;
      k.seat(slot(age,1.56,2.36,edge*40+i,80),x+1.55,y+2.6,0,3.1,5.2,6,()=>{
        box(x,y,0,3.1,5.2,.9,{top:'#a39270',edge:'#beae8a',lw:.45});
      });
    }
  }
  for(let i=0;i<7;i++){
    const x=-93+i*2.2;
    trace([[x,-60],[x,44-i*3],[x+9,53-i*3],[4,53-i*3],[13,62-i*3],[85,62-i*3]],2.4,'#899375',.65);
  }
  for(let r=0;r<4;r++)for(let c=0;c<6;c++){
    const x=-79+c*28,y=-58+r*27,start=dieTileStart(r,c);
    k.seat(ramp(age,start,start+dieTileDropDuration),x+12.5,y+11,2,25,22,24,()=>{
      const z=2;
      const hot=r===1&&c===2 ? soften(ramp(age,2.38,3.08)) : r===1&&c===3 ?
        soften(ramp(age,2.70,3.44))*(1-soften(ramp(age,3.71,4.34))) : 0;
      const tileEdge=mixInk('#909c82','#dfae7a',hot),hatch=mixInk('#a3ac90','#dec098',hot);
      box(x,y,z,25,22,2,{top:mixInk('#48574a','#78543a',hot),edge:mixInk('#a7b097',palette.amber,hot)});
      for(let v=0;v<2;v++)for(let u=0;u<3;u++){
        const xx=x+1.5+u*7.7,yy=y+1.6+v*8.6;
        box(xx,yy,z+2,6.2,6.6,.5,{top:mixInk(v?'#707b68':'#596b59','#bb8656',hot),edge:tileEdge,lw:.48});
        for(let n=1;n<6;n+=1.5)trace([[xx+n,yy+1],[xx+n,yy+5.4]],z+2.6,hatch,.43);
      }
      for(let i=0;i<3;i++)trace([[x+2,y+18+i],[x+23,y+18+i]],z+2.3,'#b0a586',.55);
    });
  }
  for(let r=0;r<6;r++){
    const y=-61+r*22;
    k.seat(slot(age,3.95,5.30,r,6),101.5,y+8.5,1,11,17,21,()=>{
      const activity=r%2 ? soften(ramp(age,4.52+r*.075,5.22+r*.075)) : 0;
      pad(96,y,1,11,17,3,activity);
      for(let j=0;j<5;j++)trace([[98,y+3+j*2.5],[105,y+3+j*2.5]],4.1,'#bdc2a6',.48);
    });
  }
  ring(-105,63,0,3,'#d6bd88',null);
}
function fan(k: Kit, x: number, y: number, z: number, r: number, turn=0, assembly=1) {
  const {ring,trace,p,path}=k;
  ring(x,y,z,r+2,'#3a4038','#b4b9a6');ring(x,y,z+.2,r,'#1b211d','#787f70');
  for(let b=0;b<11;b++){
    const seat=slot(assembly,0,1,b,12);
    k.install(seat,r*.46,()=>{
      const angle=turn+b*2*Math.PI/11-.46*(1-seat)*(1-seat);
      const polar=(rad: number, turn: number)=>p(x+r*rad*Math.cos(angle+turn),y+r*rad*Math.sin(angle+turn),z+1);
      const leading=[], trailing=[];
      for(let j=0;j<=12;j++){
        const t=j/12, radius=.25+.68*t;
        leading.push(polar(radius,-.61*t*t+.07*t));
        trailing.unshift(polar(radius,-.61*t*t+.07*t+.22+.11*Math.sin(Math.PI*t)));
      }
      path([...leading,...trailing],'#a9b0a2',b%2?'#667469':'#56665b',.7);
      const ridge=leading.slice(3,12).map(([u,v],i)=>[u+.2*i,v+.10*i]);
      path([...ridge,leading[11],leading[3]],null,'#a0aa99');
    });
  }
  k.install(slot(assembly,0,1,11,12),r*.42,()=>{
    ring(x,y,z+2,.27*r,'#727b70','#c3c6b6');
    ring(x,y,z+2.2,.17*r,'#424d42','#909a88');
    for(let i=0;i<3;i++){
      const a=turn+i*Math.PI*2/3;
      ring(x+.214*r*Math.cos(a),y+.214*r*Math.sin(a),z+2.3,.023*r,'#333b32','#b6baa9',.55);
    }
    trace([[x-r*.09,y],[x,y-r*.08],[x+r*.09,y]],z+2.4,'#d7cab1',.8);
  });
}
function screw(k: Kit, x: number, y: number, z: number, r=2.4){
  k.ring(x,y,z,r,'#777f71','#cdd0bb');k.trace([[x-r*.55,y],[x+r*.55,y]],z+.2,'#1c251e',.85);
}
function heatPipe(k: Kit, x0: number, x1: number, y: number, z: number){
  const spine=[[x0,y-12],[x0,y-5],[x0+2,y-2],[x0+6,y],[x1-6,y],[x1-2,y-2],[x1,y-5],[x1,y-12]];
  k.trace(spine,z,'#4a4538',3.8);
  k.trace(spine,z+.25,'#ac8763',2.5);
  k.trace(spine,z+.58,'#d1b494',.65);
}
function device(k: Kit, age: number) {
  const {box,prism,ring,trace,line}=k;
  const board=[[-159,-73],[-151,-80],[145,-80],[159,-67],[159,63],[146,76],[-121,76],[-128,70],[-159,70]];
  prism(board,-8,7,'#354c3e','#c3c7aa','#27392e');
  prism([[-103,74],[105,74],[105,86],[-103,86]],-8,1,'#90855d','#b8a577','#786e50');
  for(let i=0;i<48;i++){
    if(i===11||i===12)continue;
    const x=-101+i*4.25;
    k.seat(slot(age,8.22,8.58,i<11?i:i-2,46),x+1.3,80.5,-7,2.6,9,6,()=>{
      box(x,76,-7,2.6,9,2,{top:'#b69a60',edge:'#dbc48e',front:'#846c47',lw:.45});
    });
  }
  for(let j=0;j<24;j++){
    const x=-145+j*2.2,y=-69+j*2;
    trace([[x,y],[x+12,y],[x+26,y+14],[x+26,44-j*.75],[x+39,57-j*.75],[39,57-j*.75]],0,j%4?'#708477':'#ad9a72',.7);
    ring(x,y,0,1.25,'#202e25','#aeb497',.45);
  }
  for(let j=0;j<15;j++)trace([[16,-48+j*3.3],[45+j,-48+j*3.3],[62+j,-31+j*3.3],[132,-31+j*3.3]],0,'#778877',.6);
  for(let i=0;i<10;i++){
    const x=-112+i*8.2;
    k.seat(slot(age,10.60,10.80,i,10),x+2.85,62.3,0,5.7,2.6,9,()=>{
      box(x,61,0,5.7,2.6,1.3,{top:'#a4987c',edge:'#abae98',lw:.43});
      for(const offset of [.7,5])line([x+offset,61,1.4],[x+offset,63.6,1.4],'#dad5bb',.7);
    });
  }
  const chip=[[-56,-43],[-49,-50],[37,-50],[37,48],[-56,48]];
  k.seat(slot(age,8.70,8.80),-9.5,-1,0,93,98,30,()=>{
    prism(chip,0,4,'#54634c','#a6af90');
    box(-43,-37,4,67,70,3,{top:'#384e45',edge:'#d0c7aa'});
    for(let r=0;r<5;r++)for(let c=0;c<6;c++){
      const x=-40+c*10.3,y=-33+r*12.6;
      k.seat(slot(age,8.80,9.12,r*6+c,30),x+4.4,y+5,7,8.8,10,7,()=>{
        box(x,y,7,8.8,10,1,{top:r===2&&c===2?'#b48255':'#657566',edge:'#a2aa91',lw:.55});
        trace([[x+2,y+1],[x+2,y+9]],8.1,'#c5c4ab',.55);
      });
    }
  });
  for(let i=0;i<8;i++){
    const x=i<4?-79:47,y=-56+(i%4)*28;
    k.seat(slot(age,9.48,10.10,i,8),x+8,y+10,0,16,20,18,()=>{
      box(x,y,0,16,20,3,{top:'#303731',edge:'#8a9585'});
      ring(x+3,y+3,3,1,'#a4ab96',null);
      for(let a=0;a<4;a++)trace([[x+4,y+7+a*2.2],[x+12-a,y+7+a*2.2]],3.1,'#879281',.48);
    });
  }
  for(let j=0;j<8;j++){
    const y=-63+j*16;
    k.seat(slot(age,10.10,10.60,j*4,32),117,y+5,0,14,10,16,()=>{
      box(110,y,0,14,10,5,{top:'#687364',edge:'#a3ad99'});
    });
    for(let i=0;i<3;i++){
      const x=128+i*4;
      k.seat(slot(age,10.10,10.60,j*4+i+1,32),x+1.35,y+3,0,2.7,6,8,()=>{
        box(x,y,0,2.7,6,2,{top:'#b6a88c',edge:'#dbcfb3',lw:.5});
      });
    }
  }
  k.seat(slot(age,9.12,9.22),-157,.5,-14,4,165,17,()=>{
    box(-159,-82,-14,4,165,29,{top:'#c1c4b5',side:'#79857b',front:'#8a9486'});
  });
  for(let i=0;i<4;i++){
    const y=-65+i*33;
    k.seat(slot(age,9.22,9.48,i,4),-144,y+12.5,0,18,25,12,()=>{
      box(-153,y,0,18,25,10,{top:'#a0aa9c',edge:'#d1d4c1'});
      trace([[-154,y+4],[-142,y+4],[-140,y+7],[-140,y+18],[-154,y+18]],10.2,'#555f55',.8);
    });
  }
  if(age>10.8){
    const z=9;
    for(let i=0;i<3;i++)k.install(slot(age,10.80,11.05,i,3),16,()=>heatPipe(k,-118+i*8,117-i*6,70-i*3,z-3));
    const shape=[[-140,-71],[128,-71],[149,-52],[149,52],[132,68],[-127,68],[-143,52],[-143,-60]];
    for(let j=0;j<58;j++){
      const x=-134+j*4.7;
      k.seat(slot(age,11.05,11.50,j,58),x+.75,-1,z,1.5,122,16,()=>{
        box(x,-62,z,1.5,122,6,{top:'#a5ac9d',edge:'#909c8e',lw:.52});
      });
    }
    k.seat(slot(age,11.50,11.74),3,-1.5,z+5,292,139,25,()=>{
        const top=z+9;
        prism(shape,top-4,4,'#3f4941','#bec5b6','#29352e');
        for(const x of [-67,74]){
          ring(x,0,top+.2,57,'#28352e','#869784');
          for(let i=0;i<16;i++){
            const a=i*Math.PI/8,pt=(r: number, t: number)=>[x+r*Math.cos(a+t),r*Math.sin(a+t)];
            trace([pt(53,0),pt(56,.035),pt(56,.21)],top+.3,'#a1aa98',.7);
          }
          const assembly=slot(age,11.74,13.42,x<0?0:1,2);
          fan(k,x,0,top+.6,50,.35,assembly);
        }
        for(const side of [-1,1]){
          const y=side*60;
          trace([[-127,y],[-111,y-3*side],[-40,y-3*side],[-33,y],[83,y],[95,y-3*side],[125,y-3*side]],top+.4,'#a2af9e',.85);
          for(let j=0;j<17;j++){
            const x=-120+j*6.8;
            trace([[x,y-1*side],[x+3,y-5*side]],top+.5,'#26392e',1.4);
          }
        }
        for(let j=0;j<4;j++){
          const x=-124+j*4;
          trace([[x,-43],[x+5,-38],[x+5,37],[x,42]],top+.4,'#94a18f',.8);
        }
        for(let i=0;i<5;i++)trace([[-11+i*3,-63],[10+i*3,-38],[-1+i*3,13],[18+i*3,55]],top+.3,'#8b9685',.8);
        for(const [i,x] of [-128,130].entries())for(const [j,y] of [-53,52].entries()){
          k.install(slot(age,13.42,13.55,i*2+j,4),6,()=>screw(k,x,y,top+.5));
        }
        ring(0,0,top+1,2.8,palette.copper,palette.amber);
    });
  }
  for(const [i,x] of [-144,143].entries())for(const [j,y] of [-64,61].entries()){
    k.install(slot(age,8.58,8.70,i*2+j,4),6,()=>screw(k,x,y,0));
  }
}
function server(k: Kit, age: number) {
  const {box,line,prism,ring,trace,path,p}=k;
  prism([[-155,-91],[-149,-97],[146,-97],[155,-88],[155,90],[-155,90]],-22,6,'#515e53','#b4bdab');
  k.seat(slot(age,14.35,14.50),-.5,-93,-16,307,4,22,()=>{
    box(-154,-95,-16,307,4,36,{top:'#a9b2a4',front:'#647267'});
  });
  k.seat(slot(age,14.50,14.65),-152,-2.5,-16,4,177,22,()=>{
    box(-154,-91,-16,4,177,35,{top:'#a6afa1',side:'#58695d'});
  });
  k.seat(slot(age,14.65,14.82),-3,-7,-15,280,150,32,()=>{
    box(-143,-82,-15,280,150,2,{top:'#344939',edge:'#798d76'});
    for(let j=0;j<4;j++)for(let i=0;i<4;i++){
      const x=-125+j*67+i*3;
      trace([[x+5,58],[x+5,61+i],[x+14,70+i]],-12.8,'#83927a',.64);
    }
  });
  for(let j=0;j<4;j++){
    const x=-125+j*67;
    k.seat(slot(age,14.82,15.10,j,4),x+23,54.5,-13,46,7,14,()=>{
      box(x,51,-13,46,7,3,{top:'#24322a',edge:'#8d9984'});
      for(let i=0;i<12;i++)line([x+3+i*3.3,53,-9.8],[x+3+i*3.3,56,-9.8],'#b5a780',.7);
    });
  }
  for(const [i,x] of [-139,3].entries()){
    const assembly=slot(age,15.10,15.82,i,2);
    k.seat(ramp(assembly,0,.62),x+64.5,-68.5,-12,129,35,24,()=>{
      box(x,-86,-12,129,35,21,{top:'#747f71',front:'#4c574f',edge:'#bcc4b2'});
      for(let j=0;j<18;j++)trace([[x+9+j*4,-80],[x+5+j*4,-59]],9.3,'#3f4d42',1);
      k.install(ramp(assembly,.62,.82),9,()=>{
        box(x+95,-81,9,23,18,2,{top:'#334036',edge:'#939f8c'});
        trace([[x+98,-78],[x+115,-78],[x+115,-69],[x+98,-69],[x+98,-78]],11.1,'#b1b5a2',.8);
      });
      k.install(slot(assembly,.82,1,0,2),6,()=>screw(k,x+5,-81,9.2,1.65));
      k.install(slot(assembly,.82,1,1,2),6,()=>screw(k,x+120,-56,9.2,1.65));
    });
  }
  for(let j=0;j<4;j++){
    const x=-129+j*67,z=-10;
    const assembly=slot(age,15.82,18.10,j,4);
    k.seat(ramp(assembly,0,.16),x+26,3,-13,52,96,16,()=>{
      box(x,-45,-13,52,96,3,{top:'#25372a',edge:'#758970'});
    });
    k.seat(ramp(assembly,.16,.51),x+25.5,2.5,z,51,91,38,()=>{
      prism([[x,-43],[x+45,-43],[x+51,-37],[x+51,48],[x,48]],z,6,'#485a4d','#a3b49e');
      k.seat(ramp(assembly,.51,.70),x+25,-15,z+6.4,38,38,17,()=>fan(k,x+25,-15,z+6.4,18));
      for(let i=0;i<12;i++){
        const finX=x+4+i*3.7;
        k.seat(slot(assembly,.70,1,i,12),finX+.65,25.5,z+6,1.3,31,11,()=>{
          box(finX,10,z+6,1.3,31,3,{top:'#9da794',edge:'#9fa997',lw:.5});
        });
      }
      if(j===0)trace([[x+3,-37],[x+11,-37],[x+14,-33]],z+6.4,palette.amber,1.8);
    });
  }
  k.seat(slot(age,18.10,18.30),.5,87,-22,309,16,25,()=>{
    box(-154,79,-22,309,16,32,{top:'#8e998a',front:'#49584d',edge:'#c7cbb8'});
  });
  for(let j=0;j<8;j++){
    const x=-148+j*36;
    k.install(slot(age,18.50,19.38,j,8),0,()=>{
      box(x,93,-17,32,4,22,{top:'#899682',front:'#354238',edge:'#a5b39c'});
      for(let row=0;row<3;row++)for(let col=0;col<5;col++){
        const xx=x+4+col*4.1,zz=-12+row*5;
        path([[xx,zz],[xx+2,zz],[xx+2.7,zz+1.2],[xx+2,zz+2.4],[xx,zz+2.4],[xx-.7,zz+1.2]].map(([u,v])=>p(u,97.1,v)),null,'#19261e');
      }
      line([x+25,97.3,-12],[x+25,97.3,1],'#d1d2b8',1.5);
      line([x+28,97.3,1],[x+30,97.3,1],j%3===0?palette.amber:'#909d86',1.3);
    },22);
  }
  for(const [i,x] of [-165,154].entries()){
    k.seat(slot(age,18.30,18.50,i,2),x+5.5,86.5,-18,11,15,18,()=>{
      box(x,79,-18,11,15,26,{top:'#9da799',edge:'#c6cdb9'});
      ring(x+5.5,87,8,2.1,'#27342b','#74816f');
    });
  }
  for(const [i,x] of [-137,1].entries()){
    const z=27;
    k.seat(slot(age,19.38,20,i,2),x+62.5,-71,z,125,30,30,()=>{
      box(x,-86,z,125,30,2,{top:'#626e60',edge:'#a3af9c'});
      for(let j=0;j<16;j++)trace([[x+9+j*6,-81],[x+5+j*6,-63]],z+2.1,'#3c493e',1);
    });
  }
}
function overlay(k: Kit, chapter: number, frac: number, age: number) {
  const {line,pad}=k, bright=palette.amber;
  if(chapter>=4&&chapter<=8 && age>dieWaveEnd){
    const reach=112*soften(ramp(age,dieWaveEnd,dieWaveEnd+.42));
    for(let j=0;j<3;j++){
      const y=-46+j*38;
      line([-9,y,15],[-9+Math.min(92,reach),y,15],bright,1.4);
      if(reach>92){
        const bend=(reach-92)/20;
        line([83,y,15],[83+17*bend,y+10*bend,15],bright,1.4);
      }
      k.install(ramp(age,dieWaveEnd+.42,dieWaveEnd+.56),0,()=>pad(5+frac*62,y-1,15,4,3,2,1));
    }
  }
  if(chapter>=9&&chapter<=13){const count=[1,8,1,5,4][chapter-9];for(let i=0;i<count;i++){const y=59+i*1.8;line([-65,y,7],[49,y,7],bright,.95);pad(-57+frac*78+i*3,y,7,5,2,2,1)}}
  if(chapter>=16&&chapter<=18){
    for(let j=0;j<4;j++){
      const assembly=slot(age,15.82,18.10,j,4);
      k.install(ramp(assembly,.94,1),0,()=>{
        for(let i=0;i<(chapter===18?j+1:2);i++){
          const x=-120+j*67+i*6;
          line([x,44,-3.8],[x+3,44,-3.8],bright,1.4);
        }
      });
    }
  }
  if(chapter===20){
    for(let j=0;j<8;j++){
      line([-103+j*12,-56+j*3,25],[-96+j*12,-54+j*3,25],'#e8b7c2',1.35);
      line([-103+j*12,-56-j*2,25],[-96+j*12,-57-j*2,25],palette.low,1.35);
    }
  }
}
function prepare(canvas: HTMLCanvasElement, age: number, cameraSide?: number){
  const {width,height}=canvas.getBoundingClientRect();
  const ratio=Math.min(window.devicePixelRatio||1,2);
  const w=Math.max(1,Math.round(width*ratio)),h=Math.max(1,Math.round(height*ratio));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  const ctx=canvas.getContext('2d')!;
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);
  const portrait=width<height;
  const zoom=portrait?Math.min(width/395,height/355):Math.min(width/600,height/390);
  const side=cameraSide ?? Math.cos((age+1)*Math.PI);
  const centerX=width*(.5+(portrait?.16:.105)*side),centerY=height*(portrait?.47:.52);
  ctx.setTransform(zoom*ratio,0,0,zoom*ratio,(centerX-340*zoom)*ratio,(centerY-223*zoom)*ratio);
  ctx.lineJoin='round';
  return ctx;
}
export function drawMachine(canvas: HTMLCanvasElement, progress: number, reduced: boolean, cameraSide?: number){
  const ctx=prepare(canvas,progress,cameraSide),chapter=Math.min(20,Math.round(progress)),age=reduced?chapter:progress;
  function layer(alpha: number, paint: () => void){if(alpha<=0)return;ctx.save();ctx.globalAlpha*=alpha;paint();ctx.restore()}
  const dieIn=ramp(age,1.15,2),gpuIn=ramp(age,8.2,9),serverIn=ramp(age,14.35,15);
  layer(1-dieIn,()=>{
    const k=plateKit(ctx,340,280,1.5-.65*dieIn);
    transistor(k,age);
    for(let i=0;i<6;i++){
      const x=-82+i*28;
      k.seat(slot(age,.36,1.56,i,6),x+8,61,1,16,10,21,()=>{
        k.pad(x,56,1,16,10,4,i===2 ? soften(ramp(age,.74,1.24)) : 0);
      });
    }
  });
  layer(dieIn*(1-gpuIn),()=>array(plateKit(ctx,340,224,1.5-.45*gpuIn),age));
  layer(gpuIn*(1-serverIn),()=>{
    const pair=ramp(age,13.55,14.2),k=plateKit(ctx,340-pair*83,222-pair*53,1.25-.49*pair);
    device(k,age);
    if(pair>0){
      const peer=plateKit(ctx,341+pair*81,223+pair*61,.76);
      peer.seat(pair,0,0,-8,310,150,45,()=>device(peer,age));
    }
  });
  layer(serverIn,()=>server(plateKit(ctx,340,222,1.25),age));
  const current=Math.floor(age),nextMix=ramp(age-current,.40,.60);
  for(const [part,opacity] of [[current,1-nextMix],[current+1,nextMix]]){
    if((part>=4&&part<=13)||(part>=16&&part<=18)||part===20){
      const scale=part<9?1.5-.45*gpuIn:1.25;
      layer(opacity,()=>overlay(plateKit(ctx,340,part<9?224:222,scale),part,clamp01(age-part+.5),age));
    }
  }
}

const prototypeStories = ['one-token', 'foundations', 'gemm', 'rope', 'paged-kv', 'paged-attention', 'attention-v2', 'split-attention', 'flash-prefill', 'add-rmsnorm', 'swiglu', 'sampling', 'cuda-graphs', 'speculative-decoding', 'tensor-parallelism', 'memory-budget', 'serving', 'chunked-prefill', 'where-the-time-went', 'beyond', 'dead-ends'];
const stageStandIns: Record<string, string> = { quantization: 'where-the-time-went' };
export function machineStages(slugs: string[]) {
  const stages: number[] = [];
  for (const slug of slugs) {
    const at = prototypeStories.indexOf(stageStandIns[slug] ?? slug);
    stages.push(at < 0 ? (stages.at(-1) ?? 0) : at);
  }
  return stages;
}
export function machineAge(stages: number[], progress: number) {
  const stops = [-1, ...stages], at = Math.min(stops.length - 1, Math.max(0, progress + 1)), i = Math.floor(at), next = stops[Math.min(i + 1, stops.length - 1)];
  return stops[i] + (next - stops[i]) * (at - i);
}
