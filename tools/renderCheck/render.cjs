// Synthetic-volume render check of the 3-D view (no model steps): node tools/renderCheck/render.cjs <outPrefix> main nest side outerfar outernest outerside
// Run it before and after a change of volume.ts and compare the PNGs byte for byte (cmp); the scenes are fixed and the render is deterministic.
// usage: node render.cjs <outPrefix> [scene...]   (serves repo root; renders synthetic volumes with VolumeView; no model steps)
const http=require('http'),fs=require('fs'),path=require('path'),{execSync}=require('child_process');
const pw=require(path.join(execSync('npm root -g').toString().trim(),'playwright'));
const root=process.cwd(), out=process.argv[2], scenes=process.argv.slice(3);
const mime={'.html':'text/html','.js':'text/javascript'};
const srv=http.createServer((q,r)=>{try{const f=path.join(root,decodeURIComponent(q.url.split('?')[0]));r.writeHead(200,{'content-type':mime[path.extname(f)]||'application/octet-stream'});r.end(fs.readFileSync(f));}catch{r.writeHead(404);r.end();}}).listen(0,'127.0.0.1',async()=>{
 const port=srv.address().port;
 const b=await pw.chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const p=await b.newPage({viewport:{width:640,height:400}});
 p.on('console',m=>console.log('page:',m.text())); p.on('pageerror',e=>console.log('pageerror',e.message));
 await p.goto(`http://127.0.0.1:${port}/tools/renderCheck/page.html`);
 for(const s of scenes){
  const url=await p.evaluate(async(s)=>await window.renderScene(s),s);
  fs.writeFileSync(`${out}_${s}.png`,Buffer.from(url.split(',')[1],'base64')); console.log('wrote',s);
 }
 await b.close(); srv.close();
});
