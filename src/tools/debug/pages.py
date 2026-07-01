"""HTML for the debug interface pages (home + mask). Stdlib string templates, no deps."""

from src.tools.debug.capture import SLIDERS, FILTER_TOGGLES
from src.tools.debug.control_state import CONTROL_SLIDERS

_CSS = """<style>
body{background:#111;color:#ddd;font-family:monospace;margin:0;padding:10px}
a{color:#6cf} img{max-width:100%;border:1px solid #333}
button{background:#222;color:#ddd;border:1px solid #444;padding:8px 14px;margin:3px;cursor:pointer;font-family:monospace}
button:hover{background:#333}
.play{background:#151}.pause{background:#440}.stop{background:#511}
.row{margin:8px 0}.nav{border-bottom:1px solid #333;padding-bottom:6px;margin-bottom:10px}
label{display:inline-block;width:150px}input[type=range]{width:260px;vertical-align:middle}
.val{color:#6cf;width:52px;display:inline-block;text-align:right}
</style>"""

_NAV = ('<div class=nav><a href="/">Accueil</a> &nbsp;|&nbsp; '
        '<a href="/mask">Masque</a> &nbsp;|&nbsp; '
        '<a href="/intelligence">Intelligence</a></div>')


def home_page():
    return """<!doctype html><html><head><meta charset=utf-8><title>Robocar — Debug</title>
{css}</head><body>{nav}
<h3>Accueil — Profil &amp; conduite</h3>
<div class=row>Profil :
  <select id=prof onchange="setProfile(this.value)"></select>
</div>
<div class=row>
  <button class=play  onclick="drive('play')">▶ PLAY</button>
  <button class=pause onclick="drive('pause')">⏸ PAUSE</button>
  <button class=stop  onclick="drive('stop')">⏹ STOP</button>
</div>
<div class=row>état : <span id=st>—</span></div>
<script>
function refresh(s){{
  document.getElementById('st').textContent =
    'mode='+s.mode+(s.alive?' (process actif)':'')+' | profil='+s.profile;
  var sel=document.getElementById('prof');
  if(sel.options.length!==s.profiles.length){{
    sel.innerHTML=''; s.profiles.forEach(function(p){{
      var o=document.createElement('option'); o.value=p; o.textContent=p; sel.appendChild(o);}});
  }}
  sel.value=s.profile;
}}
function drive(a){{fetch('/drive?action='+a).then(r=>r.json()).then(refresh);}}
function setProfile(p){{fetch('/profile?name='+encodeURIComponent(p)).then(r=>r.json()).then(refresh);}}
fetch('/status').then(r=>r.json()).then(refresh);
setInterval(function(){{fetch('/status').then(r=>r.json()).then(refresh);}},2000);
</script></body></html>""".format(css=_CSS, nav=_NAV)


def _slider_rows(sliders=SLIDERS):
    rows = []
    for name, lo, hi, step in sliders:
        rows.append(
            '<div class=row><label>{n}</label>'
            '<input type=range min={lo} max={hi} step={st} id="s_{n}" '
            'oninput="setP(\'{n}\',this.value)">'
            '<span class=val id="v_{n}"></span></div>'.format(n=name, lo=lo, hi=hi, st=step))
    return "\n".join(rows)


def _filter_toggles():
    return " ".join(
        '<label><input type=checkbox id="en_{k}" '
        'onchange="setP(\'en_{k}\',this.checked?1:0)"> {l}</label>'.format(k=key, l=label)
        for key, label, _, _ in FILTER_TOGGLES)


def mask_page():
    return """<!doctype html><html><head><meta charset=utf-8><title>Robocar — Masque</title>
{css}</head><body>{nav}
<h3>Masque — calibration (couleur | masque)</h3>
<div class=row>profil masque : <select id=maskprofile onchange="setMaskProfile(this.value)"></select>
  <button onclick="saveProfile()">💾 sauver dans le profil</button></div>
<div class=row><img src="/stream.mjpg"></div>
<div class=row>débit : <b id=fps>—</b> fps &nbsp;|&nbsp; frames traitées : <b id=frames>—</b></div>
<div class=row style="border:1px solid #353;padding:6px">filtres actifs : {toggles}</div>
<div class=row>vue :
  <label><input type=checkbox id=show_mask onchange="setP('show_mask',this.checked?1:0)"> masque</label>
  <label><input type=checkbox id=show_rays onchange="setP('show_rays',this.checked?1:0)"> rayons</label>
</div>
{sliders}
<div class=row>état : <span id=k>—</span></div>
<script>
function setP(n,v){{fetch('/param?name='+n+'&value='+v).then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t;
  var vv=document.getElementById('v_'+n); if(vv) vv.textContent=v;}});}}
function loadParams(){{fetch('/params').then(r=>r.json()).then(function(p){{
  var sel=document.getElementById('maskprofile');
  var list=p._profiles||[];
  if(sel.options.length!==list.length){{
    sel.innerHTML=''; list.forEach(function(n){{
      var o=document.createElement('option'); o.value=n; o.textContent=n; sel.appendChild(o);}});
  }}
  if(p._profile) sel.value=p._profile;
  for(var n in p){{
    var s=document.getElementById('s_'+n); if(s) s.value=p[n];
    var vv=document.getElementById('v_'+n); if(vv) vv.textContent=p[n];
    var c=document.getElementById(n); if(c && c.type==='checkbox') c.checked=!!p[n];
  }}
}});}}
function setMaskProfile(name){{fetch('/mask_profile?name='+encodeURIComponent(name)).then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t; loadParams();}});}}
function saveProfile(){{fetch('/mask_save').then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t;}});}}
loadParams();
function pollStats(){{fetch('/stats').then(r=>r.json()).then(function(s){{
  document.getElementById('fps').textContent=s.fps;
  document.getElementById('frames').textContent=s.frames;}});}}
pollStats(); setInterval(pollStats,1000);
</script></body></html>""".format(css=_CSS, nav=_NAV, sliders=_slider_rows(), toggles=_filter_toggles())


def intelligence_page():
    return """<!doctype html><html><head><meta charset=utf-8><title>Robocar — Intelligence</title>
{css}</head><body>{nav}
<h3>Intelligence — cerveau &amp; contrôleur réactif</h3>
<div class=row>profil actif : <b id=prof>—</b>
  <button onclick="save()">💾 sauver dans le profil</button></div>
<div class=row>cerveau :
  <label><input type=radio name=brain value=model onchange="setBrain('model')"> modèle IA (ONNX)</label>
  <label><input type=radio name=brain value=reactive onchange="setBrain('reactive')"> réactif (algo)</label>
  &nbsp;<span style="color:#fa0">cerveau persisté au clic — prend effet au prochain Play</span>
</div>
<div class=row style="color:#888">Params réactifs — réglage à chaud pendant la conduite :</div>
{sliders}
<div class=row>état : <span id=k>—</span></div>
<script>
function setP(n,v){{fetch('/control_param?name='+n+'&value='+v).then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t;
  var vv=document.getElementById('v_'+n); if(vv) vv.textContent=v;}});}}
function setBrain(k){{fetch('/brain?kind='+k).then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t;}});}}
function save(){{fetch('/control_save').then(r=>r.text()).then(function(t){{
  document.getElementById('k').textContent=t;}});}}
function loadParams(){{fetch('/control_params').then(r=>r.json()).then(function(p){{
  document.getElementById('prof').textContent=p._profile||'—';
  var b=p._brain||'model';
  document.querySelectorAll('input[name=brain]').forEach(function(r){{r.checked=(r.value===b);}});
  for(var n in p){{
    var s=document.getElementById('s_'+n); if(s) s.value=p[n];
    var vv=document.getElementById('v_'+n); if(vv) vv.textContent=p[n];
  }}
}});}}
loadParams();
</script></body></html>""".format(css=_CSS, nav=_NAV, sliders=_slider_rows(CONTROL_SLIDERS))
