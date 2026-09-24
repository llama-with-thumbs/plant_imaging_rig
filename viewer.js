/* A small WebGL 2 viewer for the carving.
 *
 * Geometry is one interleaved buffer (position, normal, uv per vertex, then
 * uint16 triangle indices) fetched from model/fasting-buddha.bin; the base
 * colour and normal map are ordinary JPEGs. No library, no loader -- just the
 * parts a single textured, normal-mapped mesh actually needs.
 *
 * The one subtlety is the tangent frame. A normal map only means something
 * relative to the frame it was baked in, and this one was NOT baked against the
 * usual UV-derivative tangents: the baker built T from the smooth normal and a
 * fixed up vector. The fragment shader below reconstructs that exact frame. Using
 * the conventional cotangent frame instead makes the relief point in arbitrary
 * directions -- which is what a mismatched frame looks like, and it is subtle
 * enough to pass for "a bit noisy" if you are not looking for it.
 */
(function () {
  "use strict";

  var VERT = `#version 300 es
  in vec3 aPos; in vec3 aNrm; in vec2 aUv;
  uniform mat4 uProj, uView; uniform mat3 uRot;
  out vec3 vNrmModel; out vec3 vNrmWorld; out vec2 vUv; out vec3 vPos;
  void main(){
    vNrmModel = aNrm;                 // the frame the normal map was baked in
    vNrmWorld = uRot * aNrm;
    vUv = aUv;
    vec3 p = uRot * aPos;
    vPos = p;
    gl_Position = uProj * uView * vec4(p, 1.0);
  }`;

  var FRAG = `#version 300 es
  precision highp float;
  in vec3 vNrmModel; in vec3 vNrmWorld; in vec2 vUv; in vec3 vPos;
  uniform sampler2D uBase, uNorm;
  uniform float uRelief, uExposure; uniform mat3 uRot;
  out vec4 frag;

  void main(){
    vec3 Nm = normalize(vNrmModel);
    // the baker's frame, rebuilt exactly: T from a fixed up vector, not from UVs
    vec3 up = abs(Nm.y) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 T = normalize(cross(up, Nm));
    vec3 B = cross(Nm, T);
    vec3 nt = texture(uNorm, vUv).xyz * 2.0 - 1.0;
    nt.xy *= uRelief;
    vec3 n = normalize(uRot * normalize(T * nt.x + B * nt.y + Nm * nt.z));

    vec3 albedo = pow(texture(uBase, vUv).rgb, vec3(2.2));
    vec3 V = normalize(vec3(0.0, 0.0, 1.0));

    // a key, a cool fill from the opposite side, and a dim rim to find the edge
    vec3 L1 = normalize(vec3(-0.45, 0.62, 0.65));
    vec3 L2 = normalize(vec3(0.72, 0.10, 0.34));
    vec3 lit = albedo * (max(dot(n, L1), 0.0) * vec3(1.00, 0.96, 0.89) * 1.02
                       + max(dot(n, L2), 0.0) * vec3(0.44, 0.50, 0.58) * 0.40
                       + vec3(0.30, 0.30, 0.32));
    vec3 H = normalize(L1 + V);
    lit += vec3(0.055) * pow(max(dot(n, H), 0.0), 26.0);
    lit += vec3(0.10, 0.09, 0.08) * pow(1.0 - max(dot(n, V), 0.0), 3.4);

    lit *= uExposure;
    frag = vec4(pow(lit / (lit + 0.92), vec3(1.0 / 2.2)), 1.0);
  }`;

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  }

  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), d = near - far;
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
      0, 0, (far + near) / d, -1, 0, 0, 2 * far * near / d, 0]);
  }

  function lookAtZ(dist) {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -dist, 1]);
  }

  function rot(yaw, pitch) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    // column-major: Rx(pitch) * Ry(yaw)
    return new Float32Array([
      cy, sp * sy, -cp * sy,
      0, cp, sp,
      sy, -sp * cy, cp * cy
    ]);
  }

  /* opts: { canvas, note, hint, verts, tris, radius,
             geometry (url), base (url), normal (url), onInteract } */
  window.initViewer = function (opts) {
    var canvas = document.getElementById(opts.canvas);
    var gl = canvas.getContext("webgl2", { antialias: true, alpha: true });
    var note = document.getElementById(opts.note);
    function say(t) { if (note) note.textContent = t; }
    if (!gl) {
      say("This browser has no WebGL 2, so the model cannot be shown here. The downloadable files still open in any 3D viewer.");
      canvas.style.display = "none";
      return null;
    }

    var prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);

    var vao = null;          // set once the geometry has arrived
    var ready = 0;           // textures loaded so far
    function maybeDone() {
      if (vao && ready === 2) say(opts.hint);
      draw();
    }

    fetch(opts.geometry).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.arrayBuffer();
    }).then(function (buf) {
      var stride = 8 * 4;
      var vertBytes = opts.verts * stride;
      var v = gl.createVertexArray();
      gl.bindVertexArray(v);
      var vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(buf, 0, opts.verts * 8), gl.STATIC_DRAW);
      ["aPos", "aNrm", "aUv"].forEach(function (name, i) {
        var loc = gl.getAttribLocation(prog, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, i === 2 ? 2 : 3, gl.FLOAT, false, stride, [0, 12, 24][i]);
      });
      var ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(buf, vertBytes, opts.tris * 3), gl.STATIC_DRAW);
      vao = v;
      maybeDone();
    }).catch(function (e) {
      say("Could not load the model (" + e.message + ").");
    });

    // no UNPACK_FLIP_Y: the atlas was baked with v = 0 at the top row, which is
    // also where WebGL puts the first row of pixel data
    function loadTex(unit, url, uniform) {
      var tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE,
                    new Uint8Array([160, 140, 110]));
      var img = new Image();
      img.onload = function () {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        var ext = gl.getExtension("EXT_texture_filter_anisotropic");
        if (ext) gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT,
                                  Math.min(8, gl.getParameter(ext.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
        ready++;
        maybeDone();
      };
      img.onerror = function () { say("Could not load " + url + "."); };
      img.src = url;
      gl.uniform1i(gl.getUniformLocation(prog, uniform), unit);
    }
    loadTex(0, opts.base, "uBase");
    loadTex(1, opts.normal, "uNorm");

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    var uProj = gl.getUniformLocation(prog, "uProj");
    var uView = gl.getUniformLocation(prog, "uView");
    var uRot = gl.getUniformLocation(prog, "uRot");
    var uRelief = gl.getUniformLocation(prog, "uRelief");
    var uExposure = gl.getUniformLocation(prog, "uExposure");

    var yaw = 0.25, pitch = -0.13, dist = opts.radius * 3.15, spin = true;
    var relief = 1.0, exposure = 1.45;

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== (w * dpr | 0) || canvas.height !== (h * dpr | 0)) {
        canvas.width = w * dpr | 0;
        canvas.height = h * dpr | 0;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      return canvas.width / Math.max(canvas.height, 1);
    }

    function draw() {
      var aspect = resize();
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!vao) return;
      gl.useProgram(prog);
      gl.bindVertexArray(vao);
      gl.uniformMatrix4fv(uProj, false, perspective(0.62, aspect, dist * 0.05, dist * 6));
      gl.uniformMatrix4fv(uView, false, lookAtZ(dist));
      gl.uniformMatrix3fv(uRot, false, rot(yaw, pitch));
      gl.uniform1f(uRelief, relief);
      gl.uniform1f(uExposure, exposure);
      gl.drawElements(gl.TRIANGLES, opts.tris * 3, gl.UNSIGNED_SHORT, 0);
    }

    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) spin = false;
    var last = 0;
    function frame(t) {
      if (spin) {
        yaw += (t - last) * 0.00035;
        draw();
      }
      last = t;
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(function (t) { last = t; draw(); frame(t); });

    function changed() { if (opts.onInteract) opts.onInteract(spin); }

    var drag = null;
    function pos(e) {
      var t = e.touches ? e.touches[0] : e;
      return { x: t.clientX, y: t.clientY };
    }
    function down(e) { spin = false; drag = pos(e); changed(); }
    function move(e) {
      if (!drag) return;
      var p = pos(e);
      yaw += (p.x - drag.x) * 0.0085;
      pitch = Math.max(-1.25, Math.min(1.25, pitch + (p.y - drag.y) * 0.0075));
      drag = p;
      if (e.cancelable) e.preventDefault();
      draw();
    }
    function up() { drag = null; }
    canvas.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    canvas.addEventListener("touchstart", down, { passive: true });
    canvas.addEventListener("touchmove", move, { passive: false });
    canvas.addEventListener("touchend", up);
    canvas.addEventListener("wheel", function (e) {
      dist = Math.max(opts.radius * 1.35, Math.min(opts.radius * 7, dist * (1 + e.deltaY * 0.0012)));
      e.preventDefault();
      draw();
    }, { passive: false });
    window.addEventListener("resize", draw);

    canvas.tabIndex = 0;
    canvas.addEventListener("keydown", function (e) {
      var step = 0.12;
      if (e.key === "ArrowLeft") { spin = false; yaw -= step; }
      else if (e.key === "ArrowRight") { spin = false; yaw += step; }
      else if (e.key === "ArrowUp") { spin = false; pitch = Math.max(-1.25, pitch - step); }
      else if (e.key === "ArrowDown") { spin = false; pitch = Math.min(1.25, pitch + step); }
      else if (e.key === " ") { spin = !spin; }
      else return;
      changed();
      e.preventDefault();
      draw();
    });

    var ctl = {
      spin: function (on) { spin = on; },
      spinning: function () { return spin; },
      relief: function (v) { relief = v; draw(); },
      exposure: function (v) { exposure = v; draw(); },
      reset: function () { yaw = 0.25; pitch = -0.13; dist = opts.radius * 3.15; spin = !reduced; draw(); }
    };
    window.viewerControls = ctl;
    return ctl;
  };

  /* Wires the standard control bar (spin and reset buttons, relief and exposure
     sliders) to a viewer, and keeps the spin button honest when the canvas is
     dragged or space is pressed. Every id is optional. Returns the sync function. */
  window.wireViewer = function (ctl, ids) {
    var btn = document.getElementById(ids.spin);
    function sync() {
      if (!btn || !ctl) return;
      var on = ctl.spinning();
      btn.textContent = on ? "Pause" : "Spin";
      btn.setAttribute("aria-pressed", String(on));
    }
    if (!ctl) return sync;
    if (btn) btn.addEventListener("click", function () { ctl.spin(!ctl.spinning()); sync(); });
    var reset = document.getElementById(ids.reset);
    if (reset) reset.addEventListener("click", function () { ctl.reset(); sync(); });
    var relief = document.getElementById(ids.relief);
    if (relief) relief.addEventListener("input", function () { ctl.relief(parseFloat(this.value)); });
    var exp = document.getElementById(ids.exposure);
    if (exp) exp.addEventListener("input", function () { ctl.exposure(parseFloat(this.value)); });
    sync();
    return sync;
  };
})();
