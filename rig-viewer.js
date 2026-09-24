/* A viewer for the rig diagram: vertex colours, flat shading, no libraries.
 *
 * Separate from the model viewer because it is doing a different job. That one
 * shows a photographed surface and needs a texture and a normal map; this one
 * shows a schematic where the colour IS the label, and flat facets read as
 * "diagram" rather than as a bad render.
 *
 * Normals are not stored. The fragment shader takes the cross product of the
 * screen-space derivatives of the world position, which gives the exact facet
 * normal for free and keeps the buffer to six floats a vertex.
 *
 * The sight-line wedges and the blind cone are drawn translucent and unsorted,
 * with depth writes off. Proper back-to-front sorting would cost a sort per
 * frame and buy very little here: the wedges mostly do not overlap, and where
 * they do, seeing both is the point.
 */
(function () {
  "use strict";

  var VERT = `#version 300 es
  in vec3 aPos; in vec3 aCol;
  uniform mat4 uProj, uView; uniform mat3 uRot;
  out vec3 vCol; out vec3 vWorld;
  void main(){
    vec3 p = uRot * aPos;
    vWorld = p; vCol = aCol;
    gl_Position = uProj * uView * vec4(p, 1.0);
  }`;

  var FRAG = `#version 300 es
  precision highp float;
  in vec3 vCol; in vec3 vWorld;
  uniform float uAlpha; uniform float uExposure;
  out vec4 frag;
  void main(){
    // the facet normal, straight from the derivatives of world position
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (!gl_FrontFacing) n = -n;
    vec3 L1 = normalize(vec3(-0.40, 0.78, 0.48));
    vec3 L2 = normalize(vec3(0.66, 0.16, -0.42));
    float k = max(dot(n, L1), 0.0) * 0.82 + max(dot(n, L2), 0.0) * 0.30 + 0.34;
    vec3 c = vCol * k * uExposure;
    frag = vec4(pow(c / (c + 0.85), vec3(1.0 / 2.2)), uAlpha);
  }`;

  function compile(gl, t, src) {
    var s = gl.createShader(t);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function perspective(f_, aspect, near, far) {
    var f = 1 / Math.tan(f_ / 2), d = near - far;
    return new Float32Array([f / aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/d,-1, 0,0,2*far*near/d,0]);
  }
  function viewAt(dist) {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-dist,1]);
  }
  function rot(yaw, pitch) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return new Float32Array([cy, sp*sy, -cp*sy, 0, cp, sp, sy, -sp*cy, cp*cy]);
  }

  window.initRig = function (opts) {
    var canvas = document.getElementById(opts.canvas);
    var note = document.getElementById(opts.note);
    var gl = canvas.getContext("webgl2", { antialias: true, alpha: true });
    if (!gl) {
      if (note) note.textContent = "This browser has no WebGL 2, so the diagram cannot be shown.";
      canvas.style.display = "none"; return null;
    }
    var prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);

    // geometry is fetched from a file; nothing draws until it has arrived
    var vao = null, stride = 24;
    fetch(opts.geometry).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.arrayBuffer();
    }).then(function (buf) {
      var v = gl.createVertexArray(); gl.bindVertexArray(v);
      var vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(buf, 0, opts.verts * 6), gl.STATIC_DRAW);
      ["aPos", "aCol"].forEach(function (n, i) {
        var loc = gl.getAttribLocation(prog, n);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, stride, i * 12);
      });
      var ibo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
                    new Uint32Array(buf, opts.verts * stride, opts.tris * 3), gl.STATIC_DRAW);
      vao = v;
      if (note) note.textContent = opts.hint;
      draw();
    }).catch(function (e) {
      if (note) note.textContent = "Could not load the diagram (" + e.message + ").";
    });

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    var uProj = gl.getUniformLocation(prog, "uProj");
    var uView = gl.getUniformLocation(prog, "uView");
    var uRot = gl.getUniformLocation(prog, "uRot");
    var uAlpha = gl.getUniformLocation(prog, "uAlpha");
    var uExp = gl.getUniformLocation(prog, "uExposure");

    var yaw = -0.62, pitch = -0.30, dist = opts.radius * 3.0, spin = true;
    var solidCount = opts.solid * 3, total = opts.tris * 3;
    var showRays = true, exposure = 1.25;

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== (w*dpr|0) || canvas.height !== (h*dpr|0)) {
        canvas.width = w*dpr|0; canvas.height = h*dpr|0;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      return canvas.width / Math.max(canvas.height, 1);
    }

    function draw() {
      var aspect = resize();
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!vao) return;
      gl.useProgram(prog); gl.bindVertexArray(vao);
      gl.uniformMatrix4fv(uProj, false, perspective(0.66, aspect, dist*0.02, dist*8));
      gl.uniformMatrix4fv(uView, false, viewAt(dist));
      gl.uniformMatrix3fv(uRot, false, rot(yaw, pitch));
      gl.uniform1f(uExp, exposure);
      // opaque hardware first, writing depth
      gl.depthMask(true);
      gl.uniform1f(uAlpha, 1.0);
      gl.drawElements(gl.TRIANGLES, solidCount, gl.UNSIGNED_INT, 0);
      // then the sight lines and the blind cone, which must not occlude
      if (showRays && total > solidCount) {
        gl.depthMask(false);
        gl.uniform1f(uAlpha, 0.17);
        gl.drawElements(gl.TRIANGLES, total - solidCount, gl.UNSIGNED_INT, solidCount * 4);
        gl.depthMask(true);
      }
    }

    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) spin = false;
    var last = 0;
    function frame(t) {
      if (spin) { yaw += (t - last) * 0.00022; draw(); }
      last = t; requestAnimationFrame(frame);
    }
    requestAnimationFrame(function (t) { last = t; draw(); frame(t); });

    var drag = null;
    function pt(e) { var t = e.touches ? e.touches[0] : e; return { x: t.clientX, y: t.clientY }; }
    canvas.addEventListener("pointerdown", function (e) { spin = false; drag = pt(e); });
    window.addEventListener("pointermove", function (e) {
      if (!drag) return; var p = pt(e);
      yaw += (p.x - drag.x) * 0.0085;
      pitch = Math.max(-1.35, Math.min(1.35, pitch + (p.y - drag.y) * 0.0075));
      drag = p; if (e.cancelable) e.preventDefault(); draw();
    });
    window.addEventListener("pointerup", function () { drag = null; });
    canvas.addEventListener("wheel", function (e) {
      dist = Math.max(opts.radius*1.1, Math.min(opts.radius*7, dist*(1 + e.deltaY*0.0012)));
      e.preventDefault(); draw();
    }, { passive: false });
    window.addEventListener("resize", draw);
    canvas.tabIndex = 0;
    canvas.addEventListener("keydown", function (e) {
      var s = 0.12;
      if (e.key === "ArrowLeft") { spin = false; yaw -= s; }
      else if (e.key === "ArrowRight") { spin = false; yaw += s; }
      else if (e.key === "ArrowUp") { spin = false; pitch = Math.max(-1.35, pitch - s); }
      else if (e.key === "ArrowDown") { spin = false; pitch = Math.min(1.35, pitch + s); }
      else if (e.key === " ") { spin = !spin; }
      else return;
      e.preventDefault(); draw();
    });
    return {
      spin: function (v) { spin = v; },
      rays: function (v) { showRays = v; draw(); },
      exposure: function (v) { exposure = v; draw(); },
      reset: function () { yaw = -0.62; pitch = -0.30; dist = opts.radius*3.0; spin = !reduced; draw(); }
    };
  };
})();
