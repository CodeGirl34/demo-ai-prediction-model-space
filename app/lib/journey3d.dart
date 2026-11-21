import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'theme.dart';

/// Minimal 3D toolkit for the Earth → Mars journey: vectors, an orbit camera,
/// a low-poly Mars 2020 spacecraft mesh and painters. Pure Flutter, no plugins,
/// so it runs the same on web, iOS and Android.

class V3 {
  const V3(this.x, this.y, this.z);
  final double x, y, z;
  static const zero = V3(0, 0, 0);
  V3 operator +(V3 o) => V3(x + o.x, y + o.y, z + o.z);
  V3 operator -(V3 o) => V3(x - o.x, y - o.y, z - o.z);
  V3 operator *(double s) => V3(x * s, y * s, z * s);
  double dot(V3 o) => x * o.x + y * o.y + z * o.z;
  V3 cross(V3 o) => V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
  double get length => math.sqrt(x * x + y * y + z * z);
  V3 get unit {
    final l = length;
    return l == 0 ? this : this * (1 / l);
  }

  static V3 lerp(V3 a, V3 b, double t) => a + (b - a) * t;
  static V3 fromList(List<dynamic> l) =>
      V3((l[0] as num).toDouble(), (l[1] as num).toDouble(), (l[2] as num).toDouble());
}

/// Orbit camera around [target]: yaw around the ecliptic pole, [elevation] above the
/// ecliptic plane (0 = edge-on, π/2 = top-down), [dist] in scene units.
class OrbitCamera {
  OrbitCamera({required this.yaw, required this.elevation, required this.dist, this.target = V3.zero});
  double yaw, elevation, dist;
  V3 target;
  double focal = 600;
  Offset center = Offset.zero;

  /// Vector in camera space: x right, y up, d depth (away from the viewer).
  (double, double, double) vec(V3 v) {
    final c = math.cos(yaw), s = math.sin(yaw);
    final x1 = v.x * c + v.y * s;
    final y1 = -v.x * s + v.y * c;
    final ce = math.cos(elevation), se = math.sin(elevation);
    return (x1, y1 * se + v.z * ce, y1 * ce - v.z * se);
  }

  /// Unit vector from the target towards the eye, in world coordinates.
  V3 get eyeDir {
    final ce = math.cos(elevation);
    return V3(ce * math.sin(yaw), -ce * math.cos(yaw), math.sin(elevation));
  }

  /// World point -> (screen offset, depth). Depth <= 0 means behind the camera.
  (Offset, double) project(V3 p) {
    final (x, y, d0) = vec(p - target);
    final d = dist + d0;
    if (d <= 1e-6) return (Offset.zero, -1);
    return (center + Offset(focal * x / d, -focal * y / d), d);
  }
}

// -----------------------------------------------------------------------------
// Spacecraft mesh: Mars 2020 cruise configuration (heat shield + backshell + cruise stage).

class Face {
  Face(this.verts, this.normal, this.color);
  final List<V3> verts; // local coordinates; forward axis = +x
  final V3 normal; // local outward normal
  final Color color;

  /// Builds a face whose normal points along [hint] (outwards), whatever the vertex order.
  factory Face.outward(List<V3> verts, V3 hint, Color color) {
    var n = (verts[1] - verts[0]).cross(verts[2] - verts[0]).unit;
    if (n.dot(hint) < 0) n = n * -1;
    return Face(verts, n, color);
  }
}

List<Face> buildSpacecraftMesh() {
  const n = 18;
  final faces = <Face>[];
  V3 ring(double a, double r, int k) {
    final t = 2 * math.pi * k / n;
    return V3(a, r * math.cos(t), r * math.sin(t));
  }

  V3 radial(int k) {
    final t = 2 * math.pi * (k + 0.5) / n;
    return V3(0, math.cos(t), math.sin(t));
  }

  const heatShield = Color(0xFF8A5A3B);
  const backshell = Color(0xFFD9DEE6);
  const cruise = Color(0xFFB8C0CC);
  const solar = Color(0xFF2E4C7A);
  const apex = V3(0.62, 0, 0);
  const fwd = V3(1, 0, 0), aft = V3(-1, 0, 0);
  for (int k = 0; k < n; k++) {
    final k2 = (k + 1) % n;
    // heat shield: shallow cone facing the direction of travel
    faces.add(Face.outward([apex, ring(0.28, 1.0, k2), ring(0.28, 1.0, k)], fwd, heatShield));
    // backshell (the rover is folded up inside): frustum narrowing towards the cruise stage
    faces.add(Face.outward(
        [ring(0.28, 1.0, k), ring(0.28, 1.0, k2), ring(-0.38, 0.56, k2), ring(-0.38, 0.56, k)], radial(k), backshell));
    // cruise stage: wide short drum
    faces.add(Face.outward(
        [ring(-0.40, 1.18, k), ring(-0.40, 1.18, k2), ring(-0.56, 1.18, k2), ring(-0.56, 1.18, k)], radial(k), cruise));
    faces.add(Face.outward(
        [ring(-0.40, 0.56, k), ring(-0.40, 0.56, k2), ring(-0.40, 1.18, k2), ring(-0.40, 1.18, k)], fwd, cruise));
    // rear face: solar arrays
    faces.add(Face.outward([const V3(-0.56, 0, 0), ring(-0.56, 1.18, k2), ring(-0.56, 1.18, k)], aft,
        k.isEven ? solar : const Color(0xFF26406A)));
  }
  return faces;
}

/// Draws the mesh at [pos] with forward axis [fwd], spun by [spin] radians around it.
/// [pixelRadius] keeps the model a fixed on-screen size whatever the zoom.
void paintSpacecraft(Canvas canvas, OrbitCamera cam, List<Face> mesh, V3 pos, V3 fwd, double spin, V3 lightDir,
    double pixelRadius) {
  final (_, depth) = cam.project(pos);
  if (depth <= 0) return;
  final scale = pixelRadius * depth / cam.focal;
  final f = fwd.unit;
  var side = const V3(0, 0, 1).cross(f);
  if (side.length < 1e-6) side = const V3(0, 1, 0);
  side = side.unit;
  final up = f.cross(side).unit;
  final cs = math.cos(spin), sn = math.sin(spin);
  V3 toWorldDir(V3 l) {
    final y = l.y * cs - l.z * sn, z = l.y * sn + l.z * cs;
    return f * l.x + side * y + up * z;
  }

  final drawn = <(double, Path, Color)>[];
  for (final face in mesh) {
    final w = [for (final v in face.verts) pos + toWorldDir(v) * scale];
    final n = toWorldDir(face.normal);
    // visible if the normal points back at the eye (camera space: eye at origin)
    final (nx, ny, nd) = cam.vec(n);
    final (px, py, pd) = cam.vec(w[0] - cam.target);
    if (nx * px + ny * py + nd * (pd + cam.dist) >= 0) continue;
    final pts = <Offset>[];
    double dsum = 0;
    for (final p in w) {
      final (o, d) = cam.project(p);
      pts.add(o);
      dsum += d;
    }
    final lit = 0.28 + 0.72 * math.max(0.0, n.dot(lightDir));
    final c = Color.lerp(Colors.black, face.color, lit)!;
    drawn.add((dsum / w.length, Path()..addPolygon(pts, true), c));
  }
  drawn.sort((a, b) => b.$1.compareTo(a.$1));
  for (final (_, path, c) in drawn) {
    canvas.drawPath(path, Paint()..color = c);
    canvas.drawPath(
        path,
        Paint()
          ..color = Colors.black.withValues(alpha: 0.25)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 0.5);
  }
}

// -----------------------------------------------------------------------------
// Scene painter

class JourneyFrame {
  JourneyFrame({
    required this.ship,
    required this.shipFwd,
    required this.earth,
    required this.mars,
    required this.path,
    required this.progress,
    required this.earthOrbit,
    required this.marsOrbit,
    required this.spin,
    this.shipPx = 10,
  });
  final V3 ship, shipFwd, earth, mars;
  final List<V3> path, earthOrbit, marsOrbit;
  final double progress; // fractional index into path
  final double spin;
  final double shipPx; // on-screen radius of the spacecraft model
}

class JourneyPainter extends CustomPainter {
  JourneyPainter({required this.cam, required this.frame, required this.mesh, required this.stars, super.repaint});
  final OrbitCamera cam;
  final JourneyFrame Function() frame;
  final List<Face> mesh;
  final List<V3> stars;

  @override
  void paint(Canvas canvas, Size size) {
    final fr = frame();
    cam.center = size.center(Offset.zero);
    cam.focal = math.min(size.width, size.height) * 1.05;
    canvas.drawRect(
        Offset.zero & size,
        Paint()
          ..shader = ui.Gradient.radial(cam.center, size.longestSide * 0.7,
              [const Color(0xFF0B1018), kVoid], [0, 1]));

    _stars(canvas);
    _grid(canvas);
    _ring(canvas, fr.earthOrbit, kObserved.withValues(alpha: 0.35));
    _ring(canvas, fr.marsOrbit, const Color(0xFFE5484D).withValues(alpha: 0.35));
    _trajectory(canvas, fr);

    // depth-sorted bodies
    final bodies = <(double, VoidCallback)>[];
    void add(V3 p, VoidCallback draw) {
      final (_, d) = cam.project(p);
      if (d > 0) bodies.add((d, draw));
    }

    add(V3.zero, () => _sun(canvas));
    add(fr.earth, () => _planet(canvas, fr.earth, 7.5, const Color(0xFF6FB7FF), const Color(0xFF1B4E8C), 'EARTH',
        atmosphere: const Color(0xFF8FD3FF)));
    add(fr.mars, () => _planet(canvas, fr.mars, 6, const Color(0xFFF08A5D), const Color(0xFF7A2A16), 'MARS'));
    add(fr.ship, () => _ship(canvas, fr));
    bodies.sort((a, b) => b.$1.compareTo(a.$1));
    for (final (_, draw) in bodies) {
      draw();
    }
  }

  double _sizeAt(V3 p, double px) {
    final (_, d) = cam.project(p);
    return d <= 0 ? 0 : (px * cam.dist / d).clamp(px * 0.45, px * 2.2);
  }

  void _stars(Canvas canvas) {
    final paint = Paint();
    for (int i = 0; i < stars.length; i++) {
      final (o, d) = cam.project(stars[i] + cam.target);
      if (d <= 0) continue;
      final a = 0.25 + 0.55 * ((i * 37) % 100) / 100;
      paint.color = Colors.white.withValues(alpha: a);
      canvas.drawCircle(o, 0.5 + ((i * 13) % 10) / 10, paint);
    }
  }

  void _grid(Canvas canvas) {
    final p = Paint()
      ..color = kObserved.withValues(alpha: 0.07)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    for (final r in [0.5, 1.0, 1.5, 2.0]) {
      final path = Path();
      for (int k = 0; k <= 96; k++) {
        final t = 2 * math.pi * k / 96;
        final (o, d) = cam.project(V3(r * math.cos(t), r * math.sin(t), 0));
        if (d <= 0) continue;
        k == 0 ? path.moveTo(o.dx, o.dy) : path.lineTo(o.dx, o.dy);
      }
      canvas.drawPath(path, p);
    }
    for (int k = 0; k < 12; k++) {
      final t = 2 * math.pi * k / 12;
      final (a, da) = cam.project(V3(0.25 * math.cos(t), 0.25 * math.sin(t), 0));
      final (b, db) = cam.project(V3(2.0 * math.cos(t), 2.0 * math.sin(t), 0));
      if (da > 0 && db > 0) canvas.drawLine(a, b, p);
    }
    final (l, dl) = cam.project(const V3(1.0, 0, 0));
    if (dl > 0) _label(canvas, l + const Offset(4, 2), '1 AU', kObserved.withValues(alpha: 0.35), size: 8.5);
  }

  void _ring(Canvas canvas, List<V3> pts, Color color) {
    final path = Path();
    bool started = false;
    for (final v in pts) {
      final (o, d) = cam.project(v);
      if (d <= 0) {
        started = false;
        continue;
      }
      started ? path.lineTo(o.dx, o.dy) : path.moveTo(o.dx, o.dy);
      started = true;
    }
    canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1);
  }

  void _trajectory(Canvas canvas, JourneyFrame fr) {
    final upTo = fr.progress.floor();
    final future = Path(), past = Path();
    for (int i = 0; i < fr.path.length; i++) {
      final (o, d) = cam.project(fr.path[i]);
      if (d <= 0) continue;
      if (i == 0) {
        future.moveTo(o.dx, o.dy);
        past.moveTo(o.dx, o.dy);
      } else {
        future.lineTo(o.dx, o.dy);
        if (i <= upTo) past.lineTo(o.dx, o.dy);
      }
    }
    final (shipO, sd) = cam.project(fr.ship);
    if (sd > 0) past.lineTo(shipO.dx, shipO.dy);
    canvas.drawPath(
        future,
        Paint()
          ..color = kSignal.withValues(alpha: 0.22)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1);
    canvas.drawPath(
        past,
        Paint()
          ..color = kSignal.withValues(alpha: 0.35)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 5
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4));
    canvas.drawPath(
        past,
        Paint()
          ..color = kSignal
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.8);
    // launch and arrival markers
    final (a, da) = cam.project(fr.path.first);
    final (b, db) = cam.project(fr.path.last);
    final mark = Paint()
      ..color = kSignal
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    if (da > 0) canvas.drawRect(Rect.fromCenter(center: a, width: 6, height: 6), mark);
    if (db > 0) canvas.drawRect(Rect.fromCenter(center: b, width: 6, height: 6), mark);
  }

  void _sun(Canvas canvas) {
    final (o, d) = cam.project(V3.zero);
    if (d <= 0) return;
    final r = _sizeAt(V3.zero, 13);
    canvas.drawCircle(
        o,
        r * 4,
        Paint()
          ..shader = ui.Gradient.radial(
              o, r * 4, [const Color(0x66FFB224), const Color(0x22FF8A00), const Color(0x00FF8A00)], [0, 0.35, 1]));
    canvas.drawCircle(
        o,
        r,
        Paint()
          ..shader =
              ui.Gradient.radial(o - Offset(r * 0.3, r * 0.3), r * 1.3, [const Color(0xFFFFF4D6), const Color(0xFFFFB224)]));
    _label(canvas, o + Offset(r + 6, -r - 4), 'SUN', kSignal.withValues(alpha: 0.8));
  }

  void _planet(Canvas canvas, V3 p, double px, Color light, Color dark, String name, {Color? atmosphere}) {
    final (o, d) = cam.project(p);
    if (d <= 0) return;
    final r = _sizeAt(p, px);
    // light comes from the Sun (origin): find its direction on screen
    final l = (V3.zero - p).unit;
    final (lx, ly, ld) = cam.vec(l);
    final dir = Offset(lx, -ly);
    final facing = (1 - ld) / 2; // 1 = lit side faces us, 0 = we see the night side
    final hi = dir.distance < 1e-6 ? Offset.zero : dir / dir.distance * (r * 0.55 * math.sqrt(1 - facing * facing + 1e-9));
    if (atmosphere != null) {
      canvas.drawCircle(
          o,
          r * 1.7,
          Paint()
            ..shader = ui.Gradient.radial(
                o, r * 1.7, [atmosphere.withValues(alpha: 0.25 * (0.3 + facing)), atmosphere.withValues(alpha: 0)]));
    }
    canvas.drawCircle(
        o,
        r,
        Paint()
          ..shader = ui.Gradient.radial(o + hi, r * (1.0 + facing),
              [Color.lerp(dark, light, 0.35 + 0.65 * facing)!, dark, const Color(0xFF05070A)], [0, 0.6, 1]));
    _label(canvas, o + Offset(r + 6, -r - 4), name, kInkDim);
  }

  void _ship(Canvas canvas, JourneyFrame fr) {
    final (o, d) = cam.project(fr.ship);
    if (d <= 0) return;
    final glow = fr.shipPx * 2.4;
    canvas.drawCircle(
        o,
        glow,
        Paint()
          ..shader = ui.Gradient.radial(o, glow, [kSignal.withValues(alpha: 0.28), kSignal.withValues(alpha: 0)]));
    paintSpacecraft(canvas, cam, mesh, fr.ship, fr.shipFwd, fr.spin, (V3.zero - fr.ship).unit, fr.shipPx);
    _label(canvas, o + Offset(fr.shipPx + 6, fr.shipPx * 0.6), 'PERSEVERANCE', kSignal);
  }

  void _label(Canvas canvas, Offset at, String text, Color color, {double size = 9.5}) {
    final tp = TextPainter(
      text: TextSpan(text: text, style: consoleLabel(color: color, size: size, weight: FontWeight.w600)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, at);
  }

  @override
  bool shouldRepaint(covariant JourneyPainter old) => true;
}

/// Close-up of the spacecraft model, slowly turning.
class SpacecraftCloseupPainter extends CustomPainter {
  SpacecraftCloseupPainter({required this.mesh, required this.angle, super.repaint});
  final List<Face> mesh;
  final double Function() angle;

  @override
  void paint(Canvas canvas, Size size) {
    final a = angle();
    final cam = OrbitCamera(yaw: a, elevation: 0.42, dist: 8)
      ..center = size.center(Offset.zero)
      ..focal = math.min(size.width, size.height) * 1.6;
    canvas.drawRect(
        Offset.zero & size,
        Paint()
          ..shader = ui.Gradient.radial(
              cam.center, size.shortestSide * 0.8, [const Color(0xFF111A26), kInset], [0, 1]));
    // key light plus a headlight from the viewer, so the side we look at is never black
    final light = (const V3(0.6, -0.5, 0.65).unit * 0.5 + cam.eyeDir * 0.8).unit;
    paintSpacecraft(canvas, cam, mesh, V3.zero, const V3(1, 0, 0), a * 3, light, size.shortestSide * 0.3);
  }

  @override
  bool shouldRepaint(covariant SpacecraftCloseupPainter old) => true;
}

/// Deterministic starfield on a large sphere.
List<V3> makeStars(int n, {double radius = 40}) {
  final rnd = math.Random(7);
  return [
    for (int i = 0; i < n; i++)
      () {
        final z = rnd.nextDouble() * 2 - 1;
        final t = rnd.nextDouble() * 2 * math.pi;
        final r = math.sqrt(1 - z * z);
        return V3(r * math.cos(t) * radius, r * math.sin(t) * radius, z * radius);
      }(),
  ];
}
