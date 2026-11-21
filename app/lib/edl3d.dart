import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'journey3d.dart' show V3, OrbitCamera;
import 'theme.dart';

/// Mars entry, descent and landing (EDL) in 3D.
///
/// The event sequence follows NASA's Mars 2020 EDL concept of operations ("Mars Entry, Descent,
/// and Landing", NASA Moon to Mars Architecture, 2025). Times come from the reconstructed landing
/// trajectory where it has them; the rest are marked approximate or estimated.

// =============================================================================
// Timeline

enum EdlSource {
  data('NASA data', kObserved),
  approx('approx.', kInkDim),
  estimate('estimate', kDerived);

  const EdlSource(this.label, this.color);
  final String label;
  final Color color;
}

/// The paper's three EDL phases, plus the coast before entry.
enum EdlPhase {
  approach('Approach', Color(0xFF8A94A3),
      'Coasting toward Mars, heat shield first. The cruise stage that flew it from Earth is already gone.'),
  entry('Entry', Color(0xFFE5484D),
      'The thin atmosphere brakes and heats the capsule. The heat shield takes the heat, and small thrusters bank the capsule to steer toward the target.'),
  descent('Descent', Color(0xFF56D8F4),
      'A supersonic parachute slows it down, then rocket engines take over. Radar and a camera find the ground and a safe place to land.'),
  landing('Landing', Color(0xFF4FD08A),
      'The sky crane hovers and lowers the rover on cables, cuts them at touchdown, and flies away.');

  const EdlPhase(this.label, this.color, this.what);
  final String label;
  final Color color;
  final String what;
}

class EdlEvent {
  const EdlEvent(this.id, this.name, this.t, this.source, this.what);
  final String id, name, what;
  final double t; // seconds relative to touchdown
  final EdlSource source;
}

class EdlTimeline {
  EdlTimeline._(this.mission, this.events, this.t, this.alt, this.speed, this.heat, this.flow);

  final String mission;
  final List<EdlEvent> events;
  final List<double> t, alt, speed;
  final List<double> heat; // 0..1, estimated heating rate (Sutton–Graves)
  final List<double> flow; // 0..1, compressed dynamic pressure, drives the airflow streaks

  static const scaleHeightKm = 11.1; // Mars atmosphere, exponential model
  static const epilogueS = 14.0; // animation shown after the data ends (sky crane flyaway)

  /// [missionEvents]: `/api/missions` events; [traj]: `/api/missions/{id}/trajectory`.
  factory EdlTimeline.build(String mission, List<dynamic> missionEvents, Map<String, dynamic> traj) {
    List<double> nums(String k) =>
        [for (final x in traj[k] as List) x == null ? double.nan : (x as num).toDouble()];
    final t = nums('t_s'), alt = nums('altitude_km'), v = nums('speed_m_s'), acc = nums('sensed_accel_m_s2');
    final ev = {for (final e in missionEvents) (e as Map)['event'] as String: e};
    double te(String k) => ((ev[k] as Map)['t_s'] as num).toDouble();
    double ve(String k) => ((ev[k] as Map)['speed_m_s'] as num).toDouble();

    // Heating ~ sqrt(density) * v^3 (Sutton–Graves); dynamic pressure ~ density * v^2.
    const h = scaleHeightKm;
    final qHeat = <double>[], qDyn = <double>[];
    for (int i = 0; i < t.length; i++) {
      final ok = alt[i].isFinite && v[i].isFinite && alt[i] < 150;
      qHeat.add(ok ? math.exp(-alt[i] / (2 * h)) * v[i] * v[i] * v[i] : 0);
      qDyn.add(ok ? math.exp(-alt[i] / h) * v[i] * v[i] : 0);
    }
    final hMax = qHeat.reduce(math.max), dMax = qDyn.reduce(math.max);
    final heat = [for (final q in qHeat) hMax > 0 ? q / hMax : 0.0];
    final flow = [for (final q in qDyn) dMax > 0 ? math.pow(q / dMax, 0.35).toDouble() : 0.0];

    int argmax(List<double> xs) {
      int best = 0;
      for (int i = 1; i < xs.length; i++) {
        if (xs[i].isFinite && (!xs[best].isFinite || xs[i] > xs[best])) best = i;
      }
      return best;
    }

    int firstWhere(bool Function(int i) test, int fallback) {
      for (int i = 0; i < t.length; i++) {
        if (test(i)) return i;
      }
      return fallback;
    }

    final tEntry = te('entry_interface'), tChute = te('parachute_deploy'), tShield = te('heatshield_sep');
    final tBack = te('backshell_sep'), tRover = te('rover_sep'), tDown = te('touchdown');
    final iHeat = argmax(heat), iDecel = argmax(acc);
    final iGuided = firstWhere((i) => t[i] > tEntry && acc[i] >= 0.2 * 9.80665, iDecel);
    final trn = mission == 'm2020'; // Lander Vision System flew on Perseverance, not Curiosity
    final iTrn = firstWhere((i) => t[i] > tShield && alt[i] <= 4.2, t.length - 1);
    final tRadar = tShield + ((trn ? t[iTrn] : tBack) - tShield) * 0.35;
    final g = acc[iDecel] / 9.80665;

    final events = <EdlEvent>[
      EdlEvent('cruise_sep', 'Cruise stage separation', tEntry - 600, EdlSource.approx,
          'The cruise stage that flew it from Earth is let go about 10 minutes before entry, just before this data starts.'),
      EdlEvent('entry', 'Atmospheric entry', tEntry, EdlSource.data,
          'Reaches the top of the atmosphere at ${(ve('entry_interface') / 1000).toStringAsFixed(1)} km/s, heat shield first.'),
      EdlEvent('guided', 'Guided entry', t[iGuided], EdlSource.approx,
          'Small thrusters roll the capsule to aim its lift at the landing site. Placed where braking first reaches 0.2 g.'),
      EdlEvent('peak_heat', 'Peak heating', t[iHeat], EdlSource.estimate,
          'The heat shield is at its hottest and the air in front of it glows. Time estimated from speed and air density.'),
      EdlEvent('peak_decel', 'Peak deceleration', t[iDecel], EdlSource.data,
          'Braking peaks at ${g.toStringAsFixed(1)} g: ${g.toStringAsFixed(0)} times Earth’s gravity.'),
      EdlEvent('parachute', 'Parachute deployment', tChute, EdlSource.data,
          'A 21.5 m parachute opens at ${ve('parachute_deploy').toStringAsFixed(0)} m/s, faster than sound.'),
      EdlEvent('heatshield', 'Heat shield separation', tShield, EdlSource.data,
          'The heat shield drops away so the radar and camera can see the ground.'),
      EdlEvent('radar', 'Radar lock', tRadar, EdlSource.approx,
          'The radar locks onto the ground and measures height and speed. Time not in our data.'),
      if (trn)
        EdlEvent('trn', 'Terrain-relative navigation', t[iTrn], EdlSource.approx,
            'A camera photographs the ground and matches it to an onboard map to pick a safe spot. Starts near 4.2 km.'),
      EdlEvent('backshell', 'Back shell and parachute separation', tBack, EdlSource.data,
          'The descent stage, carrying the rover, drops out of the back shell.'),
      EdlEvent('powered', 'Powered descent', tBack + 1, EdlSource.approx,
          'Eight rocket engines slow it down and steer it away from the falling back shell.'),
      EdlEvent('rover_dep', 'Rover deployment (sky crane)', tRover, EdlSource.data,
          'The descent stage hovers and lowers the rover on 7.5 m cables.'),
      EdlEvent('touchdown', 'Touchdown', tDown, EdlSource.data,
          'Wheels on the ground at ${ve('touchdown').toStringAsFixed(1)} m/s. The cables are cut.'),
      EdlEvent('flyaway', 'Sky crane flies away', tDown + 1, EdlSource.approx,
          'The descent stage flies off to crash a safe distance away. Shown after the data ends.'),
    ]..sort((a, b) => a.t.compareTo(b.t));
    return EdlTimeline._(mission, events, t, alt, v, heat, flow);
  }

  double get tStart => t.first;
  double get tEnd => t.last;
  bool has(String id) => events.any((e) => e.id == id);
  double at(String id) => events.firstWhere((e) => e.id == id).t;

  EdlPhase phaseAt(double now) {
    if (now < at('entry')) return EdlPhase.approach;
    if (now < at('parachute')) return EdlPhase.entry;
    if (now < at('rover_dep')) return EdlPhase.descent;
    return EdlPhase.landing;
  }

  /// Index of the last event at or before [now]; -1 before the first.
  int currentIndex(double now) {
    int k = -1;
    for (int i = 0; i < events.length; i++) {
      if (events[i].t <= now + 1e-6) k = i;
    }
    return k;
  }

  double _interp(List<double> ys, double x) {
    if (x <= t.first) return ys.first;
    if (x >= t.last) return ys.last;
    int lo = 0, hi = t.length - 1;
    while (hi - lo > 1) {
      final m = (lo + hi) >> 1;
      if (t[m] <= x) {
        lo = m;
      } else {
        hi = m;
      }
    }
    return ys[lo] + (ys[hi] - ys[lo]) * (x - t[lo]) / (t[hi] - t[lo]);
  }

  double altKm(double now) => _interp(alt, now);
  double speedAt(double now) => _interp(speed, now);
  double heatAt(double now) => _interp(heat, now);
  double flowAt(double now) => _interp(flow, now);
}

// =============================================================================
// Geometry: polygons and segments in metres. Vehicle-local frame: +x points "down" (heat-shield
// side), y and z across. World frame of the close-up: z up.

class Poly {
  const Poly(this.v, this.n, this.color, {this.twoSided = false, this.glow = 0});
  final List<V3> v;
  final V3 n; // outward normal
  final Color color;
  final bool twoSided;
  final double glow; // 0..1, blends towards incandescent orange
}

class Seg {
  const Seg(this.a, this.b, this.color, {this.width = 1, this.glow = false});
  final V3 a, b;
  final Color color;
  final double width;
  final bool glow;
}

V3 _anyPerp(V3 a) => a.cross(a.x.abs() < 0.9 ? const V3(1, 0, 0) : const V3(0, 1, 0)).unit;

/// Side of a cone frustum from (c0, r0) to (c1, r1); [color] per sector.
List<Poly> frustum(V3 c0, double r0, V3 c1, double r1,
    {required Color Function(int k) color, int n = 20, bool twoSided = false}) {
  final ax = c1 - c0, h = ax.length, a = ax.unit;
  final u = _anyPerp(a), w = a.cross(u);
  V3 dir(double th) => u * math.cos(th) + w * math.sin(th);
  V3 p(V3 c, double r, int k) => c + dir(2 * math.pi * k / n) * r;
  final slope = (r1 - r0) / h;
  return [
    for (int k = 0; k < n; k++)
      Poly([p(c0, r0, k), p(c0, r0, k + 1), p(c1, r1, k + 1), p(c1, r1, k)],
          (dir(2 * math.pi * (k + 0.5) / n) - a * slope).unit, color(k),
          twoSided: twoSided),
  ];
}

Poly disc(V3 c, double r, V3 normal, Color color, {int n = 20, bool twoSided = false}) {
  final a = normal.unit, u = _anyPerp(a), w = a.cross(u);
  return Poly([for (int k = 0; k < n; k++) c + (u * math.cos(2 * math.pi * k / n) + w * math.sin(2 * math.pi * k / n)) * r],
      a, color, twoSided: twoSided);
}

List<Poly> box(V3 lo, V3 hi, Color c) {
  V3 p(int i, int j, int k) => V3(i == 0 ? lo.x : hi.x, j == 0 ? lo.y : hi.y, k == 0 ? lo.z : hi.z);
  return [
    Poly([p(0, 0, 0), p(0, 1, 0), p(0, 1, 1), p(0, 0, 1)], const V3(-1, 0, 0), c),
    Poly([p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(1, 1, 0)], const V3(1, 0, 0), c),
    Poly([p(0, 0, 0), p(0, 0, 1), p(1, 0, 1), p(1, 0, 0)], const V3(0, -1, 0), c),
    Poly([p(0, 1, 0), p(1, 1, 0), p(1, 1, 1), p(0, 1, 1)], const V3(0, 1, 0), c),
    Poly([p(0, 0, 0), p(1, 0, 0), p(1, 1, 0), p(0, 1, 0)], const V3(0, 0, -1), c),
    Poly([p(0, 0, 1), p(0, 1, 1), p(1, 1, 1), p(1, 0, 1)], const V3(0, 0, 1), c),
  ];
}

/// Spherical cap around the local x axis with its apex towards -x (a parachute canopy).
List<Poly> dome(double xc, double rs, double phiMax, {required Color Function(int k) color, int rings = 4, int n = 24}) {
  final c = V3(xc, 0, 0);
  V3 p(int i, int k) {
    final phi = phiMax * i / rings, th = 2 * math.pi * k / n;
    return V3(xc - rs * math.cos(phi), rs * math.sin(phi) * math.cos(th), rs * math.sin(phi) * math.sin(th));
  }

  return [
    for (int i = 0; i < rings; i++)
      for (int k = 0; k < n; k++)
        () {
          final v = [p(i, k), p(i, k + 1), p(i + 1, k + 1), p(i + 1, k)];
          final mid = (v[0] + v[2]) * 0.5;
          return Poly(v, (mid - c).unit, color(k), twoSided: true);
        }(),
  ];
}

const _orange = Color(0xFFE8742A), _white = Color(0xFFEDEDE8);

/// Simplified Mars 2020 / MSL flight system (both use the same EDL architecture).
class EdlMeshes {
  EdlMeshes._();
  static final EdlMeshes instance = EdlMeshes._();

  // aeroshell: 4.5 m, 70° sphere-cone heat shield and a conical back shell
  final heatShield = [
    ...frustum(const V3(0.82, 0, 0), 0.001, V3.zero, 2.25, color: (_) => const Color(0xFF6B4A35), n: 24),
    disc(V3.zero, 2.25, const V3(-1, 0, 0), const Color(0xFF3A3A3E), n: 24),
  ];
  final backShell = [
    ...frustum(V3.zero, 2.25, const V3(-1.55, 0, 0), 1.25, color: (_) => const Color(0xFFD9DEE6), n: 24, twoSided: true),
    ...frustum(const V3(-1.55, 0, 0), 1.25, const V3(-2.05, 0, 0), 0.75,
        color: (_) => const Color(0xFFC4CAD3), n: 24, twoSided: true),
    disc(const V3(-2.05, 0, 0), 0.75, const V3(-1, 0, 0), const Color(0xFFAEB5BF), n: 24),
  ];
  final cruiseStage = [
    ...frustum(const V3(-2.1, 0, 0), 2.0, const V3(-2.5, 0, 0), 2.0, color: (_) => const Color(0xFFB8C0CC), n: 24),
    disc(const V3(-2.1, 0, 0), 2.0, const V3(1, 0, 0), const Color(0xFF8E96A2), n: 24),
    disc(const V3(-2.5, 0, 0), 2.0, const V3(-1, 0, 0), const Color(0xFF2E4C7A), n: 24),
  ];

  // descent stage (sky crane): octagonal body, 8 engines in 4 pairs
  late final engineExits = [
    for (final base in const [45.0, 135.0, 225.0, 315.0])
      for (final d in const [-11.0, 11.0])
        V3(-0.68, 1.36 * math.cos((base + d) * math.pi / 180), 1.36 * math.sin((base + d) * math.pi / 180)),
  ];
  late final descentStage = [
    ...frustum(const V3(-1.45, 0, 0), 1.25, const V3(-0.88, 0, 0), 1.25, color: (_) => const Color(0xFFC79A3A), n: 8),
    disc(const V3(-1.45, 0, 0), 1.25, const V3(-1, 0, 0), const Color(0xFF8F6F2A), n: 8),
    disc(const V3(-0.88, 0, 0), 1.25, const V3(1, 0, 0), const Color(0xFF6E5A35), n: 8),
    for (final e in engineExits)
      ...frustum(V3(-0.92, e.y, e.z), 0.07, V3(-0.68, e.y, e.z), 0.15, color: (_) => const Color(0xFF555A61), n: 8),
  ];

  // rover: body, six wheels, power source at the back
  final rover = [
    ...box(const V3(-0.80, -1.30, -0.90), const V3(-0.32, 1.30, 0.90), const Color(0xFFE8E6DE)),
    for (final y in const [-1.0, 0.0, 1.0])
      for (final s in const [-1.0, 1.0]) ...[
        ...frustum(V3(-0.06, y, s * 0.92), 0.26, V3(-0.06, y, s * 1.16), 0.26,
            color: (_) => const Color(0xFF3A3F47), n: 10),
        disc(V3(-0.06, y, s * 1.16), 0.26, V3(0, 0, s), const Color(0xFF5B6068), n: 10),
      ],
    ...frustum(const V3(-0.55, -1.30, 0), 0.22, const V3(-0.85, -1.75, 0), 0.20,
        color: (_) => const Color(0xFF4A4F57), n: 10),
  ];

  // disc-gap-band parachute, drawn smaller and closer than the real 21.5 m one
  final canopy = [
    ...dome(-8.5, 6.0, 50 * math.pi / 180, color: (k) => (k ~/ 2) % 3 == 0 ? _orange : _white),
    ...frustum(const V3(-11.96, 0, 0), 4.6, const V3(-10.76, 0, 0), 4.6,
        color: (k) => k.isEven ? _orange : _white, n: 24, twoSided: true),
  ];
  late final canopyLines = [
    for (int k = 0; k < 16; k++)
      (V3(-10.76, 4.6 * math.cos(2 * math.pi * k / 16), 4.6 * math.sin(2 * math.pi * k / 16)), const V3(-3.4, 0, 0)),
    (const V3(-3.4, 0, 0), const V3(-2.05, 0, 0)),
  ];
}

// =============================================================================
// Scene at a moment of the landing

class EdlScene {
  final polys = <Poly>[];
  final segs = <Seg>[];
  final labels = <(V3, String)>[];
  V3 lo = const V3(1e9, 1e9, 1e9), hi = const V3(-1e9, -1e9, -1e9);
  double? groundZ;
  double altKm = 0, heat = 0, flow = 0, dust = 0;
  V3? shieldApex;

  /// Wind in m/s (x = east, y = north, z = up), for the streaks blowing across the view.
  V3 wind = V3.zero;

  void grow(V3 p) {
    lo = V3(math.min(lo.x, p.x), math.min(lo.y, p.y), math.min(lo.z, p.z));
    hi = V3(math.max(hi.x, p.x), math.max(hi.y, p.y), math.max(hi.z, p.z));
  }

  V3 get center => (lo + hi) * 0.5;
  double get radius => math.max(2.6, (hi - lo).length * 0.5);
}

double _clamp01(double x) => x < 0 ? 0 : (x > 1 ? 1 : x);
double _smooth(double x) {
  final c = _clamp01(x);
  return c * c * (3 - 2 * c);
}

/// Builds the flight system as it is configured at [now] (seconds to touchdown; can run past the
/// end of the data for the flyaway). [clock] is wall time, for flicker and airflow.
/// [wind] and [swing] come from a wind gust: [swing] (east, north, in radians) tilts the whole
/// stack the way the parachute line or the thrust axis leans in the gust.
EdlScene buildEdlScene(EdlTimeline tl, double now, double clock, {V3 wind = V3.zero, (double, double) swing = (0, 0)}) {
  final m = EdlMeshes.instance;
  final s = EdlScene()..wind = wind;
  final tE = tl.at('entry'), tPd = tl.at('parachute'), tHs = tl.at('heatshield'), tBs = tl.at('backshell');
  final tRs = tl.at('rover_dep'), tTd = tl.at('touchdown'), tRad = tl.at('radar');
  final tTrn = tl.has('trn') ? tl.at('trn') : null;
  final dataNow = math.min(now, tl.tEnd);
  final altM = now >= tTd ? 0.0 : tl.altKm(dataNow) * 1000;
  s.altKm = tl.altKm(dataNow);
  s.heat = now < tHs ? tl.heatAt(dataNow) : 0;
  s.flow = now < tBs ? tl.flowAt(dataNow) : 0;

  // capsule flies at an angle of attack during entry (lift for steering), upright under the chute
  final tilt = 0.28 * _smooth((now - tE) / 20) * (1 - _smooth((now - tPd) / 3));
  final ct = math.cos(tilt), st = math.sin(tilt);
  V3 toW0(V3 l) {
    final x = l.z, y = l.y, z = -l.x;
    return V3(x * ct + z * st, y, -x * st + z * ct);
  }

  // gust swing: rotate about the horizontal axis at right angles to the lean (Rodrigues)
  final (se, sn) = swing;
  final ang = math.atan(math.sqrt(se * se + sn * sn));
  final axis = ang > 1e-4 ? V3(-sn, se, 0).unit : V3.zero;
  final ca = math.cos(ang), sa = math.sin(ang);
  V3 toW(V3 l) {
    final p = toW0(l);
    if (ang <= 1e-4) return p;
    return p * ca + axis.cross(p) * sa + axis * (axis.dot(p) * (1 - ca));
  }

  void part(List<Poly> mesh, V3 off, {bool focus = true, double glow = 0, V3 Function(V3)? pre}) {
    for (final p in mesh) {
      final vs = [for (final v in p.v) toW((pre == null ? v : pre(v)) + off)];
      s.polys.add(Poly(vs, toW(p.n), p.color, twoSided: p.twoSided, glow: glow));
      if (focus) vs.forEach(s.grow);
    }
  }

  void seg(V3 a, V3 b, Color c, {double width = 1, bool glow = false}) =>
      s.segs.add(Seg(toW(a), toW(b), c, width: width, glow: glow));

  // cruise stage drifting away (it separated about a minute before the data starts)
  final cs = now - tl.tStart;
  if (cs < 30) {
    final off = V3(-(2.5 + 0.7 * cs), 0.2 * cs, 0);
    part(m.cruiseStage, off, focus: false);
    if (cs < 22) s.labels.add((toW(off + const V3(-2.3, 0, 2.2)), 'CRUISE STAGE'));
  }

  // heat shield: glows during entry, drops away at separation
  if (now < tHs) {
    part(m.heatShield, V3.zero, glow: s.heat);
    s.shieldApex = toW(const V3(0.9, 0, 0));
  } else if (now - tHs < 5) {
    final u = now - tHs;
    final off = V3(1.5 * u + 4.5 * u * u, 0.4 * u, 0);
    part(m.heatShield, off, focus: false);
    if (u < 4) s.labels.add((toW(off + const V3(0.6, 0, 2.4)), 'HEAT SHIELD'));
  }

  // back shell with the parachute, released at back shell separation
  V3? backOff;
  if (now < tBs) {
    backOff = V3.zero;
  } else if (now - tBs < 7) {
    final u = now - tBs;
    backOff = V3(-(3 * u + 4 * u * u), 1.5 * u * u, 0);
  }
  if (backOff != null) {
    final attached = now < tBs;
    part(m.backShell, backOff, focus: attached);
    if (now >= tPd) {
      final f = 0.15 + 0.85 * _smooth((now - tPd) / 1.2); // canopy inflating
      V3 inflate(V3 p) => V3(-2.05 + (p.x + 2.05) * (0.5 + 0.5 * f), p.y * f, p.z * f);
      part(m.canopy, backOff, focus: attached, pre: inflate);
      for (final (a, b) in m.canopyLines) {
        seg(inflate(a) + backOff, inflate(b) + backOff, const Color(0xFFBFC3C9), width: 0.8);
      }
      final lp = toW(inflate(const V3(-16.0, 0, 0)) + backOff); // just above the canopy apex
      if (attached) s.grow(lp); // keep the label inside the frame
      s.labels.add((lp, attached ? 'PARACHUTE' : 'BACK SHELL + PARACHUTE'));
    }
  }

  // descent stage: inside the back shell until separation, flies away after touchdown
  V3? dsOff;
  if (now < tTd + 0.4) {
    dsOff = V3.zero;
  } else if (now - tTd - 0.4 < 9) {
    final u = now - tTd - 0.4;
    dsOff = V3(-(2.5 * u + 1.2 * u * u), 2.0 * u * u, 0);
  }
  if (dsOff != null) part(m.descentStage, dsOff, focus: now < tTd + 0.4);

  // rover: stowed under the descent stage, lowered on 7.5 m bridles by the sky crane
  final ext = now < tRs ? 0.0 : math.min(7.5, 7.5 * (now - tRs) / 6.5);
  final roverOff = V3(ext, 0, 0);
  part(m.rover, roverOff);
  if (now > tTd + 2) {
    // camera mast rises once the rover is down
    final h = 1.3 * _smooth((now - tTd - 2) / 2.5);
    part(box(V3(-0.80 - h, 0.85, -0.07), const V3(-0.80, 0.99, 0.07), const Color(0xFFCFCFC8)), roverOff);
    part(box(V3(-1.02 - h, 0.78, -0.22), V3(-0.80 - h, 1.06, 0.22), const Color(0xFFDADAD2)), roverOff);
  }
  if (now >= tRs && now < tTd + 0.4 && ext > 0.05) {
    for (int k = 0; k < 3; k++) {
      final th = 2 * math.pi * k / 3;
      seg(V3(-0.88, 0.45 * math.cos(th), 0.45 * math.sin(th)),
          V3(ext - 0.80, 0.35 * math.cos(th), 0.35 * math.sin(th)), const Color(0xFFD4D4D4));
    }
    seg(const V3(-0.88, 0.1, 0.6), V3(ext - 0.80, 0.1, 0.5), const Color(0xFFC9A64A), width: 1.2);
  }
  if (now >= tRs) {
    s.labels.add((toW(roverOff + const V3(-0.2, 0, 1.6)), 'ROVER'));
    if (dsOff != null) s.labels.add((toW(dsOff + const V3(-1.6, 0, 1.6)), 'DESCENT STAGE'));
  }

  // engine plumes: powered descent, sky crane and flyaway
  if (dsOff != null && now >= tBs + 0.5) {
    final flyaway = now > tTd + 0.4;
    for (int k = 0; k < m.engineExits.length; k++) {
      final e = m.engineExits[k] + dsOff;
      final len = (flyaway ? 3.2 : 2.3) + 0.35 * math.sin(clock * 31 + k * 1.7);
      seg(e, e + V3(len, e.y * 0.12 * len, e.z * 0.12 * len), const Color(0xFFFFC27A), width: 3, glow: true);
    }
  }

  // radar beams from heat shield separation to touchdown
  if (now >= tRad && now < tTd) {
    final pulse = 0.5 + 0.5 * math.sin(clock * 6);
    for (int k = 0; k < 6; k++) {
      final th = 2 * math.pi * k / 6 + 0.3;
      final d = V3(1, 0.42 * math.cos(th), 0.42 * math.sin(th)).unit;
      seg(const V3(-0.85, 0, 0), d * (7 + 2 * pulse), kObserved.withValues(alpha: 0.35 + 0.25 * pulse),
          width: 1.4, glow: true);
    }
  }

  // terrain-relative navigation camera (Perseverance)
  if (tTrn != null && now >= tTrn && now < tBs + 3) {
    final cam = roverOff + const V3(-0.32, 1.25, 0);
    for (final (dy, dz) in const [(-3.0, -3.0), (-3.0, 3.0), (3.0, 3.0), (3.0, -3.0)]) {
      seg(cam, cam + V3(10, dy + 1.5, dz), kDerived.withValues(alpha: 0.55), width: 1, glow: true);
    }
    s.labels.add((toW(cam + const V3(6, 3, 2.5)), 'TRN CAMERA'));
  }

  // ground: visible in the last ~150 m
  final roverBottom = ext + 0.20;
  if (altM < 150 || now >= tTd) {
    var gx = now >= tTd ? roverBottom : math.max(altM, roverBottom);
    if (now < tTd) gx = gx + (roverBottom - gx) * _smooth((now - (tTd - 2)) / 2);
    s.groundZ = toW0(V3(gx, 0, 0)).z; // the ground does not swing
    if (gx < 12) s.grow(V3(0, 0, s.groundZ!));
    if (dsOff != null && now >= tBs + 0.5) s.dust = _clamp01(1 - (gx - dsOff.x) / 30); // engines' height above the ground
  }
  return s;
}

// =============================================================================
// Painter

class EdlPainter extends CustomPainter {
  EdlPainter({required this.cam, required this.scene, required this.clock, super.repaint});
  final OrbitCamera cam;
  final EdlScene Function() scene;
  final double Function() clock;

  static final _stars = () {
    final r = math.Random(11);
    return [for (int i = 0; i < 70; i++) (r.nextDouble(), r.nextDouble(), 0.3 + 0.7 * r.nextDouble())];
  }();

  @override
  void paint(Canvas canvas, Size size) {
    final s = scene();
    final t = clock();
    cam.center = size.center(Offset.zero);
    cam.focal = math.min(size.width, size.height) * 1.25;
    _sky(canvas, size, s.altKm);
    if (s.heat > 0.02 && s.shieldApex != null) _plasma(canvas, s);
    if (s.flow > 0.03) _airflow(canvas, s, t, behind: true);
    if (s.wind.length > 0.5) _windStreaks(canvas, s, t, behind: true);
    if (s.groundZ != null) _ground(canvas, s, t);

    final items = <(double, void Function())>[];
    final light = (const V3(0.45, -0.35, 0.82).unit * 0.7 + cam.eyeDir * 0.5).unit;
    for (final p in s.polys) {
      final pts = <Offset>[];
      double dsum = 0;
      bool behind = false;
      for (final v in p.v) {
        final (o, d) = cam.project(v);
        if (d <= 0.05) {
          behind = true;
          break;
        }
        pts.add(o);
        dsum += d;
      }
      if (behind) continue;
      final (nx, ny, nd) = cam.vec(p.n);
      final (px, py, pd) = cam.vec(p.v[0] - cam.target);
      final facing = nx * px + ny * py + nd * (pd + cam.dist) < 0;
      if (!facing && !p.twoSided) continue;
      final n = facing ? p.n : p.n * -1;
      var c = Color.lerp(Colors.black, p.color, 0.30 + 0.70 * math.max(0.0, n.dot(light)))!;
      if (!facing) c = Color.lerp(Colors.black, c, 0.6)!;
      if (p.glow > 0) c = Color.lerp(c, const Color(0xFFFFB070), _clamp01(p.glow * 1.1))!;
      final path = Path()..addPolygon(pts, true);
      items.add((dsum / pts.length, () => canvas.drawPath(path, Paint()..color = c)));
    }
    for (final g in s.segs) {
      final (a, da) = cam.project(g.a);
      final (b, db) = cam.project(g.b);
      if (da <= 0.05 || db <= 0.05) continue;
      items.add(((da + db) / 2, () {
        if (g.glow) {
          canvas.drawLine(
              a,
              b,
              Paint()
                ..color = g.color.withValues(alpha: g.color.a * 0.6)
                ..strokeWidth = g.width * 2.5
                ..strokeCap = StrokeCap.round
                ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3));
        }
        canvas.drawLine(
            a,
            b,
            Paint()
              ..color = g.color
              ..strokeWidth = g.glow ? g.width * 0.6 : g.width
              ..strokeCap = StrokeCap.round);
      }));
    }
    items.sort((x, y) => y.$1.compareTo(x.$1));
    for (final (_, draw) in items) {
      draw();
    }
    if (s.flow > 0.03) _airflow(canvas, s, t, behind: false);
    if (s.wind.length > 0.5) _windStreaks(canvas, s, t, behind: false);
    for (final (p, text) in s.labels) {
      final (o, d) = cam.project(p);
      if (d <= 0.05) continue;
      final tp = TextPainter(
          text: TextSpan(text: text, style: consoleLabel(color: kInkDim, size: 9, weight: FontWeight.w600)),
          textDirection: TextDirection.ltr)
        ..layout();
      final at = Offset((o.dx + 6).clamp(4.0, math.max(4.0, size.width - tp.width - 4)),
          (o.dy - 6).clamp(4.0, math.max(4.0, size.height - 16)));
      tp.paint(canvas, at);
    }
  }

  /// Space above ~125 km, butterscotch Mars sky below ~20 km.
  void _sky(Canvas canvas, Size size, double altKm) {
    final lg = math.log(math.max(altKm, 0.001)) / math.ln10;
    final space = _clamp01((lg - 1.3) / 0.8); // 1 above ~125 km, 0 below 20 km
    final top = Color.lerp(const Color(0xFF2A2018), const Color(0xFF070A0F), space)!;
    final bottom = Color.lerp(const Color(0xFF6E5139), const Color(0xFF1A0F0A), space)!;
    canvas.drawRect(
        Offset.zero & size,
        Paint()
          ..shader = ui.Gradient.linear(Offset.zero, Offset(0, size.height), [top, bottom]));
    if (space > 0.05) {
      final p = Paint();
      for (final (x, y, a) in _stars) {
        p.color = Colors.white.withValues(alpha: a * 0.6 * space);
        canvas.drawCircle(Offset(x * size.width, y * size.height * 0.75), 0.5 + a * 0.6, p);
      }
    }
  }

  void _plasma(Canvas canvas, EdlScene s) {
    final (o, d) = cam.project(s.shieldApex!);
    if (d <= 0.05) return;
    final r = cam.focal * 3.4 / d * (0.7 + 0.6 * s.heat);
    canvas.drawCircle(
        o,
        r,
        Paint()
          ..shader = ui.Gradient.radial(o, r, [
            const Color(0xFFFFC27A).withValues(alpha: 0.75 * s.heat),
            const Color(0xFFFF6A2A).withValues(alpha: 0.35 * s.heat),
            const Color(0x00FF6A2A),
          ], [0, 0.45, 1]));
  }

  /// Streaks of air rushing past (upwards relative to the falling vehicle).
  void _airflow(Canvas canvas, EdlScene s, double t, {required bool behind}) {
    final c = s.center;
    final rnd = math.Random(5);
    final span = s.radius * 3.2;
    final speed = 4 + 40 * s.flow;
    final len = s.radius * (0.15 + 0.9 * s.flow);
    final color = Color.lerp(const Color(0xFFDDE6F0), const Color(0xFFFFB070), s.heat)!;
    for (int k = 0; k < 28; k++) {
      final th = rnd.nextDouble() * 2 * math.pi, rr = s.radius * (0.9 + 1.4 * rnd.nextDouble());
      final phase = rnd.nextDouble() * span;
      final z = c.z - span / 2 + ((t * speed + phase) % span);
      final a = V3(c.x + rr * math.cos(th), c.y + rr * math.sin(th), z);
      final (_, _, dep) = cam.vec(a - cam.target);
      if ((dep > 0) != behind) continue;
      final (pa, da) = cam.project(a);
      final (pb, db) = cam.project(a + V3(0, 0, len));
      if (da <= 0.05 || db <= 0.05) continue;
      canvas.drawLine(
          pa,
          pb,
          Paint()
            ..color = color.withValues(alpha: 0.12 + 0.4 * s.flow)
            ..strokeWidth = 1);
    }
  }

  /// Gusts: streaks blowing sideways through the view, faster and denser for a stronger wind.
  void _windStreaks(Canvas canvas, EdlScene s, double t, {required bool behind}) {
    final w = s.wind, strength = math.min(1.0, w.length / 35);
    final dir = w.unit;
    final side = V3(-dir.y, dir.x, 0).unit;
    final c = s.center;
    final span = s.radius * 5;
    final rnd = math.Random(9);
    final len = s.radius * (0.3 + 0.7 * strength);
    final n = (10 + 26 * strength).round();
    for (int k = 0; k < n; k++) {
      final off = side * ((rnd.nextDouble() - 0.5) * span) + V3(0, 0, (rnd.nextDouble() - 0.5) * span * 0.6);
      final along = ((t * (6 + 1.2 * w.length) + rnd.nextDouble() * span) % span) - span / 2;
      final a = c + off + dir * along;
      final (_, _, dep) = cam.vec(a - cam.target);
      if ((dep > 0) != behind) continue;
      final (pa, da) = cam.project(a);
      final (pb, db) = cam.project(a + dir * len);
      if (da <= 0.05 || db <= 0.05) continue;
      canvas.drawLine(
          pa,
          pb,
          Paint()
            ..color = const Color(0xFFCFE0FF).withValues(alpha: 0.10 + 0.35 * strength)
            ..strokeWidth = 1.2);
    }
  }

  void _ground(Canvas canvas, EdlScene s, double t) {
    final z = s.groundZ!;
    Path ring(double r) {
      final path = Path();
      for (int k = 0; k <= 64; k++) {
        final (o, d) = cam.project(V3(r * math.cos(2 * math.pi * k / 64), r * math.sin(2 * math.pi * k / 64), z));
        if (d <= 0.05) continue;
        k == 0 ? path.moveTo(o.dx, o.dy) : path.lineTo(o.dx, o.dy);
      }
      return path..close();
    }

    final (center, cd) = cam.project(V3(0, 0, z));
    if (cd <= 0.05) return;
    canvas.drawPath(
        ring(60),
        Paint()
          ..shader = ui.Gradient.radial(center, cam.focal * 40 / cd,
              [const Color(0xFF6A4330), const Color(0xFF4A2E20), const Color(0x004A2E20)], [0, 0.55, 1]));
    for (final r in [5.0, 10.0, 20.0, 40.0]) {
      canvas.drawPath(
          ring(r),
          Paint()
            ..color = Colors.white.withValues(alpha: 0.07)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1);
    }
    // shadow under the vehicle
    canvas.drawPath(ring(1.6), Paint()..color = Colors.black.withValues(alpha: 0.35));
    // plume–surface interaction: dust thrown up by the engines
    if (s.dust > 0) {
      final rnd = math.Random(3);
      for (int k = 0; k < 14; k++) {
        final th = rnd.nextDouble() * 2 * math.pi;
        final r0 = 1.5 + 6 * ((t * 0.6 + rnd.nextDouble()) % 1.0);
        final (o, d) = cam.project(V3(r0 * math.cos(th), r0 * math.sin(th), z + 0.4));
        if (d <= 0.05) continue;
        final rad = cam.focal * (0.8 + 0.5 * r0 / 6) / d;
        canvas.drawCircle(
            o,
            rad,
            Paint()
              ..shader = ui.Gradient.radial(o, rad, [
                const Color(0xFFB08A63).withValues(alpha: 0.45 * s.dust),
                const Color(0x00B08A63),
              ]));
      }
    }
  }

  @override
  bool shouldRepaint(covariant EdlPainter old) => true;
}


/// A wind gust's effect along the replay (from the API's `wind` block), sampled at any time.
class EdlWind {
  EdlWind(this.t, Map<String, dynamic> w)
      : e = _list(w['e']),
        n = _list(w['n']),
        u = _list(w['u']),
        driftE = _list(w['drift_e_m']),
        driftN = _list(w['drift_n_m']),
        swingE = _list(w['swing_e']),
        swingN = _list(w['swing_n']);

  final List<double> t, e, n, u, driftE, driftN, swingE, swingN;

  static List<double> _list(dynamic xs) => [for (final v in xs as List) v == null ? 0.0 : (v as num).toDouble()];

  double _at(List<double> ys, double x) {
    if (x <= t.first) return ys.first;
    if (x >= t.last) return ys.last;
    int lo = 0, hi = t.length - 1;
    while (hi - lo > 1) {
      final mid = (lo + hi) >> 1;
      if (t[mid] <= x) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    final f = (x - t[lo]) / (t[hi] - t[lo]);
    return ys[lo] + (ys[hi] - ys[lo]) * f;
  }

  /// Wind (m/s, x east, y north, z up).
  V3 windAt(double x) => x > t.last ? V3.zero : V3(_at(e, x), _at(n, x), _at(u, x));

  /// Sideways distance from NASA's path, metres (east, north). After the data ends it stays put.
  (double, double) driftAt(double x) => (_at(driftE, x), _at(driftN, x));

  (double, double) swingAt(double x) => x > t.last ? (0, 0) : (_at(swingE, x), _at(swingN, x));
}
