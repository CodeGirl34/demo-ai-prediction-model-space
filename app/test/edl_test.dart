import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';

import 'package:spacecraft_ai_app/edl3d.dart';
import 'package:spacecraft_ai_app/journey3d.dart' show V3;

/// A synthetic landing with the shape of the real one (times in s to touchdown, altitude in km).
(List<Map<String, dynamic>>, Map<String, dynamic>) _landing() {
  const ev = {
    'entry_interface': (-420.0, 130.0, 5300.0),
    'parachute_deploy': (-178.0, 12.6, 430.0),
    'heatshield_sep': (-155.0, 10.5, 160.0),
    'backshell_sep': (-61.0, 2.16, 81.0),
    'rover_sep': (-16.4, 0.02, 0.4),
    'touchdown': (0.0, 0.009, 0.4),
  };
  double lerp(double t, List<(double, double)> pts) {
    for (int i = 0; i < pts.length - 1; i++) {
      final (t0, y0) = pts[i];
      final (t1, y1) = pts[i + 1];
      if (t <= t1) return y0 + (y1 - y0) * (t - t0) / (t1 - t0);
    }
    return pts.last.$2;
  }

  final t = [for (int i = -960; i <= 0; i++) i.toDouble()];
  final alt = [
    for (final x in t) lerp(x, [(-960, 1400), (-420, 130), (-330, 30), (-178, 12.6), (-155, 10.5), (-61, 2.16), (-16.4, 0.02), (0, 0.009)])
  ];
  final v = [
    for (final x in t) lerp(x, [(-960, 4700), (-420, 5300), (-340, 4700), (-178, 430), (-155, 160), (-61, 81), (-16.4, 0.4), (0, 0.4)])
  ];
  final acc = [for (final x in t) 0.05 + 105 * math.exp(-math.pow((x + 330) / 40, 2))];
  return (
    [
      for (final e in ev.entries)
        {'event': e.key, 't_s': e.value.$1, 'altitude_km': e.value.$2, 'speed_m_s': e.value.$3},
    ],
    {'t_s': t, 'altitude_km': alt, 'speed_m_s': v, 'sensed_accel_m_s2': acc},
  );
}

void main() {
  test('NASA EDL sequence placed on the landing', () {
    final (events, traj) = _landing();
    final tl = EdlTimeline.build('m2020', events, traj);
    final ids = [for (final e in tl.events) e.id];
    expect(ids.first, 'cruise_sep');
    expect(ids.last, 'flyaway');
    expect(ids, containsAllInOrder(['entry', 'guided', 'peak_decel', 'parachute', 'heatshield', 'radar', 'trn',
        'backshell', 'powered', 'rover_dep', 'touchdown']));
    for (int i = 1; i < tl.events.length; i++) {
      expect(tl.events[i].t, greaterThanOrEqualTo(tl.events[i - 1].t));
    }
    expect(tl.at('peak_decel'), -330);
    expect(tl.events.firstWhere((e) => e.id == 'peak_heat').source, EdlSource.estimate);
    expect(tl.events.firstWhere((e) => e.id == 'parachute').source, EdlSource.data);

    // Curiosity had no terrain-relative navigation
    expect(EdlTimeline.build('msl', events, traj).has('trn'), isFalse);

    expect(tl.phaseAt(-500), EdlPhase.approach);
    expect(tl.phaseAt(-300), EdlPhase.entry);
    expect(tl.phaseAt(-100), EdlPhase.descent);
    expect(tl.phaseAt(-5), EdlPhase.landing);
  });

  test('3D configuration changes through the landing', () {
    final (events, traj) = _landing();
    final tl = EdlTimeline.build('m2020', events, traj);
    for (final now in <double>[-950, -400, -330, -170, -100, -40, -10, 0.5, 6, 13]) {
      final s = buildEdlScene(tl, now, 1.0);
      expect(s.polys, isNotEmpty, reason: 't=$now');
      expect(s.radius.isFinite, isTrue);
    }
    expect(buildEdlScene(tl, -330, 0).heat, greaterThan(0.5)); // hot during entry
    expect(buildEdlScene(tl, -100, 0).heat, 0); // heat shield gone
    expect(buildEdlScene(tl, -100, 0).groundZ, isNull); // 2+ km up: ground not drawn
    expect(buildEdlScene(tl, -5, 0).groundZ, isNotNull); // sky crane: ground in view
    expect(buildEdlScene(tl, -5, 0).dust, greaterThan(0.5)); // engines kick up dust near the ground
    expect(buildEdlScene(tl, 6, 0).dust, 0); // ...and it settles once the descent stage has flown off
    final labels = [for (final (_, l) in buildEdlScene(tl, -170, 0).labels) l];
    expect(labels, contains('PARACHUTE'));
  });

  test('a wind gust leans the stack and is sampled along the replay', () {
    final (events, traj) = _landing();
    final tl = EdlTimeline.build('msl', events, traj);
    final calm = buildEdlScene(tl, -120, 0);
    final windy = buildEdlScene(tl, -120, 0, wind: const V3(20, 0, 0), swing: (0.2, 0.0));
    V3 chute(EdlScene s) => s.labels.firstWhere((l) => l.$2 == 'PARACHUTE').$1;
    expect(chute(windy).x - chute(calm).x, greaterThan(1)); // canopy leans downwind (east)
    expect(windy.groundZ, calm.groundZ); // the ground does not swing
    expect(windy.wind.x, 20);

    final w = EdlWind([-10, 0, 10], {
      'e': [0, 10, 0], 'n': [0, 0, 0], 'u': [0, 0, 0],
      'drift_e_m': [0, 50, 100], 'drift_n_m': [0, 0, 0],
      'swing_e': [0, 0.1, 0], 'swing_n': [0, 0, 0],
    });
    expect(w.windAt(-5).x, closeTo(5, 1e-9));
    expect(w.driftAt(5).$1, closeTo(75, 1e-9));
    expect(w.driftAt(30).$1, 100); // stays where the wind left it
    expect(w.windAt(30).length, 0);
  });
}

