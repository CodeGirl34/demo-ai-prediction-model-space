import 'dart:async';
import 'dart:math' as math;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../api.dart';
import '../edl3d.dart' show EdlWind;
import '../theme.dart';
import '../widgets.dart';
import 'edl_card.dart';

const _faultKinds = ['spike', 'bias', 'drift', 'stuck', 'noise', 'dropout'];
const _phases = ['approach', 'entry', 'parachute', 'powered_descent', 'sky_crane'];
const _gustPhases = ['entry', 'parachute', 'powered_descent', 'sky_crane']; // no air in approach

/// What a gust does in each phase, in plain words (from the model in faults/wind.py).
const _gustHelp = <String, String>{
  'entry': 'Very fast through thin air: drag is strong, so the capsule is shoved sideways by a few hundred metres.',
  'parachute': 'The parachute is made to catch air, so it drifts with the wind: usually the biggest push, '
      'hundreds of metres.',
  'powered_descent': 'Barely moves it: the descent stage is heavy, the air is thin, and the engines correct '
      'the push. Expect a few metres.',
  'sky_crane': 'Barely moves it: near the ground the engines hold the descent stage in place. Expect about a metre.',
};

class ReplayScreen extends StatefulWidget {
  const ReplayScreen({super.key, required this.api});
  final Api api;

  @override
  State<ReplayScreen> createState() => _ReplayScreenState();
}

class _ReplayScreenState extends State<ReplayScreen> {
  String mission = 'msl';
  String channel = 'vertical_speed_m_s';
  final List<Map<String, String>> faults = [
    {'kind': 'drift', 'phase': 'parachute'},
    {'kind': 'spike', 'phase': 'entry'},
  ];
  int randomFaults = 0;
  int seed = 0;
  final List<Map<String, String>> gusts = [];

  Map<String, dynamic>? data;
  bool loading = false;
  Object? error;

  int cursor = 0;
  double _cursorF = 0;
  bool playing = false;
  int speed = 30;
  Timer? _timer;

  // decoded series
  List<double> t = [], alt = [], value = [], healthy = [], nasa = [], score = [], driftH = [];
  EdlWind? windData;
  List<String> phasePred = [], phaseTrue = [];

  @override
  void initState() {
    super.initState();
    _run();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  List<double> _nums(dynamic xs, {double fill = double.nan}) =>
      (xs as List).map((v) => v == null ? fill : (v as num).toDouble()).toList();

  Future<void> _run() async {
    _pause();
    setState(() {
      loading = true;
      error = null;
    });
    try {
      final r = await widget.api.replay(mission,
          channel: channel, faults: faults, randomFaults: randomFaults, seed: seed, gusts: gusts);
      setState(() {
        data = r;
        t = _nums(r['t_s']);
        alt = _nums(r['altitude_km']);
        value = _nums(r['value']);
        healthy = r['value_healthy'] == null ? [] : _nums(r['value_healthy']);
        nasa = r['value_nasa'] == null ? [] : _nums(r['value_nasa']);
        windData = r['wind'] == null ? null : EdlWind(t, r['wind'] as Map<String, dynamic>);
        driftH = windData == null
            ? []
            : [
                for (int k = 0; k < t.length; k++)
                  math.sqrt(math.pow(windData!.driftE[k], 2) + math.pow(windData!.driftN[k], 2)).toDouble()
              ];
        score = _nums(r['anomaly_score']);
        phasePred = (r['phase_pred'] as List).cast<String>();
        phaseTrue = (r['phase_true'] as List).cast<String>();
        cursor = 0;
        _cursorF = 0;
        loading = false;
      });
      _play();
    } catch (e) {
      setState(() {
        error = e;
        loading = false;
      });
    }
  }

  void _play() {
    if (data == null) return;
    if (cursor >= t.length - 1) {
      cursor = 0;
      _cursorF = 0;
    }
    _timer?.cancel();
    const tick = Duration(milliseconds: 100);
    final stepS = (data!['step_s'] as num).toDouble();
    _timer = Timer.periodic(tick, (_) {
      _cursorF += speed * 0.1 / stepS;
      final next = math.min(_cursorF.floor(), t.length - 1);
      if (next != cursor) setState(() => cursor = next);
      if (cursor >= t.length - 1) _pause();
    });
    setState(() => playing = true);
  }

  /// Jump to the first sample at or after [ts] (seconds to touchdown), e.g. from the EDL step list,
  /// so the step that was tapped is the one on screen.
  void _seekTo(double ts) {
    if (t.isEmpty) return;
    _pause();
    int best = t.length - 1;
    for (int k = 0; k < t.length; k++) {
      if (t[k] >= ts - 1e-6) {
        best = k;
        break;
      }
    }
    setState(() {
      cursor = best;
      _cursorF = best.toDouble();
    });
  }

  void _pause() {
    _timer?.cancel();
    _timer = null;
    if (mounted) setState(() => playing = false);
  }

  Future<void> _addFault() async {
    String kind = 'bias', phase = 'entry';
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setD) => AlertDialog(
          title: const Text('INJECT A SENSOR FAULT'),
          content: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text(
                'A fault breaks the sensor, not the spacecraft. The landing itself (altitude, the 3D view) '
                'stays the same. Only the sensor reading goes wrong, and the AI has to notice.',
                style: TextStyle(fontSize: 12.5, color: kInkDim, height: 1.4)),
            const SizedBox(height: 14),
            DropdownButtonFormField<String>(
              initialValue: kind,
              dropdownColor: kPanel2,
              decoration: const InputDecoration(labelText: 'Fault type'),
              items: [for (final k in _faultKinds) DropdownMenuItem(value: k, child: Text(faultLabels[k]!))],
              onChanged: (v) => setD(() => kind = v!),
            ),
            const SizedBox(height: 6),
            Text(faultHelp[kind]!, style: const TextStyle(fontSize: 12, color: kInkFaint, height: 1.4)),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: phase,
              dropdownColor: kPanel2,
              decoration: const InputDecoration(labelText: 'During phase'),
              items: [for (final p in _phases) DropdownMenuItem(value: p, child: Text(phaseLabel(p)))],
              onChanged: (v) => setD(() => phase = v!),
            ),
          ]),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Add')),
          ],
        ),
      ),
    );
    if (ok == true) setState(() => faults.add({'kind': kind, 'phase': phase}));
  }

  Future<void> _addGust() async {
    String strength = 'strong', phase = 'parachute';
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setD) => AlertDialog(
          title: const Text('ADD A WIND GUST'),
          content: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text(
                'Unlike a fault, a gust is real: it pushes the spacecraft off NASA\'s path. Every sensor still works '
                'and reads the new motion. There are no gusts in approach: there is no air above ~125 km.',
                style: TextStyle(fontSize: 12.5, color: kInkDim, height: 1.4)),
            const SizedBox(height: 14),
            DropdownButtonFormField<String>(
              initialValue: strength,
              dropdownColor: kPanel2,
              decoration: const InputDecoration(labelText: 'Strength'),
              items: [
                for (final k in gustLabels.keys)
                  DropdownMenuItem(value: k, child: Text('${gustLabels[k]} · ${gustSpeeds[k]} m/s')),
              ],
              onChanged: (v) => setD(() => strength = v!),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: phase,
              dropdownColor: kPanel2,
              decoration: const InputDecoration(labelText: 'During phase'),
              items: [for (final p in _gustPhases) DropdownMenuItem(value: p, child: Text(phaseLabel(p)))],
              onChanged: (v) => setD(() => phase = v!),
            ),
            const SizedBox(height: 6),
            Text(_gustHelp[phase]!, style: const TextStyle(fontSize: 12, color: kInkFaint, height: 1.4)),
          ]),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Add')),
          ],
        ),
      ),
    );
    if (ok == true) setState(() => gusts.add({'phase': phase, 'strength': strength}));
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      final wide = c.maxWidth > 1000;
      final controls = _controls();
      final main = data == null
          ? (loading ? const LoadingOrError() : LoadingOrError(error: error, onRetry: _run))
          : _dashboard(wide);
      if (wide) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SizedBox(width: 320, child: ListView(padding: const EdgeInsets.all(16), children: [controls])),
          Expanded(child: main),
        ]);
      }
      return ListView(padding: const EdgeInsets.all(16), children: [
        controls,
        const SizedBox(height: 12),
        if (data == null) SizedBox(height: 300, child: main) else main,
      ]);
    });
  }

  Widget _controls() {
    return SectionCard(
      title: 'Replay setup',
      icon: Icons.tune,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Text('1 · CHOOSE A LANDING', style: consoleLabel(size: 9.5)),
        const SizedBox(height: 8),
        SegmentedButton<String>(
          segments: const [
            ButtonSegment(value: 'msl', label: Text('Curiosity')),
            ButtonSegment(value: 'm2020', label: Text('Perseverance')),
          ],
          selected: {mission},
          showSelectedIcon: false,
          onSelectionChanged: (s) => setState(() => mission = s.first),
        ),
        const SizedBox(height: 8),
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: LedDot(color: mission == 'msl' ? kGo : kSignal, size: 6),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              mission == 'msl'
                  ? 'Fair test: the AI never saw this landing. It was trained on Perseverance only.'
                  : 'The AI learned from this landing, so results here look better than they really are.',
              style: const TextStyle(color: kInkDim, fontSize: 12, height: 1.4),
            ),
          ),
        ]),
        const SizedBox(height: 18),
        Text('2 · PICK THE SENSOR TO WATCH', style: consoleLabel(size: 9.5)),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          initialValue: channel,
          dropdownColor: kPanel2,
          style: readout(size: 13),
          decoration: const InputDecoration(isDense: true),
          items: [
            for (final e in channelLabels.entries) DropdownMenuItem(value: e.key, child: Text(e.value)),
          ],
          onChanged: (v) => setState(() => channel = v!),
        ),
        const SizedBox(height: 18),
        Row(children: [
          Expanded(child: Text('3 · BREAK IT ON PURPOSE', style: consoleLabel(size: 9.5))),
          const EvidenceBadge(Evidence.simulated),
        ]),
        const SizedBox(height: 8),
        Wrap(spacing: 6, runSpacing: 6, children: [
          for (int i = 0; i < faults.length; i++)
            InputChip(
              label: Text('${faultLabels[faults[i]['kind']]} · ${phaseLabel(faults[i]['phase']!)}'),
              onDeleted: () => setState(() => faults.removeAt(i)),
            ),
          if (randomFaults > 0)
            InputChip(
              avatar: const Icon(Icons.casino_outlined, size: 16),
              label: Text('$randomFaults random (seed $seed)'),
              onDeleted: () => setState(() => randomFaults = 0),
            ),
          if (faults.isEmpty && randomFaults == 0)
            Text('No faults · clean landing', style: readout(size: 12, color: kInkFaint)),
        ]),
        const SizedBox(height: 8),
        Row(children: [
          Expanded(
            child: OutlinedButton.icon(
              onPressed: faults.length < 5 ? _addFault : null,
              icon: const Icon(Icons.add),
              label: const Text('Add fault'),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: () => setState(() {
                randomFaults = 2;
                seed = DateTime.now().millisecondsSinceEpoch % 100000;
              }),
              icon: const Icon(Icons.casino_outlined),
              label: const Text('Random'),
            ),
          ),
        ]),
        const SizedBox(height: 18),
        Row(children: [
          Expanded(child: Text('4 · ADD WEATHER', style: consoleLabel(size: 9.5))),
          const EvidenceBadge(Evidence.simulated),
        ]),
        const SizedBox(height: 6),
        const Text('A wind gust really moves the spacecraft. The sensors all keep working.',
            style: TextStyle(color: kInkFaint, fontSize: 12, height: 1.4)),
        const SizedBox(height: 8),
        Wrap(spacing: 6, runSpacing: 6, children: [
          for (int i = 0; i < gusts.length; i++)
            InputChip(
              avatar: const Icon(Icons.air, size: 16, color: kWind),
              label: Text('${gustLabels[gusts[i]['strength']]} · ${phaseLabel(gusts[i]['phase']!)}'),
              onDeleted: () => setState(() => gusts.removeAt(i)),
            ),
          if (gusts.isEmpty) Text('No wind · calm landing', style: readout(size: 12, color: kInkFaint)),
        ]),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: gusts.length < 3 ? _addGust : null,
          icon: const Icon(Icons.air),
          label: const Text('Add wind gust'),
        ),
        const SizedBox(height: 18),
        FilledButton.icon(
          onPressed: loading ? null : _run,
          icon: loading
              ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: kInkFaint))
              : const Icon(Icons.play_arrow, size: 18),
          label: Text(loading ? 'Running models…' : 'Run replay'),
        ),
        if (error != null && data != null) ...[
          const SizedBox(height: 8),
          Text('$error', style: readout(size: 11, color: kCaution)),
        ],
      ]),
    );
  }

  Widget _dashboard(bool wide) {
    final i = cursor;
    final now = t[i];
    final alarms = (data!['alarms'] as List).cast<Map<String, dynamic>>();
    final injected = (data!['faults'] as List).cast<Map<String, dynamic>>();
    final seen = alarms.where((a) => (a['start_t'] as num) <= now).toList().reversed.toList();
    final active = alarms.any((a) => (a['start_t'] as num) <= now && (a['end_t'] as num) >= now);
    final faultBands = [
      for (final f in injected) (x1: (f['start_t'] as num).toDouble(), x2: (f['end_t'] as num).toDouble())
    ];
    final alarmBands = [
      for (final a in alarms) (x1: (a['start_t'] as num).toDouble(), x2: (a['end_t'] as num).toDouble())
    ];
    final blown = ((data!['gusts'] as List?) ?? const []).cast<Map<String, dynamic>>();
    final gustBands = [
      for (final g in blown) (x1: (g['start_t'] as num).toDouble(), x2: (g['end_t'] as num).toDouble())
    ];
    final windy = blown.isNotEmpty;

    final charts = [
      _chartCard('Altitude above landing site', _altChart(now),
          evidence: const [Evidence.nasa], note: 'log scale · background colour = AI’s phase estimate'),
      _chartCard('Sensor: ${channelLabels[channel]!}',
          _lineChart(value, now, kObserved, faultBands, alarmBands,
              ref: healthy, ref2: windy ? nasa : const [], gustBands: gustBands),
          evidence: const [Evidence.nasa, Evidence.simulated],
          legend: [
            ('what the sensor reports', kObserved),
            ('healthy sensor', kInkDim),
            if (windy) ('NASA\'s landing, no wind', kInkFaint),
            ('injected fault', kScenario),
            if (windy) ('wind gust', kWind),
            ('AI alarm', kCaution),
          ],
          note: windy
              ? 'A fault changes only this reading. A wind gust really moves the spacecraft, so the healthy reading '
                  'moves away from NASA\'s (dotted). Tap a fault or a gust in the lists to jump to it.'
              : 'A fault changes only this reading, not the landing: altitude and the 3D view stay the same. '
                  'Faults last seconds, so look at the close-up, or tap a fault in “Faults we injected” to jump to it.',
          below: _closeUp(now, faultBands, alarmBands, gustBands)),
      if (windy)
        _chartCard('Pushed off course', _lineChart(driftH, now, kWind, const [], const [], gustBands: gustBands, floorY: 0),
            evidence: const [Evidence.simulated],
            legend: const [('sideways distance from NASA\'s path', kWind)],
            note: 'metres · shaded = the gust · the engines cancel the sideways speed but do not fly back'),
      _chartCard('AI surprise score',
          _lineChart([for (final s in score) s.isNaN ? double.nan : math.log(math.max(s, 1e-5)) / math.ln10],
              now, kDerived, const [], alarmBands),
          evidence: const [Evidence.ai],
          note: 'how far the reading is from what the AI expected · log scale · high = suspicious'),
    ];

    final feed = _feed(seen, injected, now);
    final edl = EdlCard(
      api: widget.api,
      mission: data!['mission'] as String,
      now: now,
      playing: playing,
      speed: speed.toDouble(),
      onSeek: _seekTo,
      compact: !wide,
      wind: windData,
      gusts: [
        for (final g in blown)
          (start: (g['start_t'] as num).toDouble(), end: (g['end_t'] as num).toDouble(), label: 'gust')
      ],
      faults: [
        for (final f in injected)
          (start: (f['start_t'] as num).toDouble(), end: (f['end_t'] as num).toDouble(), label: faultLabels[f['kind']]!)
      ],
      alarms: [for (final b in alarmBands) (start: b.x1, end: b.x2, label: 'alarm')],
    );
    final content = Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _status(i, active),
      const SizedBox(height: 12),
      if (wide)
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 3, child: Column(children: [for (final c in charts) ...[c, const SizedBox(height: 12)]])),
          const SizedBox(width: 12),
          Expanded(flex: 2, child: Column(children: [edl, const SizedBox(height: 12), feed])),
        ])
      else ...[
        edl,
        const SizedBox(height: 12),
        for (final c in charts) ...[c, const SizedBox(height: 12)],
        feed,
      ],
    ]);
    return wide ? ListView(padding: const EdgeInsets.fromLTRB(0, 16, 16, 16), children: [content]) : content;
  }

  Widget _status(int i, bool alarmActive) {
    final pred = phasePred[i], truth = phaseTrue[i];
    final s = data!['summary'] as Map<String, dynamic>;
    final alarmColor = alarmActive ? kCaution : kGo;
    return SectionCard(
      title: 'Live landing',
      icon: Icons.radar,
      evidence: const [Evidence.nasa, Evidence.ai],
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Wrap(spacing: 28, runSpacing: 14, crossAxisAlignment: WrapCrossAlignment.end, children: [
          Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
            Text('AI THINKS WE ARE IN', style: consoleLabel(size: 10)),
            const SizedBox(height: 6),
            PhaseChip(pred, large: true),
          ]),
          Metric(label: 'NASA says', value: phaseLabel(truth),
              color: pred == truth ? kGo : kSignal, caption: pred == truth ? '✓ AI agrees' : '✗ AI disagrees'),
          Metric(label: 'Time to touchdown', value: fmtT(t[i])),
          Metric(label: 'Altitude', value: alt[i].isNaN ? '–' : fmtAlt(alt[i])),
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: ShapeDecoration(
              color: alarmActive ? kCautionSoft : kGoSoft,
              shape: chamfer(cut: 6, side: alarmColor.withValues(alpha: 0.7)),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              LedDot(color: alarmColor, pulse: alarmActive, size: 8),
              const SizedBox(width: 10),
              Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
                Text(alarmActive ? 'ANOMALY' : 'NOMINAL',
                    style: consoleLabel(color: alarmColor, size: 14, weight: FontWeight.w700)),
                Text(alarmActive ? 'AI alarm is on' : 'sensor looks normal', style: readout(size: 10, color: kInkDim)),
              ]),
            ]),
          ),
        ]),
        const SizedBox(height: 16),
        Row(children: [
          IconButton(
            onPressed: playing ? _pause : _play,
            icon: Icon(playing ? Icons.pause : Icons.play_arrow, size: 20),
            tooltip: playing ? 'Pause' : 'Play',
            style: IconButton.styleFrom(
              foregroundColor: kSignal,
              backgroundColor: kSignalSoft,
              shape: chamfer(cut: 6, side: kSignal.withValues(alpha: 0.5)),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Slider(
              value: i.toDouble(),
              max: (t.length - 1).toDouble(),
              onChanged: (v) {
                _pause();
                setState(() {
                  cursor = v.round();
                  _cursorF = v;
                });
              },
            ),
          ),
          // compact so the slider keeps its width on phones; tap cycles 10× → 30× → 90×
          Tooltip(
            message: 'Playback speed',
            child: OutlinedButton.icon(
              onPressed: () => setState(() => speed = const {10: 30, 30: 90, 90: 10}[speed]!),
              icon: const Icon(Icons.speed, size: 18),
              label: Text('$speed×'),
            ),
          ),
        ]),
        const SizedBox(height: 8),
        Row(children: [
          SizedBox(width: 104, child: Text('AI ESTIMATE', style: consoleLabel(size: 9.5, color: kDerived))),
          Expanded(child: PhaseStrip(phases: phasePred, upTo: i)),
        ]),
        const SizedBox(height: 5),
        Row(children: [
          SizedBox(width: 104, child: Text('NASA TIMELINE', style: consoleLabel(size: 9.5, color: kObserved))),
          Expanded(child: PhaseStrip(phases: phaseTrue)),
        ]),
        const SizedBox(height: 10),
        Legend([for (final p in _phases) (phaseLabel(p), phaseColors[p]!)]),
        const SizedBox(height: 14),
        const SeamRule(),
        const SizedBox(height: 12),
        Wrap(spacing: 28, runSpacing: 10, crossAxisAlignment: WrapCrossAlignment.end, children: [
          _Score('Phase accuracy', '${(((data!['phase_accuracy'] as num?) ?? 0) * 100).toStringAsFixed(1)} %'),
          _Score('Faults caught', '${s['faults_detected']} / ${s['faults_injected']}',
              color: s['faults_detected'] == s['faults_injected'] ? kGo : kSignal),
          if ((s['gusts'] ?? 0) as num > 0)
            _Score('Gusts noticed', '${s['gusts_noticed']} / ${s['gusts']}', color: kWind),
          _Score('False alarms', '${s['false_alarms']}', color: s['false_alarms'] == 0 ? kGo : kSignal),
          Text('scores for the whole landing', style: readout(size: 10.5, color: kInkFaint)),
        ]),
      ]),
    );
  }

  Widget _feed(List<Map<String, dynamic>> seen, List<Map<String, dynamic>> injected, double now) {
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      SectionCard(
        title: 'AI alarms',
        icon: Icons.notifications_active_outlined,
        evidence: const [Evidence.ai],
        trailing: Tag('${seen.length} so far', color: seen.isEmpty ? kGo : kCaution),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          if (seen.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 6),
              child: Row(children: [
                LedDot(color: kGo, size: 6),
                SizedBox(width: 10),
                Expanded(
                  child: Text('No alarms yet. Every reading so far matches what the AI expected.',
                      style: TextStyle(color: kInkDim, fontSize: 12.5, height: 1.4)),
                ),
              ]),
            ),
          for (final a in seen) _alertTile(a, injected),
        ]),
      ),
      const SizedBox(height: 12),
      SectionCard(
        title: 'Faults we injected',
        icon: Icons.bolt,
        evidence: const [Evidence.simulated],
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Text('The answer key: the faults we put in on purpose. Did the AI catch each one?',
              style: TextStyle(color: kInkFaint, fontSize: 12, height: 1.4)),
          const SizedBox(height: 8),
          if (injected.isEmpty) Text('None · this is a clean landing.', style: readout(size: 12, color: kInkFaint)),
          for (final f in injected) _faultTile(f, now),
        ]),
      ),
      if (((data!['gusts'] as List?) ?? const []).isNotEmpty) ...[
        const SizedBox(height: 12),
        SectionCard(
          title: 'Wind gusts we added',
          icon: Icons.air,
          evidence: const [Evidence.simulated],
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            const Text(
                'A gust is real: the spacecraft moves and every sensor still works. The AI was trained to catch '
                'broken sensors, so noticing a gust is a bonus, not its job. "Noticed" means an alarm that does not '
                'go off when the same landing is replayed without the wind.',
                style: TextStyle(color: kInkFaint, fontSize: 12, height: 1.4)),
            const SizedBox(height: 8),
            for (final g in (data!['gusts'] as List).cast<Map<String, dynamic>>()) _gustTile(g, now),
          ]),
        ),
      ],
    ]);
  }

  Widget _gustTile(Map<String, dynamic> g, double now) {
    final drift = (g['drift_at_touchdown_m'] as num).toDouble(), maxDrift = (g['max_drift_m'] as num).toDouble();
    final ch = (g['max_change'] as num).toDouble();
    final over = (g['effect_end_t'] as num) <= now;
    final noticed = g['noticed'] == true;
    String m(double x) => x.toStringAsFixed(x < 10 ? 1 : 0);
    final effect = maxDrift < 10
        ? 'Barely moved it (${m(maxDrift)} m): the descent stage is heavy, the air is thin, and the engines correct the push.'
        : 'Pushed it ${m(maxDrift)} m off NASA\'s path; the rover lands ${m(drift)} m from NASA\'s spot.';
    final reading = 'Changed this reading by up to ${ch.toStringAsFixed(ch >= 10 ? 0 : (ch >= 1 ? 1 : 2))} '
        '${channelUnit(channel)}${over && !noticed ? ': smooth enough to still look like normal flight to the AI.' : '.'}';
    return InkWell(
      // land on the gust's peak: strongest wind, the stack leaning, the close-up showing the push
      onTap: () => _seekTo(((g['start_t'] as num) + (g['end_t'] as num)) / 2),
      child: Container(
        margin: const EdgeInsets.only(top: 6),
        padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
        decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
        child: Row(children: [
          Icon(Icons.air, size: 16, color: (g['start_t'] as num) <= now ? kWind : kInkFaint),
          const SizedBox(width: 10),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(
                  '${gustLabels[g['strength']]} (${(g['peak_wind_m_s'] as num).toInt()} m/s → '
                  '${compass(g['direction_deg'] as num)}) during ${phaseLabel(g['phase'])}',
                  style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
              const SizedBox(height: 2),
              Text('${fmtT(g['start_t'] as num)} → ${fmtT(g['end_t'] as num)} · tap to see it',
                  style: readout(size: 11, color: kInkFaint)),
              const SizedBox(height: 4),
              Text('$effect $reading', style: const TextStyle(fontSize: 11.5, color: kInkDim, height: 1.35)),
            ]),
          ),
          const SizedBox(width: 8),
          over
              ? Tag(noticed ? 'Noticed' : 'Not noticed', color: noticed ? kGo : kInkDim)
              : const Tag('Upcoming', color: kInkFaint),
        ]),
      ),
    );
  }

  Widget _faultTile(Map<String, dynamic> f, double now) {
    final effect = _effect(f);
    return InkWell(
      // land on the fault's last sample: the close-up shows all of it and the 3D view flags it
      onTap: () => _seekTo((f['end_t'] as num).toDouble()),
      child: Container(
        margin: const EdgeInsets.only(top: 6),
        padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
        decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
        child: Row(children: [
          Icon(Icons.bolt, size: 16, color: (f['start_t'] as num) <= now ? kScenario : kInkFaint),
          const SizedBox(width: 10),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('${faultLabels[f['kind']]} during ${phaseLabel(f['phase'])}',
                  style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
              const SizedBox(height: 2),
              Text('${fmtT(f['start_t'] as num)} → ${fmtT(f['end_t'] as num)} · tap to see it',
                  style: readout(size: 11, color: kInkFaint)),
              if (effect != null) ...[
                const SizedBox(height: 4),
                Text(effect, style: const TextStyle(fontSize: 11.5, color: kInkDim, height: 1.35)),
              ],
            ]),
          ),
          const SizedBox(width: 8),
          (f['end_t'] as num) <= now
              ? Tag(f['detected'] == true ? 'Caught' : 'Missed', color: f['detected'] == true ? kGo : kCaution)
              : const Tag('Upcoming', color: kInkFaint),
        ]),
      ),
    );
  }

  /// How much the fault actually moved the reading. A stuck sensor or a dropout can change almost
  /// nothing when the real value is flat or already near zero.
  String? _effect(Map<String, dynamic> f) {
    final ch = (f['max_change'] as num?)?.toDouble(), frac = (f['max_change_frac'] as num?)?.toDouble();
    if (ch == null || frac == null) return null;
    final v = '${ch.toStringAsFixed(ch >= 100 ? 0 : (ch >= 1 ? 1 : 2))} ${channelUnit(channel)}';
    return frac < 0.005
        ? 'Barely changed the reading (at most $v): the real value was nearly flat or near zero here, '
            'so there is almost nothing to see or to catch.'
        : 'Changed the reading by up to $v.';
  }

  Widget _alertTile(Map<String, dynamic> a, List<Map<String, dynamic>> injected) {
    final m = a['matches_fault'];
    final real = m != null;
    final gust = !real && a['matches_gust'] != null;
    final dets = (a['detectors'] as List).map((d) => d == 'persistence' ? 'baseline' : 'LSTM').join(' + ');
    final c = real ? kCaution : (gust ? kWind : kSignal);
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.fromLTRB(12, 10, 10, 10),
      decoration: BoxDecoration(
        color: kPanel2,
        border: Border(left: BorderSide(color: c, width: 3)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Wrap(spacing: 10, runSpacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
              Text(fmtT(a['start_t'] as num), style: readout(size: 12.5, weight: FontWeight.w500)),
              PhaseChip(a['phase'] as String),
            ]),
            const SizedBox(height: 6),
            Text(
                real
                    ? 'Real problem: matches the ${faultLabels[injected[m as int]['kind']]!.toLowerCase()} we injected.'
                    : gust
                        ? 'Real event: the wind gust moved the spacecraft. The sensor works; the motion was unusual.'
                        : 'False alarm: nothing was broken here.',
                style: const TextStyle(fontSize: 12.5, color: kInkDim)),
            const SizedBox(height: 2),
            Text('flagged by $dets', style: readout(size: 10.5, color: kInkFaint)),
          ]),
        ),
        const SizedBox(width: 8),
        Tag(real ? 'Correct' : (gust ? 'Wind' : 'False alarm'), color: c),
      ]),
    );
  }


  static const _closeBack = 20.0, _closeAhead = 10.0;

  /// The sensor over a 30 s window around now, where a few-second fault is easy to see.
  Widget _closeUp(double now, List<({double x1, double x2})> faultBands, List<({double x1, double x2})> alarmBands,
      List<({double x1, double x2})> gustBands) {
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      const SizedBox(height: 12),
      Text('CLOSE-UP · ${_closeBack.toInt()} s BEFORE TO ${_closeAhead.toInt()} s AFTER NOW', style: consoleLabel(size: 9.5)),
      const SizedBox(height: 6),
      Container(
        height: 150,
        padding: const EdgeInsets.fromLTRB(0, 8, 8, 0),
        decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
        child: _lineChart(value, now, kObserved, faultBands, alarmBands,
            ref: healthy,
            ref2: gustBands.isEmpty ? const [] : nasa,
            gustBands: gustBands,
            x0: now - _closeBack,
            x1: now + _closeAhead),
      ),
    ]);
  }

  Widget _chartCard(String title, Widget chart,
      {List<(String, Color)> legend = const [], List<Evidence> evidence = const [], String? note, Widget? below}) {
    return SectionCard(
      padding: 14,
      title: title,
      evidence: evidence,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        if (legend.isNotEmpty || note != null) ...[
          Wrap(spacing: 14, runSpacing: 6, crossAxisAlignment: WrapCrossAlignment.center, children: [
            if (legend.isNotEmpty) Legend(legend),
            if (note != null) Text(note, style: readout(size: 10.5, color: kInkFaint)),
          ]),
          const SizedBox(height: 10),
        ],
        Container(
          height: 170,
          padding: const EdgeInsets.fromLTRB(0, 8, 8, 0),
          decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
          child: chart,
        ),
        ?below,
      ]),
    );
  }

  Widget _altChart(double now) {
    final y = [for (final a in alt) a.isNaN ? double.nan : math.log(math.max(a, 0.005)) / math.ln10];
    return _lineChart(y, now, kInk, const [], const [],
        yTitle: (v) => v == v.roundToDouble() ? fmtAlt(math.pow(10, v)) : '', phaseBands: true);
  }

  Widget _lineChart(
    List<double> ys,
    double now,
    Color color,
    List<({double x1, double x2})> faultBands,
    List<({double x1, double x2})> alarmBands, {
    String Function(double)? yTitle,
    bool phaseBands = false,
    List<double> ref = const [], // drawn dashed underneath, e.g. what a healthy sensor would read
    List<double> ref2 = const [], // dotted, e.g. NASA's reading without the wind
    List<({double x1, double x2})> gustBands = const [],
    double? x0, // optional time window (close-up)
    double? x1,
    double? floorY, // lowest value the axis may show, e.g. 0 for a distance
  }) {
    final minX = x0 ?? t.first, maxX = x1 ?? t.last;
    final window = x0 != null || x1 != null;
    final past = <FlSpot>[], future = <FlSpot>[], refSpots = <FlSpot>[], ref2Spots = <FlSpot>[];
    double lo = double.infinity, hi = -double.infinity;
    for (int k = 0; k < ys.length; k++) {
      final v = ys[k];
      if (v.isNaN) continue;
      if (window && (t[k] < minX - 1 || t[k] > maxX + 1)) continue;
      final inView = t[k] >= minX && t[k] <= maxX;
      if (inView) {
        lo = math.min(lo, v);
        hi = math.max(hi, v);
      }
      final s = FlSpot(t[k], v);
      if (t[k] <= now) past.add(s);
      if (t[k] >= now) future.add(s);
      if (k < ref.length && !ref[k].isNaN) {
        refSpots.add(FlSpot(t[k], ref[k]));
        if (inView) {
          lo = math.min(lo, ref[k]);
          hi = math.max(hi, ref[k]);
        }
      }
      if (k < ref2.length && !ref2[k].isNaN) {
        ref2Spots.add(FlSpot(t[k], ref2[k]));
        if (inView) {
          lo = math.min(lo, ref2[k]);
          hi = math.max(hi, ref2[k]);
        }
      }
    }
    if (lo == double.infinity) {
      lo = 0;
      hi = 1;
    }
    final pad = (hi - lo) * 0.08 + 1e-9;
    final minHalf = window ? 0.15 : math.max(0.5, (t.last - t.first) * 0.002);
    final xStep = window ? 5.0 : 200.0;
    final bands = <VerticalRangeAnnotation>[
      if (phaseBands)
        for (final b in _phaseRuns())
          VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: phaseColors[b.phase]!.withValues(alpha: b.x1 <= now ? 0.16 : 0.05)),
      for (final b in gustBands)
        VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: kWind.withValues(alpha: 0.16)),
      for (final b in alarmBands)
        if (b.x1 <= now)
          VerticalRangeAnnotation(x1: b.x1 - 0.5, x2: math.min(b.x2, now) + 0.5, color: kCaution.withValues(alpha: 0.30)),
      // drawn last (and never thinner than a few pixels) so a one-sample spike stays visible over its alarm
      for (final b in faultBands)
        VerticalRangeAnnotation(x1: b.x1 - minHalf, x2: b.x2 + minHalf, color: kScenario.withValues(alpha: 0.5)),
    ];
    return LineChart(
      duration: Duration.zero,
      LineChartData(
        minX: minX,
        maxX: maxX,
        minY: floorY == null ? lo - pad : math.max(floorY, lo - pad),
        maxY: hi + pad,
        clipData: const FlClipData.all(),
        lineTouchData: const LineTouchData(enabled: false),
        gridData: FlGridData(
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => const FlLine(color: kChartGrid, strokeWidth: 1),
        ),
        borderData: FlBorderData(show: false),
        rangeAnnotations: RangeAnnotations(verticalRangeAnnotations: bands),
        extraLinesData: ExtraLinesData(verticalLines: [
          VerticalLine(x: now, color: kSignal, strokeWidth: 1.2, dashArray: [4, 3]),
        ]),
        titlesData: FlTitlesData(
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 22,
              interval: xStep,
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                child: Text(v % xStep == 0 && v > minX && v < maxX ? '${v.toInt()} s' : '', style: chartAxis()),
              ),
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 52,
              interval: yTitle != null ? 1 : null,
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                // the close-up's edge values are not round and crowd the nearest tick: leave them out
                child: Text(window && (v == meta.min || v == meta.max) ? '' : (yTitle != null ? yTitle(v) : meta.formattedValue),
                    style: chartAxis()),
              ),
            ),
          ),
        ),
        lineBarsData: [
          if (ref2Spots.isNotEmpty)
            LineChartBarData(
                spots: ref2Spots,
                color: kInkFaint.withValues(alpha: 0.8),
                barWidth: 1,
                dashArray: [1, 3],
                dotData: const FlDotData(show: false)),
          if (refSpots.isNotEmpty)
            LineChartBarData(
                spots: refSpots,
                color: kInkDim.withValues(alpha: 0.7),
                barWidth: 1,
                dashArray: [3, 3],
                dotData: const FlDotData(show: false)),
          LineChartBarData(
              spots: future, color: color.withValues(alpha: 0.14), barWidth: 1, dotData: const FlDotData(show: false)),
          LineChartBarData(spots: past, color: color, barWidth: 1.5, dotData: const FlDotData(show: false)),
        ],
      ),
    );
  }

  List<({double x1, double x2, String phase})> _phaseRuns() {
    final out = <({double x1, double x2, String phase})>[];
    int s = 0;
    for (int k = 1; k <= phasePred.length; k++) {
      if (k == phasePred.length || phasePred[k] != phasePred[s]) {
        out.add((x1: t[s], x2: t[k - 1], phase: phasePred[s]));
        s = k;
      }
    }
    return out;
  }
}

class _Score extends StatelessWidget {
  const _Score(this.label, this.value, {this.color = kInk});
  final String label;
  final String value;
  final Color color;

  @override
  Widget build(BuildContext context) => Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
        Text(label.toUpperCase(), style: consoleLabel(size: 9.5)),
        const SizedBox(height: 3),
        Text(value, style: readout(size: 15, weight: FontWeight.w500, color: color)),
      ]);
}
