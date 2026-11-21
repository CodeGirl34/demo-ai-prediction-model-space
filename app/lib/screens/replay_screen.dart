import 'dart:async';
import 'dart:math' as math;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets.dart';

const _faultKinds = ['spike', 'bias', 'drift', 'stuck', 'noise', 'dropout'];
const _phases = ['approach', 'entry', 'parachute', 'powered_descent', 'sky_crane'];

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

  Map<String, dynamic>? data;
  bool loading = false;
  Object? error;

  int cursor = 0;
  double _cursorF = 0;
  bool playing = false;
  int speed = 30;
  Timer? _timer;

  // decoded series
  List<double> t = [], alt = [], value = [], score = [];
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
          channel: channel, faults: faults, randomFaults: randomFaults, seed: seed);
      setState(() {
        data = r;
        t = _nums(r['t_s']);
        alt = _nums(r['altitude_km']);
        value = _nums(r['value']);
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
          content: Column(mainAxisSize: MainAxisSize.min, children: [
            DropdownButtonFormField<String>(
              initialValue: kind,
              dropdownColor: kPanel2,
              decoration: const InputDecoration(labelText: 'Fault type'),
              items: [for (final k in _faultKinds) DropdownMenuItem(value: k, child: Text(faultLabels[k]!))],
              onChanged: (v) => setD(() => kind = v!),
            ),
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

    final charts = [
      _chartCard('Altitude above landing site', _altChart(now),
          evidence: const [Evidence.nasa], note: 'log scale · background colour = AI’s phase estimate'),
      _chartCard('Sensor: ${channelLabels[channel]!}', _lineChart(value, now, kObserved, faultBands, alarmBands),
          evidence: const [Evidence.nasa, Evidence.simulated],
          legend: const [('reading', kObserved), ('injected fault', kScenario), ('AI alarm', kCaution)]),
      _chartCard('AI surprise score',
          _lineChart([for (final s in score) s.isNaN ? double.nan : math.log(math.max(s, 1e-5)) / math.ln10],
              now, kDerived, const [], alarmBands),
          evidence: const [Evidence.ai],
          note: 'how far the reading is from what the AI expected · log scale · high = suspicious'),
    ];

    final feed = _feed(seen, injected, now);
    final content = Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _status(i, active),
      const SizedBox(height: 12),
      if (wide)
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(flex: 3, child: Column(children: [for (final c in charts) ...[c, const SizedBox(height: 12)]])),
          const SizedBox(width: 12),
          Expanded(flex: 2, child: feed),
        ])
      else ...[
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
          for (final f in injected)
            Container(
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
                    Text('${fmtT(f['start_t'] as num)} → ${fmtT(f['end_t'] as num)}',
                        style: readout(size: 11, color: kInkFaint)),
                  ]),
                ),
                (f['end_t'] as num) <= now
                    ? Tag(f['detected'] == true ? 'Caught' : 'Missed', color: f['detected'] == true ? kGo : kCaution)
                    : const Tag('Upcoming', color: kInkFaint),
              ]),
            ),
        ]),
      ),
    ]);
  }

  Widget _alertTile(Map<String, dynamic> a, List<Map<String, dynamic>> injected) {
    final m = a['matches_fault'];
    final real = m != null;
    final dets = (a['detectors'] as List).map((d) => d == 'persistence' ? 'baseline' : 'LSTM').join(' + ');
    final c = real ? kCaution : kSignal;
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
                    : 'False alarm: nothing was broken here.',
                style: const TextStyle(fontSize: 12.5, color: kInkDim)),
            const SizedBox(height: 2),
            Text('flagged by $dets', style: readout(size: 10.5, color: kInkFaint)),
          ]),
        ),
        const SizedBox(width: 8),
        Tag(real ? 'Correct' : 'False alarm', color: c),
      ]),
    );
  }


  Widget _chartCard(String title, Widget chart,
      {List<(String, Color)> legend = const [], List<Evidence> evidence = const [], String? note}) {
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
  }) {
    final past = <FlSpot>[], future = <FlSpot>[];
    double lo = double.infinity, hi = -double.infinity;
    for (int k = 0; k < ys.length; k++) {
      final v = ys[k];
      if (v.isNaN) continue;
      lo = math.min(lo, v);
      hi = math.max(hi, v);
      final s = FlSpot(t[k], v);
      if (t[k] <= now) past.add(s);
      if (t[k] >= now) future.add(s);
    }
    if (lo == double.infinity) {
      lo = 0;
      hi = 1;
    }
    final pad = (hi - lo) * 0.08 + 1e-9;
    final minHalf = math.max(0.5, (t.last - t.first) * 0.002);
    final bands = <VerticalRangeAnnotation>[
      if (phaseBands)
        for (final b in _phaseRuns())
          VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: phaseColors[b.phase]!.withValues(alpha: b.x1 <= now ? 0.16 : 0.05)),
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
        minX: t.first,
        maxX: t.last,
        minY: lo - pad,
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
              interval: 200,
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                child: Text(v % 200 == 0 ? '${v.toInt()} s' : '', style: chartAxis()),
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
                child: Text(yTitle != null ? yTitle(v) : meta.formattedValue,
                    style: chartAxis()),
              ),
            ),
          ),
        ),
        lineBarsData: [
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
