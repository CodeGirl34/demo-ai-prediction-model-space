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
          title: const Text('Inject a sensor fault'),
          content: Column(mainAxisSize: MainAxisSize.min, children: [
            DropdownButtonFormField<String>(
              initialValue: kind,
              decoration: const InputDecoration(labelText: 'Fault type'),
              items: [for (final k in _faultKinds) DropdownMenuItem(value: k, child: Text(faultLabels[k]!))],
              onChanged: (v) => setD(() => kind = v!),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: phase,
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
      title: 'Landing replay',
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        SegmentedButton<String>(
          segments: const [
            ButtonSegment(value: 'msl', label: Text('Curiosity'), icon: Icon(Icons.science_outlined)),
            ButtonSegment(value: 'm2020', label: Text('Perseverance')),
          ],
          selected: {mission},
          onSelectionChanged: (s) => setState(() => mission = s.first),
        ),
        const SizedBox(height: 6),
        Text(
          mission == 'msl'
              ? 'Test landing: models were trained on Perseverance only.'
              : 'Training landing: models learned from this one, so results are optimistic.',
          style: TextStyle(color: mission == 'msl' ? kOk : kAlarm, fontSize: 12),
        ),
        const SizedBox(height: 14),
        DropdownButtonFormField<String>(
          initialValue: channel,
          decoration: const InputDecoration(labelText: 'Monitored sensor', border: OutlineInputBorder()),
          items: [
            for (final e in channelLabels.entries) DropdownMenuItem(value: e.key, child: Text(e.value)),
          ],
          onChanged: (v) => setState(() => channel = v!),
        ),
        const SizedBox(height: 14),
        const Text('Injected faults', style: TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
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
          if (faults.isEmpty && randomFaults == 0) const Text('none – clean landing', style: TextStyle(color: kMuted)),
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
              label: const Text('Surprise me'),
            ),
          ),
        ]),
        const SizedBox(height: 14),
        FilledButton.icon(
          onPressed: loading ? null : _run,
          icon: loading
              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.rocket_launch),
          label: Text(loading ? 'Running models…' : 'Run replay'),
        ),
        if (error != null && data != null) ...[
          const SizedBox(height: 8),
          Text('$error', style: const TextStyle(color: kFault)),
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
      _chartCard('Altitude above landing site (log scale)', _altChart(now)),
      _chartCard(channelLabels[channel]!, _lineChart(value, now, kMars, faultBands, alarmBands),
          legend: const [('injected fault', kFault), ('alarm', kAlarm)]),
      _chartCard('Anomaly score (LSTM prediction error, log scale)',
          _lineChart([for (final s in score) s.isNaN ? double.nan : math.log(math.max(s, 1e-5)) / math.ln10],
              now, const Color(0xFFB388FF), const [], alarmBands)),
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
    return SectionCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Wrap(spacing: 20, runSpacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('AI phase estimate', style: TextStyle(color: kMuted, fontSize: 12)),
            const SizedBox(height: 4),
            PhaseChip(pred, large: true),
          ]),
          Metric(label: 'Mission clock', value: fmtT(t[i])),
          Metric(label: 'Altitude', value: alt[i].isNaN ? '–' : fmtAlt(alt[i])),
          Metric(label: 'NASA timeline', value: phaseLabel(truth),
              color: pred == truth ? kOk : kAlarm, caption: pred == truth ? 'AI agrees' : 'AI disagrees'),
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: (alarmActive ? kAlarm : kOk).withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: alarmActive ? kAlarm : kOk),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(alarmActive ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                  color: alarmActive ? kAlarm : kOk),
              const SizedBox(width: 6),
              Text(alarmActive ? 'ANOMALY' : 'NOMINAL',
                  style: TextStyle(color: alarmActive ? kAlarm : kOk, fontWeight: FontWeight.w700)),
            ]),
          ),
        ]),
        const SizedBox(height: 14),
        Row(children: [
          IconButton.filledTonal(
            onPressed: playing ? _pause : _play,
            icon: Icon(playing ? Icons.pause : Icons.play_arrow),
            tooltip: playing ? 'Pause' : 'Play',
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
        const SizedBox(height: 6),
        Row(children: [
          const SizedBox(width: 92, child: Text('AI estimate', style: TextStyle(color: kMuted, fontSize: 12))),
          Expanded(child: PhaseStrip(phases: phasePred, upTo: i)),
        ]),
        const SizedBox(height: 4),
        Row(children: [
          const SizedBox(width: 92, child: Text('NASA timeline', style: TextStyle(color: kMuted, fontSize: 12))),
          Expanded(child: PhaseStrip(phases: phaseTrue)),
        ]),
        const SizedBox(height: 8),
        Wrap(spacing: 10, runSpacing: 4, children: [
          for (final p in _phases)
            Row(mainAxisSize: MainAxisSize.min, children: [
              Container(width: 10, height: 10, color: phaseColors[p]),
              const SizedBox(width: 4),
              Text(phaseLabel(p), style: const TextStyle(fontSize: 12, color: kMuted)),
            ]),
          Text(
              '· phase accuracy ${(((data!['phase_accuracy'] as num?) ?? 0) * 100).toStringAsFixed(1)} %'
              ' · faults caught ${s['faults_detected']}/${s['faults_injected']}'
              ' · false alarms ${s['false_alarms']}',
              style: const TextStyle(fontSize: 12, color: kMuted)),
        ]),
      ]),
    );
  }

  Widget _feed(List<Map<String, dynamic>> seen, List<Map<String, dynamic>> injected, double now) {
    return SectionCard(
      title: 'Alert feed',
      trailing: Tag('${seen.length} alerts', color: seen.isEmpty ? kOk : kAlarm),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        if (seen.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 12),
            child: Text('No anomalies so far. All monitored readings match the model’s expectations.',
                style: TextStyle(color: kMuted)),
          ),
        for (final a in seen) _alertTile(a, injected),
        const Divider(height: 28),
        const Text('Injected faults (ground truth)', style: TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        if (injected.isEmpty) const Text('None – this is a clean landing.', style: TextStyle(color: kMuted)),
        for (final f in injected)
          ListTile(
            dense: true,
            contentPadding: EdgeInsets.zero,
            leading: Icon(Icons.bolt, color: (f['start_t'] as num) <= now ? kFault : kMuted),
            title: Text('${faultLabels[f['kind']]} during ${phaseLabel(f['phase'])}'),
            subtitle: Text('${fmtT(f['start_t'] as num)} → ${fmtT(f['end_t'] as num)}'),
            trailing: (f['end_t'] as num) <= now
                ? Tag(f['detected'] == true ? 'CAUGHT' : 'MISSED', color: f['detected'] == true ? kOk : kFault)
                : const Tag('upcoming', color: kMuted),
          ),
      ]),
    );
  }

  Widget _alertTile(Map<String, dynamic> a, List<Map<String, dynamic>> injected) {
    final m = a['matches_fault'];
    final real = m != null;
    final dets = (a['detectors'] as List).map((d) => d == 'persistence' ? 'baseline' : 'LSTM').join(' + ');
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: kSurfaceHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border(left: BorderSide(color: real ? kAlarm : kMuted, width: 3)),
      ),
      child: Row(children: [
        Icon(Icons.warning_amber_rounded, color: real ? kAlarm : kMuted, size: 20),
        const SizedBox(width: 10),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('${fmtT(a['start_t'] as num)} · ${phaseLabel(a['phase'])}',
                style: const TextStyle(fontWeight: FontWeight.w600)),
            Text(real
                ? 'Matches injected ${faultLabels[injected[m as int]['kind']]!.toLowerCase()} · flagged by $dets'
                : 'No injected fault here (false alarm) · flagged by $dets',
                style: const TextStyle(color: kMuted, fontSize: 12)),
          ]),
        ),
      ]),
    );
  }

  Widget _chartCard(String title, Widget chart, {List<(String, Color)> legend = const []}) {
    return SectionCard(
      padding: 12,
      title: title,
      trailing: legend.isEmpty
          ? null
          : Wrap(spacing: 10, children: [
              for (final (l, c) in legend)
                Row(mainAxisSize: MainAxisSize.min, children: [
                  Container(width: 10, height: 10, color: c.withValues(alpha: 0.6)),
                  const SizedBox(width: 4),
                  Text(l, style: const TextStyle(fontSize: 11, color: kMuted)),
                ]),
            ]),
      child: SizedBox(height: 170, child: chart),
    );
  }

  Widget _altChart(double now) {
    final y = [for (final a in alt) a.isNaN ? double.nan : math.log(math.max(a, 0.005)) / math.ln10];
    return _lineChart(y, now, Colors.white, const [], const [],
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
          VerticalRangeAnnotation(x1: b.x1 - 0.5, x2: math.min(b.x2, now) + 0.5, color: kAlarm.withValues(alpha: 0.35)),
      // drawn last (and never thinner than a few pixels) so a one-sample spike stays visible over its alarm
      for (final b in faultBands)
        VerticalRangeAnnotation(x1: b.x1 - minHalf, x2: b.x2 + minHalf, color: kFault.withValues(alpha: 0.5)),
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
          getDrawingHorizontalLine: (_) => const FlLine(color: Color(0x22FFFFFF), strokeWidth: 0.6),
        ),
        borderData: FlBorderData(show: false),
        rangeAnnotations: RangeAnnotations(verticalRangeAnnotations: bands),
        extraLinesData: ExtraLinesData(verticalLines: [
          VerticalLine(x: now, color: Colors.white70, strokeWidth: 1.2, dashArray: [4, 3]),
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
                child: Text(v % 200 == 0 ? '${v.toInt()} s' : '', style: const TextStyle(fontSize: 10, color: kMuted)),
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
                    style: const TextStyle(fontSize: 10, color: kMuted)),
              ),
            ),
          ),
        ),
        lineBarsData: [
          LineChartBarData(
              spots: future, color: color.withValues(alpha: 0.15), barWidth: 1, dotData: const FlDotData(show: false)),
          LineChartBarData(spots: past, color: color, barWidth: 1.4, dotData: const FlDotData(show: false)),
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
