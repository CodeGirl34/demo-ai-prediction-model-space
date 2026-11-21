import 'dart:math' as math;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets.dart';

/// Module 1: NASA MSL telemetry channels with expert-labeled anomalies.
class TelemetryScreen extends StatefulWidget {
  const TelemetryScreen({super.key, required this.api});
  final Api api;

  @override
  State<TelemetryScreen> createState() => _TelemetryScreenState();
}

class _TelemetryScreenState extends State<TelemetryScreen> {
  List<Map<String, dynamic>>? channels;
  Object? error;
  String? selected;
  Map<String, dynamic>? detail;
  Object? detailError;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => error = null);
    try {
      final r = (await widget.api.telemetryChannels()).cast<Map<String, dynamic>>();
      setState(() => channels = r);
      if (r.isNotEmpty) _open(r.first['chan_id'] as String);
    } catch (e) {
      setState(() => error = e);
    }
  }

  Future<void> _open(String id) async {
    setState(() {
      selected = id;
      detail = null;
      detailError = null;
    });
    try {
      final d = await widget.api.telemetryChannel(id);
      if (selected == id) setState(() => detail = d);
    } catch (e) {
      if (selected == id) setState(() => detailError = e);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (channels == null) return LoadingOrError(error: error, onRetry: _load);
    return LayoutBuilder(builder: (context, c) {
      final wide = c.maxWidth > 900;
      final list = _channelList(wide);
      final det = _detail();
      if (wide) {
        return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SizedBox(width: 300, child: list),
          Expanded(child: ListView(padding: const EdgeInsets.fromLTRB(0, 16, 16, 16), children: [det])),
        ]);
      }
      return ListView(padding: const EdgeInsets.all(16), children: [
        SizedBox(height: 260, child: list),
        const SizedBox(height: 12),
        det,
      ]);
    });
  }

  Widget _channelList(bool wide) {
    final tiles = ListView.builder(
      itemCount: channels!.length,
      itemBuilder: (context, i) {
        final ch = channels![i];
        final id = ch['chan_id'] as String;
        final tp = ch['true_positives'] as int, fp = ch['false_positives'] as int, fn = ch['false_negatives'] as int;
        return ListTile(
          dense: true,
          selected: id == selected,
          selectedTileColor: kSurfaceHigh,
          title: Text(id, style: const TextStyle(fontWeight: FontWeight.w600)),
          subtitle: Text('${ch['num_true_anoms']} labeled anomal${ch['num_true_anoms'] == 1 ? 'y' : 'ies'}'),
          trailing: Wrap(spacing: 4, children: [
            if (tp > 0) Tag('$tp TP', color: kOk),
            if (fp > 0) Tag('$fp FP', color: kAlarm),
            if (fn > 0) Tag('$fn FN', color: kFault),
          ]),
          onTap: () => _open(id),
        );
      },
    );
    return Padding(
      padding: wide ? const EdgeInsets.all(16) : EdgeInsets.zero,
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        clipBehavior: Clip.antiAlias,
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(16, 14, 16, 6),
            child: Text('Curiosity (MSL) telemetry channels', style: TextStyle(fontWeight: FontWeight.w600)),
          ),
          Expanded(child: tiles),
        ]),
      ),
    );
  }

  Widget _detail() {
    if (detail == null) {
      return SectionCard(
          title: selected, child: SizedBox(height: 300, child: LoadingOrError(error: detailError, onRetry: () => _open(selected!))));
    }
    final d = detail!;
    final idx = [for (final v in d['index'] as List) (v as num).toDouble()];
    List<double> nums(String k) => [for (final v in d[k] as List) v == null ? double.nan : (v as num).toDouble()];
    final value = nums('value'), pred = nums('prediction'), err = nums('error');
    List<({double x1, double x2})> bands(String k) =>
        [for (final s in d[k] as List) (x1: (s[0] as num).toDouble(), x2: (s[1] as num).toDouble())];
    final labeled = bands('labeled_anomalies'), detected = bands('detected_anomalies');

    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      SectionCard(
        title: 'Channel ${d['chan_id']}',
        child: Wrap(spacing: 28, runSpacing: 10, children: [
          Metric(label: 'Anomalies caught', value: '${d['true_positives']}', color: kOk),
          Metric(label: 'False alarms', value: '${d['false_positives']}', color: kAlarm),
          Metric(label: 'Missed', value: '${d['false_negatives']}', color: kFault),
          const SizedBox(
            width: 380,
            child: Text(
              'The LSTM predicts each next reading from the previous 250. When the smoothed prediction error '
              'crosses a dynamic threshold, an anomaly is flagged. Red bands are NASA expert labels; green bands are '
              'what the model detected.',
              style: TextStyle(color: kMuted, fontSize: 12),
            ),
          ),
        ]),
      ),
      const SizedBox(height: 12),
      SectionCard(
        padding: 12,
        title: 'Telemetry value vs LSTM prediction',
        trailing: const _Legend([('actual', Colors.white70), ('predicted', kMars), ('labeled', kFault), ('detected', kOk)]),
        child: SizedBox(
          height: 240,
          child: _chart(idx, [(value, Colors.white70, 1.0), (pred, kMars, 1.2)], labeled, detected),
        ),
      ),
      const SizedBox(height: 12),
      SectionCard(
        padding: 12,
        title: 'Smoothed prediction error',
        child: SizedBox(height: 160, child: _chart(idx, [(err, const Color(0xFFB388FF), 1.2)], labeled, detected)),
      ),
    ]);
  }

  Widget _chart(List<double> x, List<(List<double>, Color, double)> series, List<({double x1, double x2})> labeled,
      List<({double x1, double x2})> detected) {
    double lo = double.infinity, hi = -double.infinity;
    final bars = <LineChartBarData>[];
    for (final (ys, color, w) in series) {
      final spots = <FlSpot>[];
      for (int k = 0; k < ys.length && k < x.length; k++) {
        if (ys[k].isNaN) continue;
        lo = math.min(lo, ys[k]);
        hi = math.max(hi, ys[k]);
        spots.add(FlSpot(x[k], ys[k]));
      }
      bars.add(LineChartBarData(spots: spots, color: color, barWidth: w, dotData: const FlDotData(show: false)));
    }
    if (lo == double.infinity) {
      lo = 0;
      hi = 1;
    }
    final pad = (hi - lo) * 0.08 + 1e-9;
    final span = x.isEmpty ? 1.0 : x.last - x.first;
    return LineChart(
      duration: Duration.zero,
      LineChartData(
        minX: x.isEmpty ? 0 : x.first,
        maxX: x.isEmpty ? 1 : x.last,
        minY: lo - pad,
        maxY: hi + pad,
        clipData: const FlClipData.all(),
        lineTouchData: const LineTouchData(enabled: false),
        borderData: FlBorderData(show: false),
        gridData: FlGridData(
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => const FlLine(color: Color(0x22FFFFFF), strokeWidth: 0.6),
        ),
        rangeAnnotations: RangeAnnotations(verticalRangeAnnotations: [
          for (final b in labeled) VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: kFault.withValues(alpha: 0.25)),
          for (final b in detected)
            VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: kOk.withValues(alpha: 0.22)),
        ]),
        titlesData: FlTitlesData(
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 22,
              interval: _niceInterval(span),
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                // skip the auto-added min/max labels that collide with the regular ticks
                child: Text(v % _niceInterval(span) == 0 ? '${v.toInt()}' : '',
                    style: const TextStyle(fontSize: 10, color: kMuted)),
              ),
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 44,
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                child: Text(meta.formattedValue, style: const TextStyle(fontSize: 10, color: kMuted)),
              ),
            ),
          ),
        ),
        lineBarsData: bars,
      ),
    );
  }

  double _niceInterval(double span) {
    if (span <= 0) return 1;
    final raw = span / 6;
    final mag = math.pow(10, (math.log(raw) / math.ln10).floor()).toDouble();
    for (final m in [1, 2, 5, 10]) {
      if (raw <= m * mag) return m * mag;
    }
    return 10 * mag;
  }
}

class _Legend extends StatelessWidget {
  const _Legend(this.items);
  final List<(String, Color)> items;

  @override
  Widget build(BuildContext context) => Wrap(spacing: 10, children: [
        for (final (l, c) in items)
          Row(mainAxisSize: MainAxisSize.min, children: [
            Container(width: 10, height: 10, color: c.withValues(alpha: 0.7)),
            const SizedBox(width: 4),
            Text(l, style: const TextStyle(fontSize: 11, color: kMuted)),
          ]),
      ]);
}
