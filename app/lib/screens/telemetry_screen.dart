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
        final sel = id == selected;
        return InkWell(
          onTap: () => _open(id),
          child: Container(
            padding: const EdgeInsets.fromLTRB(12, 9, 12, 9),
            decoration: BoxDecoration(
              color: sel ? kSignalSoft.withValues(alpha: 0.45) : Colors.transparent,
              border: Border(
                left: BorderSide(color: sel ? kSignal : Colors.transparent, width: 2),
                bottom: const BorderSide(color: kSeam),
              ),
            ),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Padding(
                padding: const EdgeInsets.only(top: 5),
                child: LedDot(color: fn > 0 ? kCaution : (fp > 0 ? kSignal : kGo), size: 6),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    Text(id, style: readout(size: 13, weight: FontWeight.w500, color: sel ? kSignal : kInk)),
                    const Spacer(),
                    Wrap(spacing: 4, children: [
                      if (tp > 0) Tag('$tp found', color: kGo),
                      if (fp > 0) Tag('$fp false', color: kSignal),
                      if (fn > 0) Tag('$fn missed', color: kCaution),
                    ]),
                  ]),
                  const SizedBox(height: 3),
                  Text('${ch['num_true_anoms']} labeled anomal${ch['num_true_anoms'] == 1 ? 'y' : 'ies'}',
                      style: readout(size: 10.5, color: kInkFaint)),
                ]),
              ),
            ]),
          ),
        );
      },
    );
    return Padding(
      padding: wide ? const EdgeInsets.all(16) : EdgeInsets.zero,
      child: DecoratedBox(
        decoration: ShapeDecoration(color: kPanel, shape: chamfer(), shadows: kPanelShadow),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 10),
            child: Row(children: [
              const Icon(Icons.satellite_alt, size: 14, color: kSignal),
              const SizedBox(width: 8),
              Expanded(
                  child: Text('CURIOSITY SENSORS',
                      style: consoleLabel(color: kInk, size: 11.5, weight: FontWeight.w600))),
              const EvidenceBadge(Evidence.nasa),
            ]),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
            child: Text('${channels!.length} channels · pick one to inspect',
                style: readout(size: 10.5, color: kInkFaint)),
          ),
          const SeamRule(),
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
        title: 'Sensor ${d['chan_id']}',
        icon: Icons.sensors,
        evidence: const [Evidence.nasa, Evidence.ai],
        child: Wrap(spacing: 28, runSpacing: 12, children: [
          Metric(label: 'Anomalies found', value: '${d['true_positives']}', color: kGo),
          Metric(label: 'False alarms', value: '${d['false_positives']}', color: kSignal),
          Metric(label: 'Missed', value: '${d['false_negatives']}', color: kCaution),
          const SizedBox(
            width: 400,
            child: Text(
              'The AI reads the last 250 values and guesses the next one. When its guesses start going badly '
              'wrong, it raises an alarm. Red = where NASA engineers say something really went wrong. '
              'Green = where the AI raised an alarm. Overlap means the AI found it.',
              style: TextStyle(color: kInkDim, fontSize: 12.5, height: 1.45),
            ),
          ),
        ]),
      ),
      const SizedBox(height: 12),
      SectionCard(
        padding: 14,
        title: 'Reading vs AI guess',
        evidence: const [Evidence.nasa, Evidence.ai],
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Legend([
            ('NASA reading', kObserved),
            ('AI guess', kDerived),
            ('NASA-labeled anomaly', kCaution),
            ('AI alarm', kGo),
          ]),
          const SizedBox(height: 10),
          _plot(240, _chart(idx, [(value, kObserved, 1.1), (pred, kDerived, 1.2)], labeled, detected)),
        ]),
      ),
      const SizedBox(height: 12),
      SectionCard(
        padding: 14,
        title: 'How surprised the AI is',
        evidence: const [Evidence.ai],
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('smoothed prediction error · the alarm fires when this jumps well above its recent normal level',
              style: readout(size: 10.5, color: kInkFaint)),
          const SizedBox(height: 10),
          _plot(160, _chart(idx, [(err, kDerived, 1.2)], labeled, detected)),
        ]),
      ),
    ]);
  }

  Widget _plot(double h, Widget chart) => Container(
        height: h,
        padding: const EdgeInsets.fromLTRB(0, 8, 8, 0),
        decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
        child: chart,
      );

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
          getDrawingHorizontalLine: (_) => const FlLine(color: kChartGrid, strokeWidth: 1),
        ),
        rangeAnnotations: RangeAnnotations(verticalRangeAnnotations: [
          for (final b in labeled) VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: kCaution.withValues(alpha: 0.22)),
          for (final b in detected)
            VerticalRangeAnnotation(x1: b.x1, x2: b.x2, color: kGo.withValues(alpha: 0.2)),
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
                    style: chartAxis()),
              ),
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 44,
              getTitlesWidget: (v, meta) => SideTitleWidget(
                meta: meta,
                child: Text(meta.formattedValue, style: chartAxis()),
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
