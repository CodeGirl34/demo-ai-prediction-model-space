import 'package:flutter/material.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets.dart';

class OverviewScreen extends StatefulWidget {
  const OverviewScreen({super.key, required this.api, required this.onOpenReplay});
  final Api api;
  final VoidCallback onOpenReplay;

  @override
  State<OverviewScreen> createState() => _OverviewScreenState();
}

class _OverviewScreenState extends State<OverviewScreen> {
  Map<String, dynamic>? summary;
  List<dynamic>? missions;
  Object? error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => error = null);
    try {
      final r = await Future.wait([widget.api.summary(), widget.api.missions()]);
      setState(() {
        summary = r[0] as Map<String, dynamic>;
        missions = r[1] as List<dynamic>;
      });
    } catch (e) {
      setState(() => error = e);
    }
  }

  String pct(dynamic v) => v == null ? '–' : '${((v as num) * 100).toStringAsFixed(0)} %';

  @override
  Widget build(BuildContext context) {
    if (summary == null || missions == null) return LoadingOrError(error: error, onRetry: _load);
    final s = summary!;
    final eng = s['anomaly_engine'] as Map<String, dynamic>?;
    final ph = s['phase_classifier'] as Map<String, dynamic>?;
    final fd = s['fault_detection'] as Map<String, dynamic>?;
    final best = fd?['or_persistence_aware'] as Map<String, dynamic>?;
    final blind = fd?['lstm_phase_blind'] as Map<String, dynamic>?;
    final aware = fd?['lstm_phase_aware'] as Map<String, dynamic>?;

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('Phase-aware spacecraft anomaly detection',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 4),
        const Text(
          'Learns how a spacecraft normally behaves, knows which mission phase it is in, '
          'and raises an alarm when a sensor stops making sense.',
          style: TextStyle(color: kMuted),
        ),
        const SizedBox(height: 16),
        Wrap(spacing: 12, runSpacing: 12, children: [
          if (best != null)
            _Tile(
              width: 300,
              title: 'Fault detection on Mars landings',
              icon: Icons.sensors,
              footer: 'Baseline OR phase-aware LSTM, tested on the other rover.',
              children: [
                Metric(label: 'Faults caught', value: pct(best['recall']), color: kOk,
                    caption: '${(best['tp'] as num).toInt()} of ${((best['tp'] as num) + (best['fn'] as num)).toInt()} injected faults'),
                Metric(label: 'F1', value: (best['f1'] as num).toStringAsFixed(2)),
              ],
            ),
          if (aware != null && blind != null)
            _Tile(
              width: 300,
              title: 'Why phase awareness matters',
              icon: Icons.timeline,
              footer: '≈30 % fewer false alarms while catching the same faults.',
              children: [
                Metric(label: 'False alarms, phase-blind', value: '${(blind['fp'] as num).toInt()}', color: kAlarm),
                Metric(label: 'Phase-aware', value: '${(aware['fp'] as num).toInt()}', color: kOk),
              ],
            ),
          if (ph != null)
            _Tile(
              width: 300,
              title: 'Mission-phase classifier',
              icon: Icons.flight_land,
              footer: 'Trained on one landing, tested on the other.',
              children: [
                Metric(label: 'Perseverance → Curiosity', value: pct(ph['train_m2020_test_msl']?['accuracy'])),
                Metric(label: 'Curiosity → Perseverance', value: pct(ph['train_msl_test_m2020']?['accuracy'])),
              ],
            ),
          if (eng != null)
            _Tile(
              width: 300,
              title: 'NASA MSL telemetry anomalies',
              icon: Icons.satellite_alt,
              footer: 'LSTM + dynamic threshold (Telemanom method), expert-labeled anomalies.',
              children: [
                Metric(label: 'Precision', value: pct(eng['precision'])),
                Metric(label: 'Recall', value: pct(eng['recall'])),
              ],
            ),
        ]),
        const SizedBox(height: 20),
        Text('Mars landings', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        for (final m in missions!) ...[
          _MissionCard(mission: m as Map<String, dynamic>, onReplay: widget.onOpenReplay),
          const SizedBox(height: 12),
        ],
        const SizedBox(height: 8),
        Text('Data: NASA JPL / NAIF SPICE reconstructions and SMAP–MSL telemetry. Faults are synthetic.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: kMuted)),
      ]),
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile({required this.width, required this.title, required this.icon, required this.children, this.footer});
  final double width;
  final String title;
  final IconData icon;
  final List<Widget> children;
  final String? footer;

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: BoxConstraints(minWidth: 260, maxWidth: width),
      child: SectionCard(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(icon, size: 18, color: kMars),
            const SizedBox(width: 8),
            Expanded(child: Text(title, style: const TextStyle(fontWeight: FontWeight.w600))),
          ]),
          const SizedBox(height: 12),
          Wrap(spacing: 24, runSpacing: 8, children: children),
          if (footer != null) ...[
            const SizedBox(height: 10),
            Text(footer!, style: const TextStyle(color: kMuted, fontSize: 12)),
          ],
        ]),
      ),
    );
  }
}

class _MissionCard extends StatelessWidget {
  const _MissionCard({required this.mission, required this.onReplay});
  final Map<String, dynamic> mission;
  final VoidCallback onReplay;

  @override
  Widget build(BuildContext context) {
    final events = (mission['events'] as List).cast<Map<String, dynamic>>();
    final trained = mission['models_trained_on_this_landing'] == true;
    return SectionCard(
      title: '${mission['name']} · ${mission['site']}',
      trailing: Tag(trained ? 'training landing' : 'test landing', color: trained ? kMuted : kOk),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('${mission['program']} · landed ${mission['landing_date']} · '
            'entry at ${(mission['entry_speed_m_s'] as num).toStringAsFixed(0)} m/s',
            style: const TextStyle(color: kMuted)),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(children: [
            for (final e in events)
              Padding(
                padding: const EdgeInsets.only(right: 18),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text((e['event'] as String).replaceAll('_', ' '),
                      style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
                  Text(fmtT(e['t_s'] as num), style: const TextStyle(color: kMuted, fontSize: 12)),
                  Text(fmtAlt(e['altitude_km'] as num), style: const TextStyle(color: kMuted, fontSize: 12)),
                ]),
              ),
          ]),
        ),
        const SizedBox(height: 8),
        Align(
          alignment: Alignment.centerRight,
          child: TextButton.icon(onPressed: onReplay, icon: const Icon(Icons.play_arrow), label: const Text('Replay landing')),
        ),
      ]),
    );
  }
}
