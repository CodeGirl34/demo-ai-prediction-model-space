import 'package:flutter/material.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets.dart';

class OverviewScreen extends StatefulWidget {
  const OverviewScreen({super.key, required this.api, required this.onOpenReplay, required this.onOpenJourney});
  final Api api;
  final VoidCallback onOpenReplay;
  final VoidCallback onOpenJourney;

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

    final narrow = MediaQuery.sizeOf(context).width < 700;
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(20), children: [
        Text('PHASE-AWARE ANOMALY DETECTION', style: consoleLabel(color: kSignal, size: 10.5)),
        const SizedBox(height: 8),
        ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 820),
          child: Text(
            'An AI that learns how a spacecraft normally behaves, knows which part of the landing it is in, '
            'and raises an alarm when a sensor stops making sense.',
            style: TextStyle(
                fontFamily: kFontDisplay, fontSize: narrow ? 17 : 22, fontWeight: FontWeight.w600, height: 1.3),
          ),
        ),
        const SizedBox(height: 18),
        _HowToRead(onOpenReplay: widget.onOpenReplay, onOpenJourney: widget.onOpenJourney),
        const SizedBox(height: 22),
        _SectionLabel('Results'),
        const SizedBox(height: 10),
        Wrap(spacing: 12, runSpacing: 12, children: [
          if (best != null)
            _Tile(
              title: 'Fault detection on Mars landings',
              icon: Icons.sensors,
              evidence: const [Evidence.ai, Evidence.simulated],
              footer: 'We broke sensors on purpose during a landing the AI had never seen. '
                  'This is how many of those faults it caught.',
              children: [
                Metric(
                    label: 'Faults caught',
                    value: pct(best['recall']),
                    color: kGo,
                    caption:
                        '${(best['tp'] as num).toInt()} of ${((best['tp'] as num) + (best['fn'] as num)).toInt()} injected'),
                Metric(label: 'F1 score', value: (best['f1'] as num).toStringAsFixed(2)),
              ],
            ),
          if (aware != null && blind != null)
            _Tile(
              title: 'Why knowing the phase matters',
              icon: Icons.timeline,
              evidence: const [Evidence.ai],
              footer: 'Hard braking is normal during entry but alarming during cruise. Knowing the phase '
                  'cuts false alarms while catching the same faults.',
              children: [
                Metric(label: 'False alarms · phase-blind', value: '${(blind['fp'] as num).toInt()}', color: kSignal),
                Metric(label: 'Phase-aware', value: '${(aware['fp'] as num).toInt()}', color: kGo),
              ],
            ),
          if (ph != null)
            _Tile(
              title: 'Mission-phase classifier',
              icon: Icons.flight_land,
              evidence: const [Evidence.nasa, Evidence.ai],
              footer: 'Trained on one rover’s landing, tested on the other’s. '
                  'How often it names the right phase.',
              children: [
                Metric(label: 'Perseverance → Curiosity', value: pct(ph['train_m2020_test_msl']?['accuracy'])),
                Metric(label: 'Curiosity → Perseverance', value: pct(ph['train_msl_test_m2020']?['accuracy'])),
              ],
            ),
          if (eng != null)
            _Tile(
              title: 'NASA rover telemetry anomalies',
              icon: Icons.satellite_alt,
              evidence: const [Evidence.nasa, Evidence.ai],
              footer: 'Precision: when it raises an alarm, how often it is right. '
                  'Recall: how many of NASA’s labeled anomalies it found.',
              children: [
                Metric(label: 'Precision', value: pct(eng['precision'])),
                Metric(label: 'Recall', value: pct(eng['recall'])),
              ],
            ),
        ]),
        const SizedBox(height: 26),
        _SectionLabel('Mars landings'),
        const SizedBox(height: 10),
        for (final m in missions!) ...[
          _MissionCard(mission: m as Map<String, dynamic>, onReplay: widget.onOpenReplay),
          const SizedBox(height: 12),
        ],
        const SizedBox(height: 6),
        Text('Data: NASA JPL / NAIF SPICE trajectory reconstructions and SMAP–MSL telemetry. Faults are synthetic.',
            style: readout(size: 11, color: kInkFaint)),
      ]),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Row(children: [
        Text(text.toUpperCase(), style: consoleLabel(color: kInkDim, size: 11, weight: FontWeight.w600)),
        const SizedBox(width: 12),
        const Expanded(child: SeamRule()),
      ]);
}

/// Plain-language guide to the console: what each workspace is for, and what the colours mean.
class _HowToRead extends StatelessWidget {
  const _HowToRead({required this.onOpenReplay, required this.onOpenJourney});
  final VoidCallback onOpenReplay;
  final VoidCallback onOpenJourney;

  @override
  Widget build(BuildContext context) {
    Widget step(String n, String title, String body) => ConstrainedBox(
          constraints: const BoxConstraints(minWidth: 200, maxWidth: 212), // four steps fit on one row
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
              width: 24,
              height: 24,
              alignment: Alignment.center,
              decoration: BoxDecoration(color: kSignalSoft, border: Border.all(color: kSignal.withValues(alpha: 0.5))),
              child: Text(n, style: readout(size: 11, color: kSignal)),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(title.toUpperCase(), style: consoleLabel(color: kInk, size: 11, weight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(body, style: const TextStyle(fontSize: 12.5, color: kInkDim, height: 1.4)),
              ]),
            ),
          ]),
        );

    return SectionCard(
      title: 'How to read this console',
      icon: Icons.help_outline,
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Wrap(spacing: 24, runSpacing: 16, children: [
          step('1', 'Overview', 'You are here. The scores below sum up how well the AI works.'),
          step('2', 'Journey', 'Fly along with Perseverance from Earth to Mars in 3D, on NASA’s real path.'),
          step('3', 'Landing replay',
              'Watch a real Mars landing second by second. Break a sensor on purpose and see if the AI raises an alarm.'),
          step('4', 'Telemetry',
              'Real data from the Curiosity rover. Red marks = problems NASA engineers found; green = what the AI found.'),
        ]),
        const SizedBox(height: 16),
        const SeamRule(),
        const SizedBox(height: 12),
        Wrap(spacing: 18, runSpacing: 10, crossAxisAlignment: WrapCrossAlignment.center, children: [
          Text('EVERY PANEL IS TAGGED WITH ITS SOURCE', style: consoleLabel(size: 9.5)),
          for (final e in Evidence.values)
            Row(mainAxisSize: MainAxisSize.min, children: [
              EvidenceBadge(e),
              const SizedBox(width: 8),
              Text(e.meaning, style: const TextStyle(fontSize: 12, color: kInkDim)),
            ]),
        ]),
        const SizedBox(height: 14),
        Wrap(spacing: 10, runSpacing: 10, children: [
          FilledButton.icon(
            onPressed: onOpenJourney,
            icon: const Icon(Icons.public, size: 18),
            label: const Text('Fly Earth → Mars in 3D'),
          ),
          OutlinedButton.icon(
            onPressed: onOpenReplay,
            icon: const Icon(Icons.play_arrow, size: 18),
            label: const Text('Replay a landing'),
          ),
        ]),
      ]),
    );
  }
}

class _Tile extends StatelessWidget {
  const _Tile(
      {required this.title, required this.icon, required this.children, required this.evidence, this.footer});
  final String title;
  final IconData icon;
  final List<Widget> children;
  final List<Evidence> evidence;
  final String? footer;

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 280, maxWidth: 340),
      child: SectionCard(
        title: title,
        icon: icon,
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Wrap(spacing: 28, runSpacing: 10, children: children),
          if (footer != null) ...[
            const SizedBox(height: 12),
            Text(footer!, style: const TextStyle(color: kInkFaint, fontSize: 12, height: 1.4)),
          ],
          const SizedBox(height: 12),
          Wrap(spacing: 6, children: [for (final e in evidence) EvidenceBadge(e)]),
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
      icon: Icons.public,
      evidence: const [Evidence.nasa],
      trailing: Tag(trained ? 'AI trained on this' : 'Unseen test landing', color: trained ? kInkFaint : kGo),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(
            '${mission['program']} · landed ${mission['landing_date']} · '
            'entered the atmosphere at ${(mission['entry_speed_m_s'] as num).toStringAsFixed(0)} m/s',
            style: readout(size: 12, color: kInkDim)),
        const SizedBox(height: 14),
        Text('KEY MOMENTS (T = TOUCHDOWN)', style: consoleLabel(size: 9.5)),
        const SizedBox(height: 8),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(children: [
            for (int k = 0; k < events.length; k++) ...[
              Container(
                padding: const EdgeInsets.fromLTRB(10, 8, 12, 8),
                decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text((events[k]['event'] as String).replaceAll('_', ' ').toUpperCase(),
                      style: consoleLabel(color: kInk, size: 10, weight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text(fmtT(events[k]['t_s'] as num), style: readout(size: 12)),
                  Text(fmtAlt(events[k]['altitude_km'] as num), style: readout(size: 11, color: kInkFaint)),
                ]),
              ),
              if (k < events.length - 1)
                Container(width: 14, height: 1, color: kSeamLit),
            ],
          ]),
        ),
        const SizedBox(height: 10),
        Align(
          alignment: Alignment.centerRight,
          child: TextButton.icon(
              onPressed: onReplay, icon: const Icon(Icons.play_arrow, size: 16), label: const Text('Replay landing')),
        ),
      ]),
    );
  }
}
