import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api.dart';
import 'screens/overview_screen.dart';
import 'screens/replay_screen.dart';
import 'screens/telemetry_screen.dart';
import 'theme.dart';
import 'widgets.dart';

void main() => runApp(SpacecraftAiApp(api: Api()));

class SpacecraftAiApp extends StatelessWidget {
  const SpacecraftAiApp({super.key, required this.api});
  final Api api;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Spacecraft AI',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: HomeShell(api: api),
      );
}

/// One workspace = one screen. `explain` is the one-line "what am I looking at".
class _Workspace {
  const _Workspace(this.label, this.icon, this.explain, this.evidence);
  final String label;
  final IconData icon;
  final String explain;
  final List<Evidence> evidence;
}

const _workspaces = [
  _Workspace('Overview', Icons.space_dashboard_outlined,
      'The headline results, and the two real Mars landings the AI learned from.', [Evidence.nasa, Evidence.ai]),
  _Workspace('Landing replay', Icons.flight_land_outlined,
      'Replay a real Mars landing, break a sensor on purpose, and watch whether the AI notices.',
      [Evidence.nasa, Evidence.ai, Evidence.simulated]),
  _Workspace('Telemetry', Icons.satellite_alt_outlined,
      'Real Curiosity rover telemetry. Did the AI find the anomalies NASA engineers labeled?',
      [Evidence.nasa, Evidence.ai]),
];

enum _Link { checking, online, offline }

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.api});
  final Api api;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int tab = 0;
  _Link link = _Link.checking;
  String? trainedOn;

  @override
  void initState() {
    super.initState();
    _checkLink();
  }

  Future<void> _checkLink() async {
    setState(() => link = _Link.checking);
    try {
      final h = await widget.api.health();
      if (!mounted) return;
      setState(() {
        link = h['status'] == 'ok' ? _Link.online : _Link.offline;
        trainedOn = h['models_trained_on']?.toString();
      });
    } catch (_) {
      if (mounted) setState(() => link = _Link.offline);
    }
  }

  void _go(int i) => setState(() => tab = i);

  @override
  Widget build(BuildContext context) {
    final pages = [
      OverviewScreen(api: widget.api, onOpenReplay: () => _go(1)),
      ReplayScreen(api: widget.api),
      TelemetryScreen(api: widget.api),
    ];
    final wide = MediaQuery.sizeOf(context).width >= 1024;
    final ws = _workspaces[tab];
    final body = IndexedStack(index: tab, children: pages);

    final shell = wide
        ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            _StationRail(tab: tab, onSelect: _go, link: link, trainedOn: trainedOn, onRetryLink: _checkLink),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _HeaderStrip(ws: ws, index: tab),
                Expanded(child: ColoredBox(color: kVoid, child: body)),
                const _StatusBar(),
              ]),
            ),
          ])
        : Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            _MobileTopBar(link: link),
            _MobileTabs(tab: tab, onSelect: _go),
            _HeaderStrip(ws: ws, index: tab, compact: true),
            Expanded(child: ColoredBox(color: kVoid, child: body)),
          ]);

    // Console shortcuts: 1–3 switch workspaces.
    return CallbackShortcuts(
      bindings: {
        for (int i = 0; i < _workspaces.length; i++)
          SingleActivator(LogicalKeyboardKey(LogicalKeyboardKey.digit1.keyId + i)): () => _go(i),
      },
      child: Focus(autofocus: true, child: Scaffold(backgroundColor: kVoid, body: SafeArea(child: shell))),
    );
  }
}

// -----------------------------------------------------------------------------
// Brand

class BrandMark extends StatelessWidget {
  const BrandMark({super.key, this.compact = false});
  final bool compact;

  @override
  Widget build(BuildContext context) => Row(mainAxisSize: MainAxisSize.min, children: [
        const SizedBox(width: 30, height: 30, child: CustomPaint(painter: _MarkPainter())),
        const SizedBox(width: 12),
        Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
          Text('SPACECRAFT AI',
              style: consoleLabel(color: kInk, size: compact ? 13 : 14.5, weight: FontWeight.w700)
                  .copyWith(letterSpacing: 2.6)),
          if (!compact) ...[
            const SizedBox(height: 3),
            Text('ANOMALY CONSOLE', style: consoleLabel(size: 9)),
          ],
        ]),
      ]);
}

/// Mars limb with a descent trajectory curving in to a lit landing point.
class _MarkPainter extends CustomPainter {
  const _MarkPainter();

  @override
  void paint(Canvas canvas, Size s) {
    final r = Rect.fromLTWH(0, 0, s.width, s.height);
    canvas.drawRRect(RRect.fromRectAndRadius(r, const Radius.circular(5)), Paint()..color = kHull);
    canvas.drawRRect(
        RRect.fromRectAndRadius(r.deflate(1.5), const Radius.circular(4)),
        Paint()
          ..color = kSeamLit
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1);
    canvas.save();
    canvas.clipRRect(RRect.fromRectAndRadius(r.deflate(2), const Radius.circular(4)));
    final planet = Offset(s.width * 0.5, s.height * 1.32);
    canvas.drawCircle(planet, s.width * 0.62, Paint()..color = const Color(0xFF3A1717));
    canvas.drawCircle(
        planet,
        s.width * 0.62,
        Paint()
          ..color = const Color(0xFFE5484D)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.6);
    canvas.restore();
    final path = Path()
      ..moveTo(s.width * 0.16, s.height * 0.2)
      ..quadraticBezierTo(s.width * 0.62, s.height * 0.18, s.width * 0.58, s.height * 0.66);
    canvas.drawPath(
        path,
        Paint()
          ..color = kSignal.withValues(alpha: 0.85)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.6
          ..strokeCap = StrokeCap.round);
    canvas.drawCircle(Offset(s.width * 0.58, s.height * 0.68), 2.6, Paint()..color = kSignal);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// -----------------------------------------------------------------------------
// Desktop station rail

class _StationRail extends StatelessWidget {
  const _StationRail(
      {required this.tab, required this.onSelect, required this.link, required this.trainedOn, required this.onRetryLink});
  final int tab;
  final ValueChanged<int> onSelect;
  final _Link link;
  final String? trainedOn;
  final VoidCallback onRetryLink;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 248,
      decoration: const BoxDecoration(color: kHull, border: Border(right: BorderSide(color: kSeam))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Container(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 18),
          decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: kSeam))),
          child: const BrandMark(),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 10),
          child: _SystemStatus(link: link, trainedOn: trainedOn, onRetry: onRetryLink),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 6),
          child: Text('WORKSPACES', style: consoleLabel(size: 9.5)),
        ),
        for (int i = 0; i < _workspaces.length; i++)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
            child: _NavButton(ws: _workspaces[i], index: i, selected: i == tab, onTap: () => onSelect(i)),
          ),
        const Spacer(),
        const _ClockBlock(),
      ]),
    );
  }
}

class _SystemStatus extends StatelessWidget {
  const _SystemStatus({required this.link, required this.trainedOn, required this.onRetry});
  final _Link link;
  final String? trainedOn;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final (color, label) = switch (link) {
      _Link.online => (kGo, 'LINK NOMINAL'),
      _Link.checking => (kSignal, 'CONNECTING'),
      _Link.offline => (kCaution, 'API OFFLINE'),
    };
    Widget row(String k, String v) => Padding(
          padding: const EdgeInsets.only(top: 5),
          child: Row(children: [
            Text(k, style: readout(size: 11, color: kInkFaint)),
            const SizedBox(width: 8),
            Expanded(
                child: Text(v,
                    textAlign: TextAlign.right, overflow: TextOverflow.ellipsis, style: readout(size: 11, color: kInkDim))),
          ]),
        );
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
      decoration: BoxDecoration(color: kInset.withValues(alpha: 0.6), border: Border.all(color: kSeam)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          LedDot(color: color, pulse: link != _Link.offline),
          const SizedBox(width: 8),
          Expanded(child: Text(label, style: consoleLabel(color: kInk))),
          if (link == _Link.offline)
            InkWell(
                onTap: onRetry,
                child: Padding(
                    padding: const EdgeInsets.all(2), child: Text('RETRY', style: consoleLabel(color: kSignal, size: 9.5)))),
        ]),
        const SizedBox(height: 4),
        row('Trained on', trainedOn == null ? '—' : _mission(trainedOn!)),
        row('Landings', '2 on Mars'),
        row('Telemetry', '27 sensors'),
      ]),
    );
  }

  static String _mission(String id) => switch (id.toLowerCase()) {
        'm2020' => 'Perseverance',
        'msl' => 'Curiosity',
        _ => id,
      };
}

class _NavButton extends StatefulWidget {
  const _NavButton({required this.ws, required this.index, required this.selected, required this.onTap});
  final _Workspace ws;
  final int index;
  final bool selected;
  final VoidCallback onTap;

  @override
  State<_NavButton> createState() => _NavButtonState();
}

class _NavButtonState extends State<_NavButton> {
  bool hover = false;

  @override
  Widget build(BuildContext context) {
    final sel = widget.selected;
    final fg = sel ? kSignal : (hover ? kInk : kInkDim);
    return Semantics(
      selected: sel,
      button: true,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        onEnter: (_) => setState(() => hover = true),
        onExit: (_) => setState(() => hover = false),
        child: GestureDetector(
          onTap: widget.onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 140),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
            decoration: BoxDecoration(
              color: sel ? kSignalSoft.withValues(alpha: 0.5) : Colors.transparent,
              border: Border.all(
                  color: sel ? kSignal.withValues(alpha: 0.5) : (hover ? kSeamLit : Colors.transparent)),
            ),
            child: Row(children: [
              Icon(widget.ws.icon, size: 15, color: sel ? kSignal : kInkFaint),
              const SizedBox(width: 12),
              Expanded(child: Text(widget.ws.label.toUpperCase(), style: consoleLabel(color: fg))),
              Text('${widget.index + 1}', style: readout(size: 10, color: kInkFaint)),
            ]),
          ),
        ),
      ),
    );
  }
}

class _ClockBlock extends StatefulWidget {
  const _ClockBlock();

  @override
  State<_ClockBlock> createState() => _ClockBlockState();
}

class _ClockBlockState extends State<_ClockBlock> {
  DateTime now = DateTime.now().toUtc();
  Timer? _t;

  @override
  void initState() {
    super.initState();
    _t = Timer.periodic(const Duration(seconds: 1), (_) => setState(() => now = DateTime.now().toUtc()));
  }

  @override
  void dispose() {
    _t?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    String two(int v) => v.toString().padLeft(2, '0');
    // Curiosity landed 2012-08-06 05:17:57 UTC; elapsed Earth days since touchdown.
    final days = now.difference(DateTime.utc(2012, 8, 6, 5, 17, 57)).inDays;
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 18),
      decoration: const BoxDecoration(border: Border(top: BorderSide(color: kSeam))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('UTC · EARTH', style: consoleLabel(size: 9.5)),
        const SizedBox(height: 4),
        Text('${two(now.hour)}:${two(now.minute)}:${two(now.second)}', style: readout(size: 18)),
        const SizedBox(height: 4),
        Text('Curiosity on Mars: ${math.max(0, days)} days', style: readout(size: 11, color: kInkFaint)),
      ]),
    );
  }
}

// -----------------------------------------------------------------------------
// Header strip + status bar

class _HeaderStrip extends StatelessWidget {
  const _HeaderStrip({required this.ws, required this.index, this.compact = false});
  final _Workspace ws;
  final int index;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: compact ? 16 : 20, vertical: compact ? 10 : 12),
      decoration: BoxDecoration(
          color: kHull.withValues(alpha: 0.6), border: const Border(bottom: BorderSide(color: kSeam))),
      child: Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
        if (!compact) ...[
          Text('0${index + 1}', style: readout(size: 11, color: kSignal)),
          const SizedBox(width: 12),
          Text(ws.label.toUpperCase(), style: consoleLabel(color: kInk, size: 12, weight: FontWeight.w600)),
          const SizedBox(width: 14),
          Container(width: 1, height: 14, color: kSeamLit),
          const SizedBox(width: 14),
        ],
        Expanded(
          child: Text(ws.explain,
              maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: compact ? 12 : 13, color: kInkDim)),
        ),
        if (!compact) ...[
          const SizedBox(width: 16),
          Text('SOURCES', style: consoleLabel(size: 9)),
          for (final e in ws.evidence) ...[const SizedBox(width: 6), EvidenceBadge(e)],
        ],
      ]),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar();

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 9),
        decoration: const BoxDecoration(color: kHull, border: Border(top: BorderSide(color: kSeam))),
        child: Row(children: [
          Expanded(
            child: Text.rich(
              TextSpan(children: [
                const TextSpan(text: 'Research prototype on public NASA data — '),
                TextSpan(text: 'not a flight system', style: const TextStyle(color: kInkDim)),
                const TextSpan(text: '. Faults in the replay are simulated.'),
              ]),
              style: const TextStyle(fontSize: 11, color: kInkFaint),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const KeyHint(combo: '1–3', action: 'workspaces'),
        ]),
      );
}

// -----------------------------------------------------------------------------
// Mobile

class _MobileTopBar extends StatelessWidget {
  const _MobileTopBar({required this.link});
  final _Link link;

  @override
  Widget build(BuildContext context) {
    final (color, label) = switch (link) {
      _Link.online => (kGo, 'online'),
      _Link.checking => (kSignal, 'connecting'),
      _Link.offline => (kCaution, 'offline'),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: const BoxDecoration(color: kHull, border: Border(bottom: BorderSide(color: kSeam))),
      child: Row(children: [
        const BrandMark(compact: true),
        const Spacer(),
        LedDot(color: color, pulse: link != _Link.offline),
        const SizedBox(width: 8),
        Text(label.toUpperCase(), style: consoleLabel(color: kInkDim)),
      ]),
    );
  }
}

class _MobileTabs extends StatelessWidget {
  const _MobileTabs({required this.tab, required this.onSelect});
  final int tab;
  final ValueChanged<int> onSelect;

  @override
  Widget build(BuildContext context) => Container(
        color: kHull.withValues(alpha: 0.6),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(children: [
            for (int i = 0; i < _workspaces.length; i++)
              Padding(
                padding: const EdgeInsets.only(right: 6),
                child: GestureDetector(
                  onTap: () => onSelect(i),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
                    decoration: BoxDecoration(
                      color: i == tab ? kSignalSoft.withValues(alpha: 0.5) : Colors.transparent,
                      border: Border.all(color: i == tab ? kSignal.withValues(alpha: 0.5) : kSeam),
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(_workspaces[i].icon, size: 13, color: i == tab ? kSignal : kInkFaint),
                      const SizedBox(width: 8),
                      Text(_workspaces[i].label.toUpperCase(),
                          style: consoleLabel(color: i == tab ? kSignal : kInkDim)),
                    ]),
                  ),
                ),
              ),
          ]),
        ),
      );
}
