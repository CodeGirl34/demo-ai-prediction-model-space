import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../api.dart';
import '../edl3d.dart';
import '../journey3d.dart' show OrbitCamera, V3;
import '../theme.dart';
import '../widgets.dart';

/// The flight system in 3D, configured for the current moment of the replay, with NASA's EDL
/// sequence (Entry · Descent · Landing) as a phase bar and a step list.
/// A stretch of the replay (seconds to touchdown), e.g. an injected sensor fault or an AI alarm.
typedef EdlMark = ({double start, double end, String label});

class EdlCard extends StatefulWidget {
  const EdlCard({
    super.key,
    required this.api,
    required this.mission,
    required this.now,
    required this.playing,
    required this.speed,
    required this.onSeek,
    this.compact = false,
    this.faults = const [],
    this.alarms = const [],
    this.wind,
    this.gusts = const [],
  });
  final Api api;
  final String mission;
  final double now; // replay time, seconds to touchdown
  final bool playing;
  final double speed; // data seconds per real second
  final ValueChanged<double> onSeek;
  final bool compact;

  /// Injected sensor faults and AI alarms. They change the sensor's reading, not the spacecraft,
  /// so the 3D model does not move differently: the view says when one is active instead.
  final List<EdlMark> faults;
  final List<EdlMark> alarms;

  /// Wind gusts we added. Unlike a sensor fault, a gust really moves the spacecraft: the stack
  /// leans in the wind and drifts off NASA's path.
  final EdlWind? wind;
  final List<EdlMark> gusts;

  @override
  State<EdlCard> createState() => _EdlCardState();
}

class _EdlCardState extends State<EdlCard> with SingleTickerProviderStateMixin {
  static final _cache = <String, EdlTimeline>{};

  EdlTimeline? tl;
  Object? error;
  late final Ticker _ticker = createTicker(_tick);
  Duration _last = Duration.zero;
  double clock = 0, shown = 0, epilogue = 0, _lastDrag = -10;
  EdlScene? scene;
  bool allSteps = false;
  final cam = OrbitCamera(yaw: 0.7, elevation: 0.30, dist: 9);

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(covariant EdlCard old) {
    super.didUpdateWidget(old);
    if (old.mission != widget.mission) _load();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final mission = widget.mission;
    setState(() => error = null);
    try {
      var t = _cache[mission];
      if (t == null) {
        final r = await Future.wait([widget.api.missions(), widget.api.trajectory(mission)]);
        final m = (r[0] as List).cast<Map<String, dynamic>>().firstWhere((x) => x['id'] == mission);
        t = _cache[mission] = EdlTimeline.build(mission, m['events'] as List, r[1] as Map<String, dynamic>);
      }
      if (!mounted || mission != widget.mission) return;
      setState(() {
        tl = t;
        shown = widget.now.clamp(t!.tStart, t.tEnd);
        epilogue = 0;
        scene = buildEdlScene(t, shown, clock);
      });
      if (!_ticker.isActive) _ticker.start();
    } catch (e) {
      if (mounted) setState(() => error = e);
    }
  }

  void _tick(Duration elapsed) {
    final dt = math.min(0.05, (elapsed - _last).inMicroseconds / 1e6);
    _last = elapsed;
    final l = tl;
    if (l == null) return;
    clock += dt;
    // follow the replay clock smoothly (it advances in 0.1 s steps)
    if (widget.playing) shown += dt * widget.speed;
    shown += (widget.now - shown) * (1 - math.exp(-dt * 6));
    if ((widget.now - shown).abs() > 30) shown = widget.now;
    if (!widget.playing && (widget.now - shown).abs() < 0.01) shown = widget.now;
    shown = shown.clamp(l.tStart, l.tEnd);
    // after the data ends, play the sky crane flyaway on its own
    epilogue = shown >= l.tEnd - 0.6 && widget.now >= l.tEnd - 1
        ? math.min(EdlTimeline.epilogueS, epilogue + dt)
        : 0;
    final ts = shown + epilogue, w = widget.wind;
    final s = buildEdlScene(l, ts, clock,
        wind: w?.windAt(ts) ?? V3.zero, swing: w?.swingAt(ts) ?? (0.0, 0.0));
    if (clock - _lastDrag > 2.5) cam.yaw += dt * 0.15;
    final k = 1 - math.exp(-dt * 3);
    cam.target = V3.lerp(cam.target, s.center, k);
    cam.dist += (s.radius * 3.1 - cam.dist) * k;
    setState(() => scene = s);
  }

  @override
  Widget build(BuildContext context) {
    final l = tl;
    if (l == null || scene == null) {
      return SectionCard(
        title: 'Spacecraft in 3D',
        icon: Icons.view_in_ar_outlined,
        child: SizedBox(height: 180, child: LoadingOrError(error: error, onRetry: _load)),
      );
    }
    final t = shown + epilogue;
    final phase = l.phaseAt(t);
    EdlMark? faultNow;
    for (final f in widget.faults) {
      if (f.start - 0.5 <= t && t <= f.end + 0.5) faultNow = f;
    }
    final alarmNow = widget.alarms.any((a) => a.start - 0.5 <= t && t <= a.end + 0.5);
    final windNow = widget.wind?.windAt(t) ?? V3.zero;
    final windH = math.sqrt(windNow.x * windNow.x + windNow.y * windNow.y);
    final gustNow = windH > 0.5;
    final (driftE, driftN) = widget.wind?.driftAt(t) ?? (0.0, 0.0);
    final drift = math.sqrt(driftE * driftE + driftN * driftN);
    final offCourse = drift > 0.5;
    final k = l.currentIndex(t);
    final ev = k >= 0 ? l.events[k] : null;
    final alt = scene!.altKm;

    return SectionCard(
      title: 'Spacecraft in 3D',
      icon: Icons.view_in_ar_outlined,
      evidence: const [Evidence.nasa],
      trailing: Tag(phase.label, color: phase.color),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        SizedBox(
          height: widget.compact ? 250 : 290,
          child: LayoutBuilder(builder: (context, box) => ClipRect(
            child: Stack(children: [
              Positioned.fill(
                child: GestureDetector(
                  onPanUpdate: (d) => setState(() {
                    cam.yaw -= d.delta.dx * 0.01;
                    cam.elevation = (cam.elevation + d.delta.dy * 0.006).clamp(-0.25, 1.35);
                    _lastDrag = clock;
                  }),
                  child: MouseRegion(
                    cursor: SystemMouseCursors.grab,
                    child: CustomPaint(painter: EdlPainter(cam: cam, scene: () => scene!, clock: () => clock)),
                  ),
                ),
              ),
              Positioned(
                left: 12,
                top: 10,
                right: 120,
                child: IgnorePointer(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(phase.label.toUpperCase(),
                        style: consoleLabel(color: phase.color, size: 12, weight: FontWeight.w700)),
                    const SizedBox(height: 3),
                    Text(ev?.name ?? 'Before entry', style: readout(size: 12.5)),
                  ]),
                ),
              ),
              Positioned(
                right: 12,
                top: 10,
                child: IgnorePointer(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                    Text(fmtT(t), style: readout(size: 12.5, color: kSignal)),
                    const SizedBox(height: 2),
                    Text(t >= 0 ? 'on the ground' : fmtAlt(alt), style: readout(size: 11, color: kInkDim)),
                    if (epilogue > 0) ...[
                      const SizedBox(height: 6),
                      const Tag('After the data ends', color: kInkDim),
                    ],
                  ]),
                ),
              ),
              if (faultNow == null && !alarmNow && !gustNow && !offCourse) // the badges need the room on a phone
                Positioned(
                  left: 12,
                  bottom: 8,
                  child: IgnorePointer(child: Text('DRAG TO TURN', style: consoleLabel(size: 9))),
                ),
              if (offCourse)
                Positioned(
                  left: 10,
                  bottom: 8,
                  child: IgnorePointer(child: _OffCourse(driftE, driftN, landed: t >= 0)),
                ),
              if (faultNow != null || alarmNow || gustNow)
                Positioned(
                  right: 10,
                  // on a phone the badges sit above the off-course map instead of on top of it
                  bottom: offCourse && box.maxWidth < 420 ? 72 : 8,
                  child: IgnorePointer(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                      if (gustNow) ...[
                        _Badge('WIND ${windH.toStringAsFixed(0)} m/s → ${compass(math.atan2(windNow.x, windNow.y) * 180 / math.pi)}',
                            'a real push: the sensors all work', kWind),
                        if (faultNow != null || alarmNow) const SizedBox(height: 6),
                      ],
                      if (faultNow != null)
                        _Badge('SENSOR FAULT · ${faultNow.label.toUpperCase()}',
                            'the reading is wrong, the spacecraft is fine', kScenario),
                      if (faultNow != null && alarmNow) const SizedBox(height: 6),
                      if (alarmNow) const _Badge('AI ALARM', 'the AI flags this reading', kCaution),
                    ]),
                  ),
                ),
            ]),
          )),
        ),
        const SizedBox(height: 12),
        _PhaseBar(tl: l, now: t, faults: widget.faults, gusts: widget.gusts),
        const SizedBox(height: 12),
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(
            child: Text(ev?.what ?? phase.what, style: const TextStyle(fontSize: 12.5, color: kInk, height: 1.4)),
          ),
          if (ev != null) ...[const SizedBox(width: 8), Tag(ev.source.label, color: ev.source.color)],
        ]),
        const SizedBox(height: 6),
        Text(phase.what, style: const TextStyle(fontSize: 11.5, color: kInkFaint, height: 1.4)),
        const SizedBox(height: 12),
        const SeamRule(),
        const SizedBox(height: 10),
        Row(children: [
          Expanded(
            child: Text('THE ${l.events.length} STEPS OF MARS ENTRY, DESCENT AND LANDING',
                style: consoleLabel(size: 9.5)),
          ),
          TextButton(
            onPressed: () => setState(() => allSteps = !allSteps),
            child: Text(allSteps ? 'Show fewer' : 'Show all'),
          ),
        ]),
        for (int i = 0; i < l.events.length; i++)
          if (allSteps || (i >= k - 1 && i <= k + 3) || (k < 0 && i < 5))
            _EventRow(
              index: i,
              event: l.events[i],
              state: i < k ? _Step.done : (i == k ? _Step.now : _Step.next),
              color: l.phaseAt(l.events[i].t).color,
              onTap: () => widget.onSeek(l.events[i].t),
            ),
        const SizedBox(height: 10),
        Text(
          'Sequence from NASA’s “Mars Entry, Descent, and Landing” overview (2025), placed on the reconstructed '
          'landing. Tap a step to jump to it. Shapes are simplified, and the parachute is drawn smaller and closer '
          'than the real 21.5 m one.',
          style: readout(size: 10, color: kInkFaint),
        ),
      ]),
    );
  }
}

enum _Step { done, now, next }

class _EventRow extends StatelessWidget {
  const _EventRow(
      {required this.index, required this.event, required this.state, required this.color, required this.onTap});
  final int index;
  final EdlEvent event;
  final _Step state;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final now = state == _Step.now;
    final fg = switch (state) { _Step.done => kInkDim, _Step.now => kInk, _Step.next => kInkFaint };
    return InkWell(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 3),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          color: now ? kSignalSoft.withValues(alpha: 0.55) : Colors.transparent,
          border: Border(left: BorderSide(color: state == _Step.next ? kSeam : color, width: 3)),
        ),
        child: Row(children: [
          SizedBox(
            width: 22,
            child: state == _Step.done
                ? Icon(Icons.check, size: 13, color: color)
                : Text('${index + 1}', style: readout(size: 10.5, color: now ? kSignal : kInkFaint)),
          ),
          Expanded(
            child: Text(event.name,
                style: TextStyle(fontSize: 12.5, color: fg, fontWeight: now ? FontWeight.w600 : FontWeight.w400)),
          ),
          Text(fmtT(event.t), style: readout(size: 10.5, color: kInkFaint)),
          const SizedBox(width: 8),
          SizedBox(
            width: 62,
            child: Text(event.source.label.toUpperCase(),
                textAlign: TextAlign.right, style: consoleLabel(size: 8.5, color: event.source.color)),
          ),
        ]),
      ),
    );
  }
}

/// Approach · Entry · Descent · Landing, equal widths (not to time scale), with event ticks.
class _Badge extends StatelessWidget {
  const _Badge(this.title, this.sub, this.color);
  final String title, sub;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.fromLTRB(8, 5, 8, 5),
        decoration: BoxDecoration(
          color: kPanel.withValues(alpha: 0.85),
          border: Border(left: BorderSide(color: color, width: 3)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.end, mainAxisSize: MainAxisSize.min, children: [
          Text(title, style: consoleLabel(color: color, size: 10, weight: FontWeight.w700)),
          const SizedBox(height: 2),
          Text(sub, style: readout(size: 10, color: kInkDim)),
        ]),
      );
}

/// Where the wind has pushed the spacecraft: NASA's path in the middle, the spacecraft as a dot.
class _OffCourse extends StatelessWidget {
  const _OffCourse(this.e, this.n, {required this.landed});
  final double e, n;
  final bool landed;

  @override
  Widget build(BuildContext context) {
    final d = math.sqrt(e * e + n * n);
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 6, 8, 6),
      decoration: BoxDecoration(color: kPanel.withValues(alpha: 0.85), border: const Border(left: BorderSide(color: kWind, width: 3))),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        SizedBox(width: 46, height: 46, child: CustomPaint(painter: _OffCoursePainter(e, n))),
        const SizedBox(width: 8),
        Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
          Text('${d.toStringAsFixed(d < 10 ? 1 : 0)} m', style: readout(size: 13, color: kWind, weight: FontWeight.w600)),
          Text(landed ? 'landed off\nNASA\'s spot' : 'off NASA\'s\npath', style: readout(size: 9.5, color: kInkDim)),
        ]),
      ]),
    );
  }
}

class _OffCoursePainter extends CustomPainter {
  _OffCoursePainter(this.e, this.n);
  final double e, n;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero), r = size.width / 2 - 2;
    final d = math.sqrt(e * e + n * n);
    final scale = const [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0].firstWhere((s) => s >= d, orElse: () => d);
    final ring = Paint()
      ..color = kInkFaint.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.drawCircle(c, r, ring);
    canvas.drawCircle(c, r / 2, ring);
    final cross = Paint()..color = kInk.withValues(alpha: 0.8)..strokeWidth = 1.2;
    canvas.drawLine(c + const Offset(-4, 0), c + const Offset(4, 0), cross);
    canvas.drawLine(c + const Offset(0, -4), c + const Offset(0, 4), cross);
    final p = c + Offset(e / scale * r, -n / scale * r);
    canvas.drawLine(c, p, Paint()..color = kWind.withValues(alpha: 0.6)..strokeWidth = 1);
    canvas.drawCircle(p, 3.2, Paint()..color = kWind);
    final tp = TextPainter(
        text: TextSpan(text: '${scale.toStringAsFixed(0)} m', style: consoleLabel(size: 7.5)),
        textDirection: TextDirection.ltr)
      ..layout();
    tp.paint(canvas, Offset(size.width - tp.width, size.height - tp.height + 2));
  }

  @override
  bool shouldRepaint(covariant _OffCoursePainter old) => old.e != e || old.n != n;
}

class _PhaseBar extends StatelessWidget {
  const _PhaseBar({required this.tl, required this.now, this.faults = const [], this.gusts = const []});
  final EdlTimeline tl;
  final double now;
  final List<EdlMark> faults, gusts;

  @override
  Widget build(BuildContext context) =>
      SizedBox(height: 40, child: CustomPaint(painter: _PhaseBarPainter(tl, now, faults, gusts), size: Size.infinite));
}

class _PhaseBarPainter extends CustomPainter {
  _PhaseBarPainter(this.tl, this.now, this.faults, this.gusts);
  final EdlTimeline tl;
  final double now;
  final List<EdlMark> faults, gusts;

  @override
  void paint(Canvas canvas, Size size) {
    final b = [tl.tStart, tl.at('entry'), tl.at('parachute'), tl.at('rover_dep'), tl.at('touchdown') + EdlTimeline.epilogueS];
    final w = size.width / 4;
    double x(double t) {
      if (t <= b.first) return 0;
      for (int i = 0; i < 4; i++) {
        if (t < b[i + 1]) return (i + (t - b[i]) / (b[i + 1] - b[i])) * w;
      }
      return size.width;
    }

    const y = 10.0;
    final current = tl.phaseAt(now);
    for (int i = 0; i < 4; i++) {
      final p = EdlPhase.values[i];
      final alpha = p == current ? 0.9 : (p.index < current.index ? 0.45 : 0.15);
      canvas.drawRect(Rect.fromLTWH(i * w + 1, y - 4, w - 2, 8), Paint()..color = p.color.withValues(alpha: alpha));
      final tp = TextPainter(
          text: TextSpan(
              text: p.label.toUpperCase(),
              style: consoleLabel(size: 9, color: p == current ? p.color : kInkFaint, weight: FontWeight.w600)),
          textDirection: TextDirection.ltr)
        ..layout(maxWidth: w);
      tp.paint(canvas, Offset(i * w + (w - tp.width) / 2, y + 10));
    }
    for (final e in tl.events) {
      final ex = x(e.t);
      canvas.drawRect(Rect.fromLTWH(ex - 0.5, y - 7, 1, 14), Paint()..color = kInk.withValues(alpha: 0.55));
    }
    // wind gusts, then injected faults, at least 3 px wide so a short one still shows
    for (final g in gusts) {
      final x0 = x(g.start), x1 = x(g.end);
      canvas.drawRect(Rect.fromCenter(center: Offset((x0 + x1) / 2, y), width: math.max(3.0, x1 - x0), height: 8),
          Paint()..color = kWind);
    }
    for (final f in faults) {
      final x0 = x(f.start), x1 = x(f.end);
      final c = (x0 + x1) / 2, w = math.max(3.0, x1 - x0);
      canvas.drawRect(Rect.fromCenter(center: Offset(c, y), width: w, height: 8), Paint()..color = kScenario);
    }
    final cx = x(now);
    canvas.drawRect(Rect.fromCenter(center: Offset(cx, y), width: 4, height: 18), Paint()..color = kSignal);
  }

  @override
  bool shouldRepaint(covariant _PhaseBarPainter old) =>
      old.now != now || old.tl != tl || old.faults != faults || old.gusts != gusts;
}
