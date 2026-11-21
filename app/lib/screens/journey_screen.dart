import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../api.dart';
import '../journey3d.dart';
import '../theme.dart';
import '../widgets.dart';

const _auKm = 149597870.7;
const _speeds = [5.0, 12.0, 30.0]; // mission days per second of playback
const _months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];

enum _View {
  // yaw ≈ 1.9 puts the camera on the side of the Sun the ship flies through (around 20° ecliptic longitude)
  overview('Overview', 1.9, 0.68, 4.0),
  top('Top down', 0, 1.50, 4.3),
  follow('Follow ship', 1.9, 0.35, 0.9);

  const _View(this.label, this.yaw, this.elevation, this.dist);
  final String label;
  final double yaw, elevation, dist;
}

/// Perseverance's real 2020–21 cruise from Earth to Mars, drawn in 3D from NASA SPICE data.
class JourneyScreen extends StatefulWidget {
  const JourneyScreen({super.key, required this.api});
  final Api api;

  @override
  State<JourneyScreen> createState() => _JourneyScreenState();
}

class _JourneyScreenState extends State<JourneyScreen> with SingleTickerProviderStateMixin {
  _Journey? data;
  Object? error;

  late final Ticker _ticker = createTicker(_tick);
  Duration _last = Duration.zero;
  double clock = 0; // seconds, drives spin and the close-up
  double progress = 0; // fractional sample index
  bool playing = true;
  int speedIx = 1;

  final mesh = buildSpacecraftMesh();
  final stars = makeStars(900);
  final cam = OrbitCamera(yaw: _View.overview.yaw, elevation: _View.overview.elevation, dist: _View.overview.dist);
  _View view = _View.overview;
  bool easing = false; // camera gliding to a preset
  double _pinchStartDist = 0;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => error = null);
    try {
      final j = _Journey.fromJson(await widget.api.journey());
      if (!mounted) return;
      setState(() => data = j);
      _ticker.start();
    } catch (e) {
      if (mounted) setState(() => error = e);
    }
  }

  void _tick(Duration elapsed) {
    final dt = math.min(0.05, (elapsed - _last).inMicroseconds / 1e6);
    _last = elapsed;
    final d = data;
    if (d == null) return;
    setState(() {
      clock += dt;
      if (playing) {
        progress += dt * _speeds[speedIx] * d.samplesPerDay;
        if (progress >= d.n - 1) {
          progress = (d.n - 1).toDouble();
          playing = false;
        }
      }
      final k = 1 - math.exp(-dt * 5);
      final goal = view == _View.follow ? d.shipAt(progress) : V3.zero;
      cam.target = V3.lerp(cam.target, goal, view == _View.follow ? math.min(1, k * 2) : k);
      if (easing) {
        cam.yaw += _angleDiff(view.yaw, cam.yaw) * k;
        cam.elevation += (view.elevation - cam.elevation) * k;
        cam.dist += (view.dist - cam.dist) * k;
        if ((view.dist - cam.dist).abs() < 1e-3 && (view.elevation - cam.elevation).abs() < 1e-3) easing = false;
      }
    });
  }

  static double _angleDiff(double a, double b) {
    var d = (a - b) % (2 * math.pi);
    if (d > math.pi) d -= 2 * math.pi;
    return d;
  }

  void _setView(_View v) => setState(() {
        view = v;
        easing = true;
      });

  void _togglePlay() {
    final d = data;
    if (d == null) return;
    setState(() {
      if (!playing && progress >= d.n - 1) progress = 0;
      playing = !playing;
    });
  }

  void _seek(double p) => setState(() => progress = p.clamp(0, (data!.n - 1).toDouble()));

  @override
  Widget build(BuildContext context) {
    final d = data;
    if (d == null) return LoadingOrError(error: error, onRetry: _load);
    return LayoutBuilder(builder: (context, box) {
      final wide = box.maxWidth >= 980;
      final narrow = box.maxWidth < 620; // phone: stack overlays and controls
      final scene = _scene(d, compact: narrow);
      final controls = _controls(d, compact: narrow);
      final readouts = _readouts(d);
      final closeup = _closeup();
      if (wide) {
        return Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Expanded(
              child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                Expanded(child: scene),
                const SizedBox(width: 14),
                SizedBox(
                  width: 330,
                  child: SingleChildScrollView(
                    child: Column(children: [readouts, const SizedBox(height: 14), closeup]),
                  ),
                ),
              ]),
            ),
            const SizedBox(height: 14),
            controls,
          ]),
        );
      }
      return ListView(padding: const EdgeInsets.all(12), children: [
        SizedBox(height: math.min(480, box.maxWidth * 1.2), child: scene),
        const SizedBox(height: 12),
        controls,
        const SizedBox(height: 12),
        readouts,
        const SizedBox(height: 12),
        closeup,
      ]);
    });
  }

  // ---------------------------------------------------------------------------
  // 3D scene

  Widget _scene(_Journey d, {required bool compact}) {
    final i = progress.floor().clamp(0, d.n - 2);
    final f = progress - i;
    final frame = JourneyFrame(
      ship: d.shipAt(progress),
      shipFwd: d.ship[i + 1] - d.ship[i],
      earth: V3.lerp(d.earth[i], d.earth[i + 1], f),
      mars: V3.lerp(d.mars[i], d.mars[i + 1], f),
      path: d.ship,
      progress: progress,
      earthOrbit: d.earthOrbit,
      marsOrbit: d.marsOrbit,
      spin: clock * 0.6,
      // grows as the camera closes in, so the model is readable in Follow ship
      shipPx: (10 * math.sqrt(4.0 / cam.dist)).clamp(9.0, 24.0),
    );
    final arrived = progress >= d.n - 1;
    final viewButtons = Wrap(spacing: 6, runSpacing: 6, alignment: WrapAlignment.end, children: [
      for (final v in _View.values) _ViewButton(label: v.label, selected: view == v, onTap: () => _setView(v)),
    ]);

    return DecoratedBox(
      decoration: ShapeDecoration(color: kInset, shape: chamfer(), shadows: kPanelShadow),
      child: ClipPath(
        clipper: ShapeBorderClipper(shape: chamfer()),
        child: Stack(children: [
          Positioned.fill(
            child: Listener(
              onPointerSignal: (e) {
                if (e is! PointerScrollEvent) return;
                // claim the wheel so the page around the scene does not scroll as well
                GestureBinding.instance.pointerSignalResolver.register(e, (e) {
                  final dy = (e as PointerScrollEvent).scrollDelta.dy;
                  setState(() {
                    easing = false;
                    cam.dist = (cam.dist * math.exp(dy * 0.0015)).clamp(0.35, 12.0);
                  });
                });
              },
              child: GestureDetector(
                onScaleStart: (_) {
                  _pinchStartDist = cam.dist;
                  easing = false;
                },
                onScaleUpdate: (u) => setState(() {
                  cam.yaw -= u.focalPointDelta.dx * 0.008;
                  cam.elevation = (cam.elevation + u.focalPointDelta.dy * 0.006).clamp(0.04, 1.55);
                  if (u.pointerCount > 1 && u.scale > 0) {
                    cam.dist = (_pinchStartDist / u.scale).clamp(0.35, 12.0);
                  }
                }),
                child: MouseRegion(
                  cursor: SystemMouseCursors.grab,
                  child: CustomPaint(
                    painter: JourneyPainter(cam: cam, frame: () => frame, mesh: mesh, stars: stars),
                    size: Size.infinite,
                  ),
                ),
              ),
            ),
          ),
          // top-left: where and when
          Positioned(
            left: 16,
            top: 14,
            child: IgnorePointer(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(mainAxisSize: MainAxisSize.min, children: [
                  Text('DAY ${d.dayAt(progress).floor()}',
                      style: readout(size: compact ? 18 : 24, weight: FontWeight.w500, color: kSignal)),
                  const SizedBox(width: 10),
                  Text('OF ${d.day.last.round()}', style: readout(size: 11, color: kInkFaint)),
                ]),
                const SizedBox(height: 2),
                Text(d.dateAt(progress), style: readout(size: compact ? 11 : 12.5, color: kInkDim)),
                const SizedBox(height: 8),
                Tag(arrived ? 'Arrived · landing' : (d.phaseAt(progress) == 'approach' ? 'Approach' : 'Cruise'),
                    color: arrived ? kGo : (d.phaseAt(progress) == 'approach' ? kObserved : kSignal)),
              ]),
            ),
          ),
          // top-right: source badge, and the view presets (moved to the bottom on phones)
          Positioned(
            right: 12,
            top: 12,
            child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
              const EvidenceBadge(Evidence.nasa),
              if (!compact) ...[const SizedBox(height: 10), viewButtons],
            ]),
          ),
          // bottom: how to interact + honest scale note
          Positioned(
            left: compact ? 12 : 16,
            right: compact ? 12 : 16,
            bottom: 12,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
              IgnorePointer(
                child: Wrap(spacing: 18, runSpacing: 4, children: [
                  Text(compact ? 'DRAG TO TURN · PINCH TO ZOOM' : 'DRAG TO TURN · SCROLL TO ZOOM',
                      style: consoleLabel(size: 9.5)),
                  Text('Planets and ship drawn much larger than real; distances to scale',
                      style: readout(size: 10.5, color: kInkFaint)),
                ]),
              ),
              if (compact) ...[const SizedBox(height: 10), viewButtons],
            ]),
          ),
        ]),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Playback controls and timeline

  Widget _controls(_Journey d, {required bool compact}) {
    final arrived = progress >= d.n - 1;
    final play = IconButton.filled(
      tooltip: playing ? 'Pause' : (arrived ? 'Replay from launch' : 'Play'),
      onPressed: _togglePlay,
      icon: Icon(playing ? Icons.pause : (arrived ? Icons.replay : Icons.play_arrow)),
    );
    final timeline = _Timeline(journey: d, progress: progress, onSeek: _seek);
    final speed = Row(mainAxisSize: MainAxisSize.min, children: [
      Text('SPEED', style: consoleLabel(size: 9)),
      const SizedBox(width: 4),
      for (int k = 0; k < _speeds.length; k++)
        Padding(
          padding: const EdgeInsets.only(left: 4),
          child: _ViewButton(
              label: '${_speeds[k].toStringAsFixed(0)} d/s',
              selected: k == speedIx,
              onTap: () => setState(() => speedIx = k)),
        ),
    ]);
    return SectionCard(
      padding: 14,
      child: compact
          ? Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Row(children: [play, const SizedBox(width: 12), Expanded(child: timeline)]),
              const SizedBox(height: 10),
              Align(alignment: Alignment.centerRight, child: speed),
            ])
          : Row(children: [
              play,
              const SizedBox(width: 14),
              Expanded(child: timeline),
              const SizedBox(width: 18),
              speed,
            ]),
    );
  }

  // ---------------------------------------------------------------------------
  // Readouts

  Widget _readouts(_Journey d) {
    final p = progress;
    final fromEarth = d.lerp(d.distEarth, p);
    final toMars = d.lerp(d.distMars, p);
    final speed = d.lerp(d.speed, p);
    final light = d.lerp(d.lightMin, p);
    final travelled = d.lerp(d.cumKm, p);

    Widget row(String label, String value, {String? caption, Color color = kInk}) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 7),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(label.toUpperCase(), style: consoleLabel(size: 9.5)),
                if (caption != null) ...[
                  const SizedBox(height: 2),
                  Text(caption, style: const TextStyle(fontSize: 11, color: kInkFaint)),
                ],
              ]),
            ),
            const SizedBox(width: 10),
            Text(value, style: readout(size: 15, color: color, weight: FontWeight.w500)),
          ]),
        );

    return SectionCard(
      title: 'Flight data',
      icon: Icons.rocket_launch_outlined,
      evidence: const [Evidence.nasa],
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        row('From Earth', _km(fromEarth), color: kObserved),
        const SeamRule(),
        row('To Mars', _km(toMars), color: const Color(0xFFF08A5D)),
        const SeamRule(),
        row('Speed', '${speed.toStringAsFixed(1)} km/s',
            caption: '${_thousands(speed * 3600)} km/h, measured against the Sun'),
        const SeamRule(),
        row('Radio delay', '${light.toStringAsFixed(1)} min',
            caption: 'How long a command from Earth takes to arrive'),
        const SeamRule(),
        row('Travelled', _km(travelled), caption: 'of ${_km(d.totalKm)} along a curved path'),
        const SizedBox(height: 6),
        Text(
          'NASA did not fly in a straight line: the ship coasts on a curved orbit around the Sun '
          'and meets Mars where Mars will be 7 months later.',
          style: const TextStyle(fontSize: 12, color: kInkDim, height: 1.4),
        ),
      ]),
    );
  }

  Widget _closeup() {
    Widget part(Color c, String name, String what) => Padding(
          padding: const EdgeInsets.only(top: 6),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(margin: const EdgeInsets.only(top: 3), width: 9, height: 9, color: c),
            const SizedBox(width: 8),
            Expanded(
              child: Text.rich(TextSpan(children: [
                TextSpan(text: '$name  ', style: consoleLabel(color: kInk, size: 10, weight: FontWeight.w600)),
                TextSpan(text: what, style: const TextStyle(fontSize: 11.5, color: kInkDim)),
              ])),
            ),
          ]),
        );
    return SectionCard(
      title: 'The spacecraft',
      icon: Icons.satellite_alt_outlined,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        SizedBox(
          height: 170,
          child: ClipRect(
            child: CustomPaint(painter: SpacecraftCloseupPainter(mesh: mesh, angle: () => clock * 0.45)),
          ),
        ),
        const SizedBox(height: 8),
        part(const Color(0xFF8A5A3B), 'HEAT SHIELD', 'faces forward, takes the heat of entry'),
        part(const Color(0xFFD9DEE6), 'BACKSHELL', 'Perseverance rover folded up inside'),
        part(const Color(0xFF2E4C7A), 'CRUISE STAGE', 'solar panels and thrusters; let go before entry'),
        const SizedBox(height: 8),
        Text('Simplified shape for illustration.', style: readout(size: 10.5, color: kInkFaint)),
      ]),
    );
  }

  static String _km(double km) {
    if (km >= 1e6) return '${(km / 1e6).toStringAsFixed(km >= 1e8 ? 0 : 1)} million km';
    return '${_thousands(km)} km';
  }

  static String _thousands(double v) {
    final s = v.round().toString();
    final b = StringBuffer();
    for (int k = 0; k < s.length; k++) {
      if (k > 0 && (s.length - k) % 3 == 0) b.write(',');
      b.write(s[k]);
    }
    return b.toString();
  }
}

// -----------------------------------------------------------------------------

class _Journey {
  _Journey({
    required this.t,
    required this.day,
    required this.phase,
    required this.ship,
    required this.earth,
    required this.mars,
    required this.earthOrbit,
    required this.marsOrbit,
    required this.speed,
    required this.distEarth,
    required this.distMars,
    required this.lightMin,
    required this.totalKm,
  }) : cumKm = _cumulative(ship);

  factory _Journey.fromJson(Map<String, dynamic> j) {
    List<V3> v3s(String k) => [for (final p in j[k] as List) V3.fromList(p as List)];
    List<double> nums(String k) => [for (final x in j[k] as List) (x as num).toDouble()];
    return _Journey(
      t: [for (final s in j['t_utc'] as List) DateTime.parse('${s}Z')],
      day: nums('day'),
      phase: (j['phase'] as List).cast<String>(),
      ship: v3s('ship'),
      earth: v3s('earth'),
      mars: v3s('mars'),
      earthOrbit: v3s('earth_orbit'),
      marsOrbit: v3s('mars_orbit'),
      speed: nums('speed_sun_km_s'),
      distEarth: nums('dist_earth_km'),
      distMars: nums('dist_mars_km'),
      lightMin: nums('light_time_min'),
      totalKm: (j['total_path_km'] as num).toDouble(),
    );
  }

  final List<DateTime> t;
  final List<double> day, speed, distEarth, distMars, lightMin, cumKm;
  final List<String> phase;
  final List<V3> ship, earth, mars, earthOrbit, marsOrbit;
  final double totalKm;

  int get n => ship.length;
  double get samplesPerDay => (n - 1) / (day.last - day.first);

  static List<double> _cumulative(List<V3> pts) {
    final out = <double>[0];
    for (int i = 1; i < pts.length; i++) {
      out.add(out.last + (pts[i] - pts[i - 1]).length * _auKm);
    }
    return out;
  }

  double lerp(List<double> a, double p) {
    final i = p.floor().clamp(0, n - 2);
    return a[i] + (a[i + 1] - a[i]) * (p - i);
  }

  V3 shipAt(double p) {
    final i = p.floor().clamp(0, n - 2);
    return V3.lerp(ship[i], ship[i + 1], p - i);
  }

  double dayAt(double p) => lerp(day, p);
  String phaseAt(double p) => phase[p.round().clamp(0, n - 1)];

  String dateAt(double p) {
    final i = p.floor().clamp(0, n - 2);
    final ms = t[i].millisecondsSinceEpoch +
        ((t[i + 1].millisecondsSinceEpoch - t[i].millisecondsSinceEpoch) * (p - i)).round();
    final dt = DateTime.fromMillisecondsSinceEpoch(ms, isUtc: true);
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(dt.day)} ${_months[dt.month - 1]} ${dt.year} · ${two(dt.hour)}:${two(dt.minute)} UTC';
  }

  /// First sample of the approach phase.
  int get approachStart => phase.indexOf('approach');
}

/// Timeline with the cruise and approach phases, launch / approach / arrival marks; tap or drag to seek.
class _Timeline extends StatelessWidget {
  const _Timeline({required this.journey, required this.progress, required this.onSeek});
  final _Journey journey;
  final double progress;
  final ValueChanged<double> onSeek;

  @override
  Widget build(BuildContext context) {
    final d = journey;
    return LayoutBuilder(builder: (context, box) {
      double toP(double x) => (x / box.maxWidth).clamp(0, 1) * (d.n - 1);
      final narrow = box.maxWidth < 420;
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTapDown: (e) => onSeek(toP(e.localPosition.dx)),
        onHorizontalDragUpdate: (e) => onSeek(toP(e.localPosition.dx)),
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          child: SizedBox(
            height: 46,
            child: CustomPaint(
              painter: _TimelinePainter(
                frac: progress / (d.n - 1),
                approachFrac: d.approachStart < 0 ? 1 : d.approachStart / (d.n - 1),
                labels: narrow
                    ? const ['LAUNCH', 'APPROACH', 'MARS']
                    : const ['LAUNCH · 30 JUL 2020', 'APPROACH', 'MARS · 18 FEB 2021'],
              ),
            ),
          ),
        ),
      );
    });
  }
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter({required this.frac, required this.approachFrac, required this.labels});
  final double frac, approachFrac;
  final List<String> labels;

  @override
  void paint(Canvas canvas, Size size) {
    const y = 14.0;
    final w = size.width;
    final track = Paint()..color = kSeam;
    canvas.drawRect(Rect.fromLTWH(0, y - 2, w, 4), track);
    canvas.drawRect(
        Rect.fromLTWH(w * approachFrac, y - 2, w * (1 - approachFrac), 4), Paint()..color = kObserved.withValues(alpha: 0.25));
    canvas.drawRect(Rect.fromLTWH(0, y - 2, w * frac, 4), Paint()..color = kSignal);
    for (final (x, align) in [(0.0, 0), (approachFrac, 1), (1.0, 2)]) {
      canvas.drawRect(Rect.fromLTWH(w * x - (align == 2 ? 1 : 0), y - 7, 1, 14), Paint()..color = kSeamLit);
    }
    // handle
    final hx = w * frac;
    canvas.drawRect(Rect.fromCenter(center: Offset(hx, y), width: 10, height: 18), Paint()..color = kSignal);
    canvas.drawRect(Rect.fromCenter(center: Offset(hx, y), width: 2, height: 8), Paint()..color = kVoid);

    TextPainter text(String s) =>
        TextPainter(text: TextSpan(text: s, style: consoleLabel(size: 9)), textDirection: TextDirection.ltr)..layout();
    final start = text(labels[0]), mid = text(labels[1]), end = text(labels[2]);
    start.paint(canvas, const Offset(0, y + 12));
    end.paint(canvas, Offset(w - end.width, y + 12));
    // the middle mark is dropped when the bar is too short for all three labels
    final midX = (w * approachFrac - mid.width / 2).clamp(0.0, w - mid.width);
    if (midX > start.width + 8 && midX + mid.width < w - end.width - 8) {
      mid.paint(canvas, Offset(midX, y + 12));
    }
  }

  @override
  bool shouldRepaint(covariant _TimelinePainter old) => old.frac != frac || old.approachFrac != approachFrac;
}

class _ViewButton extends StatelessWidget {
  const _ViewButton({required this.label, required this.selected, required this.onTap});
  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Semantics(
        button: true,
        selected: selected,
        child: InkWell(
          onTap: onTap,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
            decoration: BoxDecoration(
              color: selected ? kSignalSoft.withValues(alpha: 0.8) : kPanel.withValues(alpha: 0.85),
              border: Border.all(color: selected ? kSignal.withValues(alpha: 0.6) : kSeamLit),
            ),
            child: Text(label.toUpperCase(), style: consoleLabel(color: selected ? kSignal : kInkDim, size: 9.5)),
          ),
        ),
      );
}
