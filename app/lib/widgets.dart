import 'package:flutter/material.dart';

import 'theme.dart';

/// Where a number on screen comes from. Every panel says it, so nobody mistakes a
/// simulated fault or a model guess for something NASA measured.
enum Evidence {
  nasa('NASA data', kObserved, kObservedSoft, 'Measured or reconstructed by NASA'),
  ai('AI output', kDerived, kDerivedSoft, 'Produced by our models'),
  simulated('Simulated', kScenario, kScenarioSoft, 'Faults we injected on purpose to test the AI');

  const Evidence(this.label, this.color, this.soft, this.meaning);
  final String label;
  final Color color;
  final Color soft;
  final String meaning;
}

class EvidenceBadge extends StatelessWidget {
  const EvidenceBadge(this.evidence, {super.key});
  final Evidence evidence;

  @override
  Widget build(BuildContext context) => Tooltip(
        message: evidence.meaning,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
          decoration: BoxDecoration(
            color: evidence.soft,
            border: Border.all(color: evidence.color.withValues(alpha: 0.45)),
          ),
          child: Text(evidence.label.toUpperCase(), style: consoleLabel(color: evidence.color, size: 9.5)),
        ),
      );
}

/// Instrument panel: chamfered corner, thin seam border, uppercase header, lit rule.
class SectionCard extends StatelessWidget {
  const SectionCard({
    super.key,
    required this.child,
    this.title,
    this.trailing,
    this.evidence = const [],
    this.icon,
    this.padding = 16,
  });
  final Widget child;
  final String? title;
  final Widget? trailing;
  final List<Evidence> evidence;
  final IconData? icon;
  final double padding;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: ShapeDecoration(color: kPanel, shape: chamfer(), shadows: kPanelShadow),
      child: Padding(
        padding: EdgeInsets.all(padding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (title != null) ...[
              Row(children: [
                if (icon != null) ...[Icon(icon, size: 14, color: kSignal), const SizedBox(width: 8)],
                Expanded(
                  child: Text(title!.toUpperCase(),
                      style: consoleLabel(color: kInk, size: 11.5, weight: FontWeight.w600)),
                ),
                for (final e in evidence) ...[const SizedBox(width: 6), EvidenceBadge(e)],
                if (trailing != null) ...[const SizedBox(width: 8), trailing!],
              ]),
              const SizedBox(height: 10),
              const SeamRule(),
              const SizedBox(height: 12),
            ],
            child,
          ],
        ),
      ),
    );
  }
}

/// Fine horizontal rule with a lit left edge (RedSight `.rule-seam`).
class SeamRule extends StatelessWidget {
  const SeamRule({super.key});

  @override
  Widget build(BuildContext context) => Container(
        height: 1,
        decoration: const BoxDecoration(
          gradient: LinearGradient(colors: [kSeamLit, kSeam], stops: [0, 0.6]),
        ),
      );
}

/// Status LED, optionally pulsing.
class LedDot extends StatefulWidget {
  const LedDot({super.key, required this.color, this.pulse = false, this.size = 7});
  final Color color;
  final bool pulse;
  final double size;

  @override
  State<LedDot> createState() => _LedDotState();
}

class _LedDotState extends State<LedDot> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 2400));

  @override
  void initState() {
    super.initState();
    _sync();
  }

  @override
  void didUpdateWidget(LedDot old) {
    super.didUpdateWidget(old);
    _sync();
  }

  void _sync() {
    final reduce = WidgetsBinding.instance.platformDispatcher.accessibilityFeatures.disableAnimations;
    if (widget.pulse && !reduce) {
      if (!_c.isAnimating) _c.repeat();
    } else {
      _c.stop();
      _c.value = 0;
    }
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
        animation: _c,
        builder: (context, _) {
          final v = _c.value;
          final opacity = 1 - 0.55 * (v < 0.5 ? v * 2 : (1 - v) * 2);
          return Container(
            width: widget.size,
            height: widget.size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: widget.color.withValues(alpha: opacity),
              boxShadow: [BoxShadow(color: widget.color.withValues(alpha: 0.5 * opacity), blurRadius: 6)],
            ),
          );
        },
      );
}

class Metric extends StatelessWidget {
  const Metric({super.key, required this.label, required this.value, this.color, this.caption});
  final String label;
  final String value;
  final Color? color;
  final String? caption;

  @override
  Widget build(BuildContext context) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      Text(label.toUpperCase(), style: consoleLabel(size: 10)),
      const SizedBox(height: 5),
      Text(value, style: readout(size: 22, weight: FontWeight.w500, color: color ?? kInk)),
      if (caption != null) ...[
        const SizedBox(height: 2),
        Text(caption!, style: readout(size: 11, color: kInkFaint)),
      ],
    ]);
  }
}

class PhaseChip extends StatelessWidget {
  const PhaseChip(this.phase, {super.key, this.large = false});
  final String phase;
  final bool large;

  @override
  Widget build(BuildContext context) {
    final c = phaseColors[phase] ?? kInkFaint;
    return Container(
      padding: EdgeInsets.symmetric(horizontal: large ? 12 : 8, vertical: large ? 7 : 3),
      decoration: ShapeDecoration(
        color: c.withValues(alpha: 0.12),
        shape: chamfer(cut: large ? 6 : 4, side: c.withValues(alpha: 0.7)),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        LedDot(color: c, size: large ? 7 : 5),
        SizedBox(width: large ? 8 : 6),
        Text(phaseLabel(phase).toUpperCase(),
            style: consoleLabel(color: c, size: large ? 13 : 10, weight: FontWeight.w600)),
      ]),
    );
  }
}

/// Small squared status badge.
class Tag extends StatelessWidget {
  const Tag(this.text, {super.key, required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.12),
          border: Border.all(color: color.withValues(alpha: 0.45)),
        ),
        child: Text(text.toUpperCase(), style: consoleLabel(color: color, size: 9.5, weight: FontWeight.w600)),
      );
}

/// Keyboard hint, e.g. [1–3] workspaces.
class KeyHint extends StatelessWidget {
  const KeyHint({super.key, required this.combo, required this.action});
  final String combo;
  final String action;

  @override
  Widget build(BuildContext context) => Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
          decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeamLit)),
          child: Text(combo, style: readout(size: 10, color: kInkDim)),
        ),
        const SizedBox(width: 6),
        Text(action.toUpperCase(), style: consoleLabel(size: 9.5)),
      ]);
}

class Legend extends StatelessWidget {
  const Legend(this.items, {super.key});
  final List<(String, Color)> items;

  @override
  Widget build(BuildContext context) => Wrap(spacing: 12, runSpacing: 4, children: [
        for (final (l, c) in items)
          Row(mainAxisSize: MainAxisSize.min, children: [
            Container(width: 9, height: 9, color: c.withValues(alpha: 0.8)),
            const SizedBox(width: 5),
            Text(l.toUpperCase(), style: consoleLabel(size: 9.5)),
          ]),
      ]);
}

class LoadingOrError extends StatelessWidget {
  const LoadingOrError({super.key, this.error, this.onRetry});
  final Object? error;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    if (error == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const LedDot(color: kSignal, pulse: true, size: 9),
            const SizedBox(height: 12),
            Text('RUNNING MODELS…', style: consoleLabel(color: kInkDim)),
          ]),
        ),
      );
    }
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const LedDot(color: kCaution, size: 9),
          const SizedBox(height: 12),
          Text('Could not reach the API', style: consoleLabel(color: kInk, size: 13, weight: FontWeight.w600)),
          const SizedBox(height: 6),
          Text('$error', textAlign: TextAlign.center, style: readout(size: 11, color: kInkFaint)),
          if (onRetry != null) ...[
            const SizedBox(height: 14),
            FilledButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ]),
      ),
    );
  }
}

/// Coloured strip showing phases along the time axis.
class PhaseStrip extends StatelessWidget {
  const PhaseStrip({super.key, required this.phases, this.height = 12, this.upTo});
  final List<String> phases;
  final double height;
  final int? upTo;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(color: kInset, border: Border.all(color: kSeam)),
      child: CustomPaint(painter: _StripPainter(phases, upTo), size: Size.infinite),
    );
  }
}

class _StripPainter extends CustomPainter {
  _StripPainter(this.phases, this.upTo);
  final List<String> phases;
  final int? upTo;

  @override
  void paint(Canvas canvas, Size size) {
    if (phases.isEmpty) return;
    final n = phases.length;
    final w = size.width / n;
    int start = 0;
    for (int i = 1; i <= n; i++) {
      if (i == n || phases[i] != phases[start]) {
        final visible = upTo == null || start <= upTo!;
        final c = (phaseColors[phases[start]] ?? kInkFaint).withValues(alpha: visible ? 0.85 : 0.14);
        canvas.drawRect(Rect.fromLTWH(start * w, 0, (i - start) * w + 0.5, size.height), Paint()..color = c);
        start = i;
      }
    }
  }

  @override
  bool shouldRepaint(covariant _StripPainter old) => old.phases != phases || old.upTo != upTo;
}

/// Shared chart styling: seam grid, faint mono axis labels.
const kChartGrid = Color(0xFF1E2733);
TextStyle chartAxis() => readout(size: 9.5, color: kInkFaint);
